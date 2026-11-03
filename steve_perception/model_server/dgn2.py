"""
dgn2.py — DexGraspNet 2.0 (PKU-EPIC/DexGraspNet2, CoRL 2024; LEAP hand, CC BY-NC 4.0) on one depth image.

Runs in the dgn2_env interpreter (Python 3.8, torch 2.0.1 + CUDA 11.7, MinkowskiEngine, pytorch3d 0.7.5, see
envs/dgn2_env.yml), inside server.py. The repo's own code does the work, as its src/eval/predict_dexterous.py:

  depth (+ mask of the area to grasp in) -> camera-frame points (optical frame: x right, y down, z forward),
  40000 sampled with replacement (as the training data) -> MinkowskiEngine sparse tensor (voxel 5 mm) ->
  GraspnessSample.sample(): graspness -> seed points (FPS) -> diffusion wrist pose + MLP joint angles,
  scored by log-probability + 5 x graspness. Grasps on depth edges are suppressed (`edge`, Canny on the
  depth image, as the training data).

Every grasp comes back in the camera frame with what the robot side needs to retarget it to another hand
(steve_grasping.retarget): the LEAP fingertip pads (index, middle, ring, thumb; RobotModel.get_ft_center) and
the knuckles (mean of the three finger MCP links) from the repo's forward kinematics.
"""

import os
import sys
import time
import types
from pathlib import Path

import numpy as np

TIPS = ("fingertip", "fingertip_2", "fingertip_3", "thumb_fingertip")   # index, middle, ring, thumb
KNUCKLES = ("mcp_joint", "mcp_joint_2", "mcp_joint_3")


class DexGraspNet2:
    def __init__(self, repo, ckpt, device="cuda", urdf=None, meta=None):
        repo = Path(repo).expanduser().resolve()
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))
        # graspnetAPI: imported by the repo's dataset visualiser (src/utils/vis_plotly.py) for GraspNet-1B scene files,
        # never used for inference -> stubbed instead of installed (it pulls open3d, autolab, cvxopt, ...)
        for name in ("graspnetAPI", "graspnetAPI.utils", "graspnetAPI.utils.utils", "graspnetAPI.utils.xmlhandler"):
            sys.modules.setdefault(name, types.ModuleType(name))
        sys.modules["graspnetAPI.utils.utils"].parse_posevector = None
        sys.modules["graspnetAPI.utils.xmlhandler"].xmlReader = None
        # open3d: src/network/backbones/conv.py imports it (and refuses to load without it) but never uses it
        if "open3d" not in sys.modules:
            try:
                import open3d  # noqa: F401  (installed: fine)
            except ImportError:
                o3d = types.ModuleType("open3d")
                o3d.__getattr__ = lambda name: (_ for _ in ()).throw(
                    ImportError(f"open3d.{name}: open3d is stubbed in dgn2.py (not installed in dgn2_env)"))
                sys.modules["open3d"] = o3d
        # torch 2.0.1 rejects allocator options of later versions (expandable_segments) at the first CUDA use
        conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
        if "expandable_segments" in conf:
            kept = ",".join(o for o in conf.split(",") if o and not o.startswith("expandable_segments"))
            if kept:
                os.environ["PYTORCH_CUDA_ALLOC_CONF"] = kept
            else:
                os.environ.pop("PYTORCH_CUDA_ALLOC_CONF")
        cwd = os.getcwd()
        os.chdir(str(repo))  # the repo resolves robot_models/, configs/ relative to its root
        try:
            import torch
            from src.network.model import get_model
            from src.utils.config import ckpt_to_config
            from src.utils.dataset import get_sparse_tensor
            from src.utils.edge import detect_edge
            from src.utils.robot_model import RobotModel
            t0 = time.monotonic()
            self.torch, self.get_sparse_tensor, self.detect_edge = torch, get_sparse_tensor, detect_edge
            self.device = torch.device(device)
            ckpt = str(Path(ckpt).expanduser().resolve())
            config = ckpt_to_config(ckpt)  # <experiment>/config.yaml next to ckpt/
            self.model = get_model(config.model)
            self.model.config.voxel_size = self.voxel = config.data.voxel_size
            self.num_points = int(config.data.num_points or 40000)
            state = torch.load(ckpt, map_location="cpu")
            self.model.load_state_dict(state["model"], strict=False)
            self.model.to(self.device).eval()
            self.robot = RobotModel(str(urdf or repo / "robot_models/urdf/leap_hand_simplified.urdf"),
                                    str(meta or repo / "robot_models/meta/leap_hand/meta.yaml"))
            self.joint_names = list(self.robot.joint_names)
        finally:
            os.chdir(cwd)
        print(f"[model_server] DexGraspNet 2.0 ready in {time.monotonic() - t0:.1f} s ({Path(ckpt).name}, "
              f"{len(self.joint_names)} joints)", flush=True)

    def grasps(self, depth, K, mask=None, z_range=(0.15, 1.5), n_seeds=128, graspness_scale=5.0, top_k=32,
               seed=0):
        """Best `top_k` of `n_seeds` grasps (one per seed point) for a depth image [m] with intrinsics K; only
        points in `mask` (HxW bool, e.g. the table area around the target) are the scene."""
        torch = self.torch
        D = np.asarray(depth, np.float32)
        D = np.where(np.isfinite(D), D, 0.0)
        K = np.asarray(K, float).reshape(3, 3)
        h, w = D.shape
        v, u = np.mgrid[0:h, 0:w]
        ok = (D > z_range[0]) & (D < z_range[1])
        if mask is not None:
            ok &= np.asarray(mask, bool)
        if ok.sum() < 500:
            raise ValueError(f"only {int(ok.sum())} depth points in the mask / z_range {list(z_range)}")
        pc = np.stack([(u[ok] - K[0, 2]) * D[ok] / K[0, 0], (v[ok] - K[1, 2]) * D[ok] / K[1, 1], D[ok]], 1)
        # edges as in the training data: Canny on the depth image scaled to 0..200, grown by 5 px
        img = D / max(float(D.max()), 1e-6) * 200.0
        edge = self.detect_edge(img.astype(np.uint8))[ok] > 0
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(pc), self.num_points, replace=True)
        pc_t = torch.tensor(pc[idx][None], dtype=torch.float)
        edge_t = torch.tensor(edge[idx][None].astype(np.int64))
        with torch.no_grad():
            data = self.get_sparse_tensor(pc_t, self.voxel)
            data["seg"] = torch.zeros_like(edge_t)
            data = {k: x.to(self.device) for k, x in data.items()}
            rot, trans, qpos, score = (x[0].cpu() for x in self.model.sample(
                data, int(n_seeds), graspness_scale=graspness_scale, allow_fail=True, cate=False,
                edge=edge_t.to(self.device))[:4])
            order = torch.argsort(score, descending=True)[:int(top_k)]
            rot, trans, qpos, score = rot[order], trans[order], qpos[order], score[order]
            q = {n: qpos[:, i].float() for i, n in enumerate(self.joint_names)}
            lt, lr = self.robot.forward_kinematics(q)
            try:
                tips = torch.stack([self.robot.get_ft_center(lt, lr, n) for n in TIPS], 1)
            except Exception:  # no collision boxes for the pads: the link origins
                tips = torch.stack([lt[n] for n in TIPS], 1)
            knuck = torch.stack([lt[n] for n in KNUCKLES], 1).mean(1)
            # hand frame -> camera frame
            tips = torch.einsum("nab,nkb->nka", rot.float(), tips.float()) + trans[:, None].float()
            knuck = torch.einsum("nab,nb->na", rot.float(), knuck.float()) + trans.float()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        return {"rotation": rot.numpy().astype(float), "translation": trans.numpy().astype(float),
                "qpos": qpos.numpy().astype(float), "score": score.numpy().astype(float),
                "tips": tips.numpy().astype(float), "knuckles": knuck.numpy().astype(float),
                "joint_names": self.joint_names, "n_points": int(ok.sum())}


def _selftest():
    """python dgn2.py --repo R --ckpt C: the model on a synthetic view (camera 0.6 m above a table with a 5 cm ball
    and a 6 x 3 x 4 cm box), no ROS, no server — checks the env, the checkpoint and the GPU."""
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--repo", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    m = DexGraspNet2(a.repo, a.ckpt, a.device)
    h, w, f = 480, 640, 600.0
    K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]])
    v, u = np.mgrid[0:h, 0:w]
    x, y = (u - w / 2) / f, (v - h / 2) / f            # ray directions (optical frame, z = 1)
    D = np.full((h, w), 0.60, np.float32)             # table plane
    for cx, cy in ((0.0, 0.0),):                       # ball r 2.5 cm resting on the table, under the camera
        cz, r = 0.575, 0.025                          # its centre 0.575 m from the camera
        b = x * cx + y * cy + cz
        c = cx ** 2 + cy ** 2 + cz ** 2 - r ** 2
        n2 = x ** 2 + y ** 2 + 1.0
        disc = b ** 2 - n2 * c
        hit = disc > 0
        D[hit] = np.minimum(D[hit], ((b - np.sqrt(np.maximum(disc, 0))) / n2)[hit])
    box = (np.abs(x * 0.56 - 0.10) < 0.03) & (np.abs(y * 0.56) < 0.015)   # box top 4 cm above the table
    D[box] = 0.56
    t0 = time.monotonic()
    r = m.grasps(D, K)
    print(f"selftest: {len(r['score'])} grasp(s) in {time.monotonic() - t0:.2f} s from {r['n_points']} points; "
          f"best score {r['score'][0]:.2f}, wrist at {np.round(r['translation'][0], 3)} m (camera frame), "
          f"thumb tip {np.round(r['tips'][0][3], 3)}")


if __name__ == "__main__":
    _selftest()
