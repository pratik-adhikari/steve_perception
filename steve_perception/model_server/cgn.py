"""
cgn.py — Contact-GraspNet (PyTorch port, elchun/contact_graspnet_pytorch) on one depth image.

Runs in the vgn_env interpreter, inside server.py. The port is pure PyTorch (PointNet++ without
compiled ops); its data module imports a pyrender scene renderer that inference never uses,
so that module is stubbed out instead of installing pyrender/OpenGL.

Output grasps are Contact-GraspNet's (Panda hand) in the camera optical frame: origin = gripper
base, +z = approach, +x = finger closing axis, fingertip centre 0.1034 m along +z
(grasp_test/cgn_client.py converts them to the WBC's pinch frame).
"""

import sys
import time
import types
from pathlib import Path

import numpy as np


class ContactGraspNet:
    def __init__(self, repo, ckpt_dir=None, device="cuda"):
        repo = Path(repo)
        for p in (repo, repo / "Pointnet_Pointnet2_pytorch"):
            if str(p) not in sys.path:
                sys.path.insert(0, str(p))
        stub = types.ModuleType("contact_graspnet_pytorch.scene_renderer")
        stub.SceneRenderer = None  # training-data renderer (pyrender): not needed for inference
        sys.modules.setdefault("contact_graspnet_pytorch.scene_renderer", stub)
        import torch
        from contact_graspnet_pytorch import config_utils
        from contact_graspnet_pytorch.checkpoints import CheckpointIO
        from contact_graspnet_pytorch.contact_grasp_estimator import GraspEstimator

        t0 = time.monotonic()
        self.torch = torch
        ckpt_dir = Path(ckpt_dir) if ckpt_dir else repo / "checkpoints" / "contact_graspnet"
        self.cfg = config_utils.load_config(str(ckpt_dir), batch_size=1, arg_configs=[])
        self.est = GraspEstimator(self.cfg)
        CheckpointIO(checkpoint_dir=str(ckpt_dir / "checkpoints"), model=self.est.model).load("model.pt")
        self.est.model.eval()
        print(f"[model_server] Contact-GraspNet ready in {time.monotonic() - t0:.1f} s ({ckpt_dir.name})",
              flush=True)

    def grasps(self, depth, K, segmap=None, rgb=None, z_range=(0.2, 1.5), forward_passes=1,
               local_regions=True, filter_grasps=True, segmap_id=0):
        """DiNBV-Grasp reply dict: {pred_grasps_cam, scores, contact_pts, gripper_openings} per segment id
        (-1: whole scene when no segmap). With a segmap, only grasps whose contact lies on the segment."""
        depth = np.asarray(depth, np.float32)
        if depth.ndim == 3:  # a burst of frames: per-pixel median (less RealSense noise)
            depth = np.median(depth, axis=0)
        depth = np.where(np.isfinite(depth), depth, 0.0).astype(np.float32)
        K = np.asarray(K, float).reshape(3, 3)
        seg = None if segmap is None else np.asarray(segmap, np.int32)
        if seg is not None and not seg.any():
            seg = None
        local_regions = bool(local_regions and seg is not None)
        filter_grasps = bool(filter_grasps and seg is not None)
        try:
            with self.torch.no_grad():
                pc_full, pc_segments, _ = self.est.extract_point_clouds(
                    depth, K, segmap=seg, rgb=None, z_range=list(z_range), segmap_id=int(segmap_id))
                if len(pc_full) < 100:
                    raise ValueError(f"only {len(pc_full)} depth points within z_range {list(z_range)}")
                # ~2.5 GB per forward pass: 3 passes do not fit next to the gsam server in 8 GB
                g, s, c, w = self.est.predict_scene_grasps(pc_full, pc_segments=pc_segments,
                                                            local_regions=local_regions,
                                                            filter_grasps=filter_grasps,
                                                            forward_passes=int(forward_passes))
        finally:  # called once per grasp scan: give the memory back to the always-running gsam server
            self.torch.cuda.empty_cache()
        # predict_grasps squeezes its outputs: one grasp gives a scalar opening and a (3,) contact
        out ={"pred_grasps_cam": {}, "scores": {}, "contact_pts": {}, "gripper_openings": {}}
        for k in g:
            n = len(np.atleast_1d(s[k])) if np.size(s[k]) else 0
            kk = int(k)
            out["pred_grasps_cam"][kk] = np.asarray(g[k], float).reshape(-1, 4, 4)
            out["scores"][kk] = np.atleast_1d(np.asarray(s[k], float))[:n]
            out["contact_pts"][kk] = np.asarray(c[k], float).reshape(-1, 3)
            out["gripper_openings"][kk] = np.atleast_1d(np.asarray(w[k], float))[:n]
        if local_regions or filter_grasps:  # keep only the segments, not the whole-scene pass (-1)
            for d in out.values():
                d.pop(-1, None)
        return out
