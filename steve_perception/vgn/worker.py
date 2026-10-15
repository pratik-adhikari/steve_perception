#!/usr/bin/env python3
"""
worker.py
=========
Standalone VGN inference worker — must be run under vgn_env Python 3.10.
Imports only the vendored VGN library (third_party/vgn), never ROS.

Usage (called by bridge.py via subprocess):
  python3 worker.py <data.npz> <model.pth> <tsdf_size> <tsdf_res> <vgn_src>     one inference, exit
  python3 worker.py --serve <model.pth> <vgn_src>                                 persistent
      loads torch + the network once, prints "READY", then per stdin line
      {"npz": path, "tsdf_size": m, "tsdf_res": n} one JSON result line. A one-shot call spends
      ~4-7 s on interpreter start, imports and model load; the network itself takes ~10 ms.

Input npz keys
--------------
  n_views       : int scalar
  tsdf_origin   : (3,) float64
  depth_<i>     : (H, W) float32  metres
  cam_pos_<i>   : (3,) float64    world position of camera
  cam_mat_<i>   : (9,) float64    cam-to-world rotation (row-major 3×3)
  fovy_<i>      : float64 scalar  vertical fov (degrees)        [MuJoCo views]
  intr_<i>      : (4,) float64    fx, fy, cx, cy; cam_mat is then world<-ROS optical frame [real views]
  img_h_<i>     : int scalar
  img_w_<i>     : int scalar

Output: one JSON line to stdout
--------------------------------
  {"found": bool, "pos_tsdf": [x,y,z], "rot_flat": [9 floats],
   "width_m": float, "score": float, "n_candidates": int, "error": null|str}
"""

import json
import sys
import numpy as np
from pathlib import Path


def load(model_path, vgn_src):
    """torch, the vendored VGN library and the network (the slow part of a one-shot call)."""
    # Add the vendored VGN library to the path (the ROS 1 vis / ros_utils modules are not
    # vendored, so nothing here imports ROS)
    if vgn_src not in sys.path:
        sys.path.insert(0, vgn_src)

    import torch
    from vgn.networks import load_network

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return load_network(Path(model_path), device), device


def infer(npz_path, tsdf_size, tsdf_res, net, device):
    """TSDF of the npz's views -> VGN -> result dict (see the module doc)."""
    from vgn.detection import predict, process, select
    from vgn.perception import TSDFVolume, CameraIntrinsic
    from vgn.grasp import from_voxel_coordinates
    from vgn.utils.transform import Transform

    voxel_size = tsdf_size / tsdf_res

    data       = np.load(npz_path, allow_pickle=False)
    n_views    = int(data["n_views"])
    tsdf_origin = data["tsdf_origin"]   # (3,)

    tsdf = TSDFVolume(tsdf_size, tsdf_res)

    for i in range(n_views):
        depth   = data[f"depth_{i}"].astype(np.float32)
        cam_pos = data[f"cam_pos_{i}"]                      # (3,)
        cam_mat = data[f"cam_mat_{i}"].reshape(3, 3)        # world←cam
        img_h   = int(data[f"img_h_{i}"])
        img_w   = int(data[f"img_w_{i}"])
        if f"intr_{i}" in data:
            # real camera: ROS optical frame (+X right, +Y down, +Z forward) = Open3D's,
            # intrinsics [fx, fy, cx, cy] from camera_info
            fx, fy, cx, cy = (float(v) for v in data[f"intr_{i}"])
        else:
            # MuJoCo camera frame (+X right, +Y up, -Z forward) -> Open3D camera frame
            cam_mat = cam_mat @ np.diag([1.0, -1.0, -1.0])
            fovy = float(data[f"fovy_{i}"])
            fy = 0.5 * img_h / np.tan(np.deg2rad(fovy) / 2.0)
            fx = fy
            cx = img_w / 2.0
            cy = img_h / 2.0
        intrinsic = CameraIntrinsic(img_w, img_h, fx, fy, cx, cy)

        # Build T_cam_tsdf = T_cam_world @ T_world_tsdf
        R_cw = cam_mat.T
        T_cw = np.eye(4)
        T_cw[:3, :3] = R_cw
        T_cw[:3,  3] = -R_cw @ cam_pos

        T_wt = np.eye(4)
        T_wt[:3, 3] = tsdf_origin

        T_ct = T_cw @ T_wt
        extrinsic = Transform.from_matrix(T_ct)

        tsdf.integrate(depth, intrinsic, extrinsic)

    # VGN inference
    tsdf_grid        = tsdf.get_grid()          # (1, 40, 40, 40)
    qual, rot, width = predict(tsdf_grid, net, device)
    qual, rot, width = process(tsdf_grid, qual, rot, width)
    grasps, scores   = select(qual.copy(), rot, width)

    if len(grasps) == 0:
        return {"found": False, "n_candidates": 0, "error": None}

    grasps_m = [from_voxel_coordinates(g, voxel_size) for g in grasps]
    best     = int(np.argmax(scores))
    g        = grasps_m[best]

    return {
        "found":        True,
        "pos_tsdf":     g.pose.translation.tolist(),
        "rot_flat":     g.pose.rotation.as_matrix().flatten().tolist(),
        "width_m":      float(g.width),
        "score":        float(scores[best]),
        "n_candidates": len(grasps),
        # every candidate, so the caller can keep only grasps on its target object
        "candidates":   [{"pos_tsdf": gm.pose.translation.tolist(),
                          "rot_flat": gm.pose.rotation.as_matrix().flatten().tolist(),
                          "width_m": float(gm.width), "score": float(sc)}
                         for gm, sc in zip(grasps_m, scores)],
        "error":        None,
    }


def serve(model_path, vgn_src):
    import traceback
    net, device = load(model_path, vgn_src)
    # warm-up (lazy imports, CUDA kernels: the first real call would take ~1.5 s instead of ~0.1 s)
    from vgn.detection import predict, process, select  # noqa: F401
    from vgn.perception import TSDFVolume
    import open3d  # noqa: F401
    grid = TSDFVolume(0.3, 40).get_grid()
    predict(grid, net, device)
    print("READY", flush=True)
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            req = json.loads(line)
            res = infer(req["npz"], float(req["tsdf_size"]), int(req["tsdf_res"]), net, device)
        except Exception:
            res = {"found": False, "n_candidates": 0, "error": traceback.format_exc()}
        print(json.dumps(res), flush=True)


def main():
    if sys.argv[1] == "--serve":
        serve(sys.argv[2], sys.argv[3])
        return
    net, device = load(sys.argv[2], sys.argv[5])
    print(json.dumps(infer(sys.argv[1], float(sys.argv[3]), int(sys.argv[4]), net, device)))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        print(json.dumps({
            "found": False, "n_candidates": 0,
            "error": traceback.format_exc(),
        }))
        sys.exit(1)
