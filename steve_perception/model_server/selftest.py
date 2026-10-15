#!/usr/bin/env python3
"""
selftest.py — offline check of the model servers on a saved RGB-D snapshot (no robot, no camera,
no ROS graph): Grounded SAM on the colour image, Contact-GraspNet on the depth with the target
mask, optionally VGN on the same view; annotated images + npz into an output folder.

    ros2 run steve_perception perception_selftest                 # newest grasp_test snapshot
    ros2 run steve_perception perception_selftest --snapshot ~/ros_logs/grasp_test/<run>/snapshot.npz \\
        --prompt "red ball" --vgn

A snapshot is an npz with rgb (HxWx3), depth (HxW or NxHxW, metres) and K (3x3), as written by
handheld_grasp_test. Its camera pose is unknown: the camera is assumed to look straight down from
0.8 m, so the printed positions are only meaningful relative to each other. The model servers must
be running (ros2 run steve_perception model_server gsam / cgn).
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from steve_perception.model_server.client import GSAM_ENDPOINT, ModelClient, ModelServerError
from steve_perception.utils.object_geometry import mask_points
from steve_perception.utils.run_record import save_folder, save_scan

R_DOWN = np.array([[0, -1, 0], [-1, 0, 0], [0, 0, -1]], float)  # world <- optical, looking down
T_DOWN = np.array([0.0, 0.0, 0.8])


def main(argv=None):
    p = argparse.ArgumentParser(description="offline check of the perception model servers on a snapshot")
    p.add_argument("--snapshot", help="npz with rgb, depth, K (default: newest ~/ros_logs/grasp_test/*/snapshot.npz)")
    p.add_argument("--prompt", default="red ball")
    p.add_argument("--size", type=float, default=0.065, help="object diameter [m]")
    p.add_argument("--vgn", action="store_true", help="also VGN on the same view (vgn_env)")
    p.add_argument("--out", default="auto", help="output folder (default: ~/ros_logs/perception_<time>/selftest)")
    a = p.parse_args(argv)

    snap = Path(a.snapshot).expanduser() if a.snapshot else max(
        (Path.home() / "ros_logs" / "grasp_test").glob("*/snapshot.npz"), key=lambda f: f.stat().st_mtime, default=None)
    if snap is None or not snap.is_file():
        sys.exit("no snapshot: give --snapshot <npz with rgb, depth, K>")
    d = np.load(snap)
    rgb, K = d["rgb"], d["K"]
    depth = np.median(d["depth"], axis=0) if d["depth"].ndim == 3 else d["depth"]
    depth = depth.astype(np.float32)
    out = save_folder(a.out, "selftest")
    print(f"snapshot {snap}\noutput   {out}")

    # 1. Grounded SAM
    gsam = ModelClient(GSAM_ENDPOINT, timeout_s=30)
    try:
        t0 = time.monotonic()
        dets = gsam.segment(rgb, a.prompt)
        print(f"[gsam] '{a.prompt}': {len(dets)} mask(s) in {time.monotonic() - t0:.2f} s  " +
              ", ".join(f"{x['phrase']} {x['score']:.2f}" for x in dets))
    except ModelServerError as e:
        sys.exit(f"[gsam] {e}")
    import cv2
    vis = rgb.copy()
    for i, x in enumerate(dets):
        col = (0, 255, 0) if i == 0 else (255, 160, 0)
        vis[x["mask"]] = (0.55 * vis[x["mask"]] + 0.45 * np.array(col)).astype(np.uint8)
        x0, y0, x1, y1 = (int(v) for v in x["box"])
        cv2.rectangle(vis, (x0, y0), (x1, y1), col, 2)
        cv2.putText(vis, f"{x['phrase']} {x['score']:.2f}", (x0, max(14, y0 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
    cv2.imwrite(str(out / "gsam.jpg"), vis[:, :, ::-1])
    if not dets:
        sys.exit("[gsam] nothing found: check the prompt (gsam.jpg)")

    # target centre (world of the assumed pose): mask surface + one radius along the viewing ray
    P = mask_points(depth, dets[0]["mask"], K, 0.1, 2.0)
    s = np.median(P, axis=0)
    centre = R_DOWN @ (s + 0.5 * a.size * s / np.linalg.norm(s)) + T_DOWN
    near = (centre, 0.5 * a.size + 0.03, 0.5 * a.size + 0.08)  # the WBC's grasp window

    # 2. Contact-GraspNet (+ 3. VGN) through the WBC's bridges
    from steve_perception.cgn import CGNBridge
    backends = [("Contact-GraspNet", CGNBridge(a.prompt, 0.5 * a.size))]
    if a.vgn:
        from steve_perception.vgn import VGNBridge
        backends.append(("VGN", VGNBridge()))
    for n, (name, b) in enumerate(backends, 1):
        b.start_health_check()
        while b.health() is None:
            time.sleep(0.1)
        if not b.available:
            print(f"[{name}] not available")
            continue
        b.init_tsdf(centre)
        b.add_real_view(depth, T_DOWN, R_DOWN, K, centre, rgb=rgb)
        t0 = time.monotonic()
        ok = b.run_inference()
        best = b.best_grasp_world(*near) if ok else None
        b.save_scan(out, n, best)
        print(f"[{name}] {time.monotonic() - t0:.2f} s: " + (
            f"best grasp score {best[3]:.3f}, width {best[2] * 100:.1f} cm, fingertip centre "
            f"{np.linalg.norm(best[0] - centre) * 100:.1f} cm from the object centre" if best else "no grasp on the target")
            + f"  -> scan{n:02d}_view1.jpg")
    print(f"done: {out}")


if __name__ == "__main__":
    main()
