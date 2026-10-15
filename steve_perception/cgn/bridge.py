#!/usr/bin/env python3
"""
bridge.py — Contact-GraspNet as the grasp detector of the WBC's grasp scan, behind the interface
of VGNBridge (steve_perception.vgn), so steve_wbc's active_perception_wbc switches with
`--grasp_backend cgn`.

VGN fuses the scan's depth views into one TSDF. Contact-GraspNet works on single views, so here
every view is used on its own and the grasps of all views are pooled:
  1. target mask in the view's colour image: Grounded SAM with the text prompt (model server
     gsam); among several masks the one whose depth points lie nearest the tracked target. No
     server / no matching mask: the depth pixels within the object radius (+1.5 cm) of the
     tracked target position (a geometric mask; enough for a known, tracked object).
  2. Contact-GraspNet (model server cgn) on the whole depth view, grasps restricted to the
     target segment — the rest of the view is the collision context (clutter, table).
  3. grasps -> world (the view's camera pose) -> WBC pinch frame (fingertip centre; z = approach,
     y = finger axis), as VGN's.
best_grasp_world(near, xy_radius, z_window) then picks the best-scoring grasp on the target, as
VGNBridge's does.

Public API (as used by steve_wbc's RosWorld)
    CGNBridge(prompt, radius, gsam_endpoint, cgn_endpoint)
    .start_health_check() / .health(timeout) -> None while checking, then True/False
    .init_tsdf(center)                        a new scan
    .add_real_view(depth, cam_pos, R_world_optical, K, center, rgb) -> view count
    .start_inference() + .poll_inference() | .run_inference()
    .best_grasp_world(near, xy_radius, z_window) -> (pos, R, width, score) | None
    .save_scan(folder, scan_no, best)        views with target mask + grasps as images / npz
"""

import threading
import time
from typing import Optional

import numpy as np

from steve_perception.grasp_test.cgn_client import to_pinch_frame
from steve_perception.model_server.client import (CGN_ENDPOINT, GSAM_ENDPOINT, ModelClient,
                                                  ModelServerError)
from steve_perception.utils.object_geometry import mask_points
from steve_perception.utils.run_record import save_scan


class CGNBridge:
    def __init__(self, prompt="red ball", radius=0.0325, gsam_endpoint=GSAM_ENDPOINT,
                 cgn_endpoint=CGN_ENDPOINT, z_range=(0.15, 1.2), max_mask_dist=0.06):
        self.prompt, self.radius = prompt, radius
        self.gsam_endpoint, self.cgn_endpoint = gsam_endpoint, cgn_endpoint
        self.z_range, self.max_mask_dist = z_range, max_mask_dist
        self.available = True
        self.health_msg = ""
        self.use_gsam = True
        self._views, self._grasps = [], []
        self._center = None
        self._health = self._job = None
        try:
            ModelClient(cgn_endpoint)  # pyzmq present?
        except ModelServerError as e:
            print(f"  [CGN] {e}")
            self.available = False
        else:
            print(f"  [CGN] Contact-GraspNet at {cgn_endpoint}, target mask: Grounded SAM '{prompt}' at "
                  f"{gsam_endpoint} (fallback: depth within {radius * 100:.1f} + 1.5 cm of the tracked target)")

    # ── environment check: both servers answering (they may still be loading) ───
    def start_health_check(self, wait_s=60.0):
        if not self.available:
            return
        def check():
            cgn, gsam = ModelClient(self.cgn_endpoint), ModelClient(self.gsam_endpoint)
            t0, have_cgn, have_gsam = time.monotonic(), None, None
            while time.monotonic() - t0 < wait_s and not (have_cgn and have_gsam):
                have_cgn = have_cgn or cgn.ping(1.0) == "cgn"
                have_gsam = have_gsam or gsam.ping(1.0) == "gsam"
            cgn.close(), gsam.close()
            self.use_gsam = bool(have_gsam)
            if not have_gsam:
                print(f"  [CGN] no Grounded SAM server at {self.gsam_endpoint}: target mask from the tracked "
                      "position only")
            self._health_ok = bool(have_cgn)
            if not have_cgn:
                self.health_msg = (f"no Contact-GraspNet server at {self.cgn_endpoint} within {wait_s:.0f} s "
                                   "(ros2 run steve_perception model_server cgn)")
        self._health_ok = None
        self._health = threading.Thread(target=check, daemon=True)
        self._health.start()

    def health(self, timeout: float = 60.0) -> Optional[bool]:
        if self._health is None:
            return self.available
        if self._health.is_alive():
            return None
        self._health = None
        if not self._health_ok:
            self.available = False
            print(f"  [CGN] {self.health_msg}\n        -> grasp scan disabled, falling back to the tracked "
                  "object position")
        return self.available

    # ── views ──────────────────────────────────────────────────────────────────
    def init_tsdf(self, center) -> None:
        """A new scan (the name is VGNBridge's): forget views and grasps."""
        self._center = np.asarray(center, float).copy()
        self._views, self._grasps = [], []

    def add_real_view(self, depth, cam_pos, R_world_optical, K, center_approx, rgb=None) -> int:
        if not self.available:
            return 0
        if self._center is None:
            self.init_tsdf(center_approx)
        d = np.asarray(depth, np.float32).copy()
        d[~np.isfinite(d)] = 0.0
        self._views.append({"depth": d, "t": np.asarray(cam_pos, float), "R": np.asarray(R_world_optical, float),
                            "K": np.asarray(K, float), "rgb": rgb, "center": np.asarray(center_approx, float)})
        return len(self._views)

    @property
    def n_views(self) -> int:
        return len(self._views)

    # ── inference (background thread) ───────────────────────────────────────────
    def start_inference(self) -> bool:
        if not self.available or not self._views:
            return False
        views = list(self._views)
        self._job = {"done": False, "ok": False, "t0": time.monotonic()}
        threading.Thread(target=self._infer, args=(views, self._job), daemon=True).start()
        return True

    def poll_inference(self, timeout: float = 60.0) -> Optional[bool]:
        job = self._job
        if job is None:
            return False
        if not job["done"]:
            if time.monotonic() - job["t0"] < timeout:
                return None
            print(f"  [CGN] inference timed out after {timeout:.0f} s")
            self._job = None
            return False
        self._job = None
        return job["ok"]

    def run_inference(self, timeout: float = 120.0) -> bool:
        if not self.start_inference():
            return False
        while True:
            r = self.poll_inference(timeout)
            if r is not None:
                return r
            time.sleep(0.05)

    def _infer(self, views, job):
        cgn = ModelClient(self.cgn_endpoint, timeout_s=30.0)
        gsam = ModelClient(self.gsam_endpoint, timeout_s=10.0) if self.use_gsam else None
        grasps = []
        try:
            for i, v in enumerate(views):
                seg, how = self._target_mask(v, gsam)
                v["mask"], v["grasps"] = seg, []
                if seg is None:
                    print(f"  [CGN] view {i + 1}: {how} -> skipped")
                    continue
                try:
                    r = cgn.grasps(v["depth"], v["K"], seg.astype(np.int32), z_range=self.z_range)
                except ModelServerError as e:
                    print(f"  [CGN] view {i + 1}: {e}")
                    continue
                T_wc = np.eye(4)
                T_wc[:3, :3], T_wc[:3, 3] = v["R"], v["t"]
                n = 0
                for G, sc, w in zip(r["pred_grasps_cam"].get(1, []), r["scores"].get(1, []),
                                    r["gripper_openings"].get(1, [])):
                    pos, R = to_pinch_frame(T_wc @ np.asarray(G, float))
                    grasps.append((pos, R, float(w), float(sc)))
                    v["grasps"].append(grasps[-1])
                    n += 1
                print(f"  [CGN] view {i + 1}: target mask {how}, {n} grasp(s) on it ({r['t']:.2f} s)")
            self._grasps = grasps
            job["ok"] = bool(grasps)
            print(f"  [CGN] {len(grasps)} grasp candidate(s) on the target from {len(views)} view(s)")
        finally:
            cgn.close()
            if gsam is not None:
                gsam.close()
            job["done"] = True

    def _target_mask(self, v, gsam):
        """(HxW bool mask of the target, description) or (None, reason)."""
        D, K, R, t = v["depth"], v["K"], v["R"], v["t"]
        target = v["center"]
        world = lambda m: mask_points(D, m, K, self.z_range[0], self.z_range[1]) @ R.T + t
        if gsam is not None and v["rgb"] is not None:
            try:
                dets = gsam.segment(v["rgb"], self.prompt)
            except ModelServerError as e:
                dets = []
                print(f"  [CGN] Grounded SAM: {e}")
            best = None
            for d in dets:
                P = world(d["mask"])
                if len(P) < 30:
                    continue
                # visible surface: its median lies about one radius from the centre
                dist = abs(np.linalg.norm(np.median(P, axis=0) - target) - self.radius)
                if dist < self.max_mask_dist and (best is None or dist < best[0]):
                    best = (dist, d)
            if best is not None:
                d = best[1]
                return d["mask"], f"'{d['phrase']}' {d['score']:.2f} ({best[0] * 100:.1f} cm from the track)"
            n = len(dets)
        # geometric mask: depth pixels within the object around the tracked position
        h, w = D.shape
        v_, u_ = np.mgrid[0:h, 0:w]
        z = D
        P = np.stack([(u_ - K[0, 2]) * z / K[0, 0], (v_ - K[1, 2]) * z / K[1, 1], z], -1) @ R.T + t
        m = (z > self.z_range[0]) & (np.linalg.norm(P - target, axis=-1) < self.radius + 0.015)
        if m.sum() < 30:
            return None, "target not in view"
        return m, (f"geometric (Grounded SAM: {n} mask(s), none at the track)" if gsam is not None
                   and v["rgb"] is not None else "geometric")

    def best_grasp_world(self, near=None, xy_radius=np.inf, z_window=np.inf):
        """(pos_world, R (pinch frame), width_m, score) of the best grasp within xy_radius / z_window of
        `near`, or None."""
        best = None
        for pos, R, w, sc in self._grasps:
            if near is not None:
                d = pos - np.asarray(near, float)
                if np.hypot(d[0], d[1]) > xy_radius or abs(d[2]) > z_window:
                    continue
            if best is None or sc > best[3]:
                best = (pos, R, w, sc)
        if near is not None:
            print(f"  [CGN] {len(self._grasps)} candidate(s), "
                  f"{'best on the target: score %.3f' % best[3] if best else 'none on the target'}")
        return best

    def save_scan(self, folder, scan_no, best):
        save_scan(folder, scan_no, self._views, self._grasps, best, "Contact-GraspNet")
