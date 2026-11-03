#!/usr/bin/env python3
"""
bridge.py — DexGraspNet 2.0 as the grasp detector of the WBC's grasp scan (`--grasp_backend dgn2`), behind the
interface of VGNBridge / CGNBridge, with every grasp retargeted from DexGraspNet's LEAP hand to the tool's hand
(steve_grasping.retarget: Ability Hand wrap / pinch).

Per view (single-view, as DexGraspNet 2.0 is trained):
  1. the scene: depth pixels whose world points lie within `area` (xy) of the tracked target and between the
     table top - 3 cm and 35 cm above the target — the table around the object, the clutter next to it
     (collision context for the network), not the rest of the room
  2. DexGraspNet 2.0 (model server dgn2): LEAP grasps with fingertip pads and knuckles, camera frame
  3. -> world (the view's camera pose) -> Retargeter -> the hand's grasp type and its frame (z = approach,
     y = closing direction), grasp point between thumb and fingers
best_grasp_world(near, xy_radius, z_window) picks the best-scoring retargeted grasp on the target whose approach
is within max_tilt of vertical (else the best on the target, reported); its grasp type and approach are in
`best_meta` (HandGrasp) for the WBC.

Public API (as VGNBridge / CGNBridge)
    DGN2Bridge(tool, radius, endpoint, table_z=None)
    .start_health_check() / .health(timeout) -> None while checking, then True/False
    .init_tsdf(center) / .add_real_view(depth, cam_pos, R_world_optical, K, center, rgb) -> view count
    .start_inference() + .poll_inference() | .run_inference()
    .best_grasp_world(near, xy_radius, z_window) -> (grasp point, R (grasp type frame), aperture, score) | None
    .scan_data() / .save_scan(folder, scan_no, best)
"""

import threading
import time
from typing import Optional

import numpy as np

from steve_perception.model_server.client import DGN2_ENDPOINT, ModelClient, ModelServerError
from steve_perception.utils.run_record import save_scan

FINGERS = ("index", "middle", "ring", "thumb")   # the server's tips order


class DGN2Bridge:
    def __init__(self, tool, radius=0.025, endpoint=DGN2_ENDPOINT, table_z=None, area=0.25, max_tilt=45.0,
                 z_range=(0.15, 1.5), n_seeds=64, top_k=32, max_views=2):
        from steve_grasping.retarget import Retargeter
        self.radius, self.endpoint, self.table_z, self.area = radius, endpoint, table_z, area
        self.max_tilt, self.z_range, self.n_seeds, self.top_k = max_tilt, z_range, n_seeds, top_k
        self.max_views = max_views  # inference on the views nearest the object (~1-2 s each on the laptop GPU)
        self.available, self.health_msg = True, ""
        self._views, self._grasps, self._meta = [], [], []
        self._center = self._health = self._job = None
        self.best_meta = None
        self.retarget = Retargeter(tool)
        try:
            ModelClient(endpoint)  # pyzmq present?
        except ModelServerError as e:
            print(f"  [DGN2] {e}")
            self.available = False
        else:
            print(f"  [DGN2] DexGraspNet 2.0 at {endpoint}, LEAP grasps retargeted to {tool.name} "
                  f"({', '.join(self.retarget.types) or 'its grasp'})")

    # ── environment check ────────────────────────────────────────────────────────
    def start_health_check(self, wait_s=60.0):
        if not self.available:
            return

        def check():
            c, t0, ok = ModelClient(self.endpoint), time.monotonic(), False
            while time.monotonic() - t0 < wait_s and not ok:
                ok = c.ping(1.0) == "dgn2"
            c.close()
            self._health_ok = ok
            if not ok:
                self.health_msg = (f"no DexGraspNet 2.0 server at {self.endpoint} within {wait_s:.0f} s "
                                   "(ros2 run steve_perception model_server dgn2)")
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
            print(f"  [DGN2] {self.health_msg}\n         -> grasp scan disabled, the shape-based grasp instead")
        return self.available

    # ── views ──────────────────────────────────────────────────────────────────
    def init_tsdf(self, center) -> None:
        self._center = np.asarray(center, float).copy()
        self._views, self._grasps, self._meta = [], [], []

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
        # single-view network: the views from nearest the object (finest depth), at most max_views of them
        views = sorted(self._views, key=lambda v: np.linalg.norm(v["t"] - v["center"]))[:self.max_views]
        if len(views) < len(self._views):
            print(f"  [DGN2] inference on the {len(views)} view(s) nearest the object (of {len(self._views)}; "
                  + ", ".join(f"{np.linalg.norm(v['t'] - v['center']):.2f} m" for v in views) + ")")
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
            print(f"  [DGN2] inference timed out after {timeout:.0f} s")
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

    def _area_mask(self, v):
        D, K, R, t, c = v["depth"], v["K"], v["R"], v["t"], v["center"]
        h, w = D.shape
        vv, uu = np.mgrid[0:h, 0:w]
        P = np.stack([(uu - K[0, 2]) * D / K[0, 0], (vv - K[1, 2]) * D / K[1, 1], D], -1) @ R.T + t
        lo = (self.table_z if self.table_z is not None else c[2] - self.radius) - 0.03
        return ((D > self.z_range[0]) & (np.hypot(P[..., 0] - c[0], P[..., 1] - c[1]) < self.area)
                & (P[..., 2] > lo) & (P[..., 2] < c[2] + 0.35))

    def _infer(self, views, job):
        from steve_grasping.retarget import SourceGrasp
        cli = ModelClient(self.endpoint, timeout_s=30.0)
        grasps, meta = [], []
        try:
            for i, v in enumerate(views):
                v["mask"], v["grasps"] = self._area_mask(v), []
                try:
                    r = cli.dex_grasps(v["depth"], v["K"], v["mask"], self.z_range, self.n_seeds, self.top_k)
                except ModelServerError as e:
                    print(f"  [DGN2] view {i + 1}: {e}")
                    continue
                R, t = v["R"], v["t"]
                for tips, kn, sc in zip(r["tips"], r["knuckles"], r["score"]):
                    W = np.asarray(tips, float) @ R.T + t
                    hg = self.retarget(SourceGrasp(dict(zip(FINGERS, W)), np.asarray(kn, float) @ R.T + t,
                                                   float(sc)))
                    g = (hg.p, hg.R, hg.aperture, hg.score)
                    grasps.append(g)
                    meta.append(hg)
                    v["grasps"].append(g)
                types = [m.gtype for m in meta[-len(v["grasps"]):]] if v["grasps"] else []
                print(f"  [DGN2] view {i + 1}: {len(v['grasps'])} grasp(s) from {r['n_points']} points "
                      f"({r['t']:.2f} s): " + ", ".join(f"{n}x {k}" for k in sorted(set(types))
                                                        for n in [types.count(k)]))
            self._grasps, self._meta = grasps, meta
            job["ok"] = bool(grasps)
        finally:
            cli.close()
            job["done"] = True

    def best_grasp_world(self, near=None, xy_radius=np.inf, z_window=np.inf):
        """(grasp point, R of the hand's grasp type frame, aperture, score) of the best grasp on the target,
        approach within max_tilt of vertical preferred; its HandGrasp in best_meta. None: no grasp on it."""
        on, steep = [], []
        cos_max = np.cos(np.radians(self.max_tilt))
        for g, m in zip(self._grasps, self._meta):
            if near is not None:
                d = g[0] - np.asarray(near, float)
                if np.hypot(d[0], d[1]) > xy_radius or abs(d[2]) > z_window:
                    continue
            on.append((g, m))
            if -m.approach[2] >= cos_max:
                steep.append((g, m))
        pool = steep or on
        best = max(pool, key=lambda gm: gm[0][3]) if pool else None
        self.best_meta = best[1] if best else None
        if near is not None:
            print(f"  [DGN2] {len(self._grasps)} candidate(s), {len(on)} on the target, {len(steep)} within "
                  f"{self.max_tilt:.0f} deg of vertical" + (f"; best: {best[1]}" if best else ""))
        return best[0] if best else None

    def scan_data(self):
        return self._views, self._grasps, "DexGraspNet 2.0"

    def save_scan(self, folder, scan_no, best):
        save_scan(folder, scan_no, *self.scan_data()[:2], best, "DexGraspNet 2.0")
