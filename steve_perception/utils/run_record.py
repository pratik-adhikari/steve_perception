"""Per-run image records: detections and grasp-scan views as annotated images (+ CSV / npz).

Folder: the launch's log folder ($STEVE_LAUNCH_LOG_DIR, set by `ros2 launch` through the
workspace hook in steve_hardware_bringup: ~/ros_logs/<launch file>_<time>/), else
~/ros_logs/perception_<time>/ for a node started on its own; a sub-folder per producer
(detector/, grasp_scan/)."""

import csv
import os
import time
from pathlib import Path

import numpy as np

_STARTED = time.strftime("%Y%m%d_%H%M%S")


def run_folder():
    d = os.environ.get("STEVE_LAUNCH_LOG_DIR")
    return Path(d) if d else Path.home() / "ros_logs" / f"perception_{_STARTED}"


def save_folder(value, sub):
    """'' / off / false / none -> None (nothing saved); auto -> <run folder>/<sub>; a path -> <path>/<sub>."""
    v = str(value or "").strip()
    if v.lower() in ("", "off", "false", "none", "0"):
        return None
    d = (run_folder() if v.lower() in ("auto", "true", "1") else Path(v).expanduser()) / sub
    d.mkdir(parents=True, exist_ok=True)
    return d


class ImageRecorder:
    """Annotated frames as JPEG (at most one per `period` s) and every row of a CSV log."""

    def __init__(self, folder, period=0.5, columns=None):
        self.folder, self.period = Path(folder), period
        self.n, self._last = 0, -np.inf
        self._csv = self._writer = None
        if columns:
            self._csv = open(self.folder / "detections.csv", "w", newline="", buffering=1)
            self._writer = csv.writer(self._csv)
            self._writer.writerow(columns)

    def due(self):
        return time.monotonic() - self._last >= self.period

    def save(self, rgb, tag=""):
        """rgb: HxWx3 RGB uint8. File: <n>_<wall time>[_tag].jpg."""
        import cv2
        self._last = time.monotonic()
        self.n += 1
        name = f"{self.n:05d}_{time.strftime('%H%M%S')}{int(time.time() * 1000) % 1000:03d}" + (f"_{tag}" if tag else "")
        path = self.folder / f"{name}.jpg"
        cv2.imwrite(str(path), np.ascontiguousarray(rgb[:, :, ::-1]), [cv2.IMWRITE_JPEG_QUALITY, 85])
        return path

    def log(self, row):
        if self._writer is not None:
            self._writer.writerow(row)

    def close(self):
        if self._csv is not None:
            self._csv.close()
            self._csv = None


def depth_image(depth, d_max=1.5):
    """Depth [m] -> RGB colour map (black: no depth)."""
    import cv2
    d = np.nan_to_num(np.asarray(depth, np.float32))
    u8 = np.clip(255.0 * (1.0 - d / d_max), 0, 255).astype(np.uint8)
    img = cv2.applyColorMap(u8, cv2.COLORMAP_TURBO)[:, :, ::-1].copy()
    img[d <= 0] = 0
    return img


def _same(g, best):
    return best is not None and g[3] == best[3] and np.allclose(g[0], best[0])


def draw_grasps(img, K, R_wc, t_wc, grasps, best=None, mask=None, label=""):
    """Grasps (world frame, WBC pinch convention: fingertip centre, R with z = approach, y = finger
    axis, width) drawn into a camera image whose pose is world <- optical (R_wc, t_wc).
    best: the chosen grasp (green, thick); others orange. mask: target mask (tinted)."""
    import cv2
    img = np.ascontiguousarray(img, np.uint8).copy()
    if mask is not None:
        img[mask] = (0.55 * img[mask] + 0.45 * np.array([255, 0, 255])).astype(np.uint8)

    def px(p):
        c = R_wc.T @ (np.asarray(p, float) - t_wc)
        return None if c[2] < 0.05 else (int(K[0, 0] * c[0] / c[2] + K[0, 2]), int(K[1, 1] * c[1] / c[2] + K[1, 2]))

    def one(pos, R, width, col, thick):
        a, b = px(pos - 0.08 * R[:, 2]), px(pos)
        f1, f2 = px(pos + 0.5 * width * R[:, 1]), px(pos - 0.5 * width * R[:, 1])
        if None in (a, b, f1, f2):
            return
        cv2.arrowedLine(img, a, b, col, thick, tipLength=0.2)   # approach, towards the fingertips
        cv2.line(img, f1, f2, col, thick + 1)                    # finger closing line

    order = sorted(grasps, key=lambda g: g[3])
    for g in order:
        if not _same(g, best):
            one(g[0], g[1], g[2], (255, 160, 0), 1)
    if best is not None:
        one(best[0], best[1], best[2], (0, 255, 0), 3)
    text = label + (f" | best score {best[3]:.2f}, width {best[2] * 100:.1f} cm" if best is not None else
                    f" | {len(grasps)} grasp(s), none chosen")
    cv2.putText(img, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 3)
    cv2.putText(img, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return img


def save_scan(folder, scan_no, views, grasps, best, backend):
    """One grasp scan: per view <scanNN>_view<i>.jpg (colour image, else depth colour map, with the
    target mask and all grasps; the chosen one green) and .npz (rgb, depth, K, R_wc, t_wc, mask,
    grasps as pos Nx3 / R Nx3x3 / width / score) + <scanNN>_grasps.csv (all candidates, world)."""
    import cv2
    folder = Path(folder)
    stem = f"scan{scan_no:02d}"
    for i, v in enumerate(views, 1):
        if v.get("K") is None:
            continue
        base = v["rgb"] if v.get("rgb") is not None else depth_image(v["depth"])
        mine = v.get("grasps", grasps)  # CGN: the grasps of this view; VGN: the fused TSDF's
        img = draw_grasps(base, v["K"], v["R"], v["t"], mine, best if any(_same(g, best) for g in mine) else None,
                          v.get("mask"), f"{backend} {stem} view {i}")
        cv2.imwrite(str(folder / f"{stem}_view{i}.jpg"), img[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 90])
        G = list(mine)
        np.savez_compressed(
            folder / f"{stem}_view{i}.npz", depth=v["depth"], K=v["K"], R_wc=v["R"], t_wc=v["t"],
            **({"rgb": v["rgb"]} if v.get("rgb") is not None else {}),
            **({"mask": v["mask"]} if v.get("mask") is not None else {}),
            grasp_pos=np.array([g[0] for g in G]).reshape(-1, 3), grasp_R=np.array([g[1] for g in G]).reshape(-1, 3, 3),
            grasp_width=np.array([g[2] for g in G]), grasp_score=np.array([g[3] for g in G]))
    with open(folder / f"{stem}_grasps.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["chosen", "score", "width", "x", "y", "z", "approach_x", "approach_y", "approach_z",
                    "finger_x", "finger_y", "finger_z"])
        for g in sorted(grasps, key=lambda g: -g[3]):
            w.writerow([int(_same(g, best)), f"{g[3]:.4f}", f"{g[2]:.4f}", *(f"{x:.4f}" for x in g[0]),
                        *(f"{x:.3f}" for x in g[1][:, 2]), *(f"{x:.3f}" for x in g[1][:, 1])])
