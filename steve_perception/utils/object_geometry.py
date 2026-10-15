"""Object centre from an RGB-D mask (camera optical frame) and small image/TF helpers.

sphere_center is the estimator of steve_wbc's red_ball_detector (same maths, kept here so that
steve_perception does not depend on steve_wbc)."""

import numpy as np

# optical (x right, y down, z forward) -> x-forward camera frame (x forward, y left, z up)
R_OPT_TO_XFWD = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])


def image_to_array(msg):
    """sensor_msgs/Image -> depth [m] float32 (32FC1, 16UC1 mm) or RGB uint8."""
    if msg.encoding == "32FC1":
        return np.frombuffer(msg.data, np.float32).reshape(msg.height, msg.width)
    if msg.encoding in ("16UC1", "mono16"):  # millimetres (RealSense)
        return np.frombuffer(msg.data, np.uint16).reshape(msg.height, msg.width).astype(np.float32) / 1000.0
    ch = {"rgb8": 3, "bgr8": 3, "rgba8": 4, "bgra8": 4}.get(msg.encoding)
    if ch is None:
        raise ValueError(f"unsupported image encoding '{msg.encoding}'")
    img = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.step // ch if msg.step else msg.width, ch)
    img = img[:, :msg.width, :3]
    return np.ascontiguousarray(img[:, :, ::-1] if msg.encoding.startswith("bgr") else img)


def quat_to_mat(q):
    x, y, z, w = q.x, q.y, q.z, q.w
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def mask_points(depth, mask, K, zmin, zmax, erode_px=0):
    """Nx3 optical-frame points of the mask's valid depth pixels (rim eroded by erode_px)."""
    if erode_px:
        import cv2
        k = np.ones((2 * erode_px + 1, 2 * erode_px + 1), np.uint8)
        mask = cv2.erode(mask.astype(np.uint8), k) > 0
    ys, xs = np.nonzero(mask)
    z = depth[ys, xs]
    ok = np.isfinite(z) & (z > zmin) & (z < zmax)
    return np.stack([(xs[ok] - K[0, 2]) * z[ok] / K[0, 0], (ys[ok] - K[1, 2]) * z[ok] / K[1, 1], z[ok]], 1)


def sphere_center(depth, mask, K, r, zmin, zmax, erode_px=3, max_sd=0.006):
    """Centre (optical frame) of a sphere of radius r from the depth of its mask without the rim
    (RealSense "flying pixels"), by least squares on |p - c| = r; None if too few points or a bad fit."""
    from scipy.optimize import least_squares
    P = mask_points(depth, mask, K, zmin, zmax, erode_px)
    if len(P) < 30:
        return None
    P = P[P[:, 2] < np.percentile(P[:, 2], 5) + 2.5 * r]       # drop remaining points behind the ball
    m = np.median(P, axis=0)
    res = least_squares(lambda c: np.linalg.norm(P - c, axis=1) - r, m + r * m / np.linalg.norm(m),
                        loss="soft_l1", f_scale=0.003)
    return res.x if np.std(res.fun) < max_sd else None
