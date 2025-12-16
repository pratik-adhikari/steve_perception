#!/usr/bin/env python3

import numpy as np

# DIFFERENCE from tomato_perception.utils.depth_utils:
# - We keep depth_to_points and masked_median_centroid conceptually the same.
# - We add optional z_min/z_max and do NOT hard-code tomato-specific ranges.
# - We remove any mention of “tomato”, trusses, or crop geometry.
# - Reason: we want a generic depth → 3D utility for any robot/camera.


def depth_to_points(
    mask: np.ndarray,
    depth_m: np.ndarray,
    intr,
    T_cb: np.ndarray,
    stride: int = 2,
    z_min: float = 0.04,
    z_max: float = 5.0,
) -> np.ndarray:
    """
    Convert a depth image + mask into 3D points in the base frame.

    Parameters
    ----------
    mask : HxW bool
        True where pixels should be used.
        (In tomato_perception, this is usually a segmentation mask; in our MVP we can use an all-True mask.)
    depth_m : HxW float32
        Depth in meters.
    intr : Intrinsics-like
        Must have attributes fx, fy, cx, cy.
    T_cb : 4x4 np.ndarray
        Transform from camera frame to base frame.
        (Note: tomato_perception uses the same idea, T_cb, but in a tomato-specific context.)
    stride : int
        Downsampling factor (same idea as tomato_perception).
    z_min, z_max : float
        Z filtering bounds in meters, generic (no tomato-size assumptions).

    Returns
    -------
    pts_base : Nx3 float32
        3D points in base frame.
    """
    # CHANGED: We keep the same structure as tomato's depth_to_points, but expose z limits as arguments.
    ys, xs = np.where(mask[::stride, ::stride])
    xs = xs * stride
    ys = ys * stride

    if xs.size == 0:
        return np.empty((0, 3), np.float32)

    d = depth_m[ys, xs]
    # CHANGED: z_min/z_max are arguments instead of hard-coded 0.04/5.0 in tomato_perception.
    valid = np.isfinite(d) & (d > z_min) & (d < z_max)
    xs = xs[valid]
    ys = ys[valid]
    d = d[valid]

    if d.size == 0:
        return np.empty((0, 3), np.float32)

    X = (xs - intr.cx) * d / intr.fx
    Y = (ys - intr.cy) * d / intr.fy
    Z = d

    # Homogeneous coords in camera frame
    pts_cam = np.stack([X, Y, Z, np.ones_like(Z)], axis=1)  # Nx4

    # Transform to base: T_cb @ [X,Y,Z,1]^T
    pts_base = (T_cb @ pts_cam.T).T[:, :3]
    return pts_base.astype(np.float32)


def masked_median_centroid(
    mask: np.ndarray,
    depth_m: np.ndarray,
    intr,
    T_cb: np.ndarray,
    stride: int = 2,
    z_min: float = 0.04,
    z_max: float = 5.0,
):
    """
    Median 3D point of all valid masked depth points.

    DIFFERENCE from tomato_perception.utils.depth_utils.masked_median_centroid:
    - Same logic, but again we expose z_min/z_max as parameters and treat this as a
      generic utility for any object/region, not just tomatoes.
    """
    pts = depth_to_points(mask, depth_m, intr, T_cb, stride=stride, z_min=z_min, z_max=z_max)
    if pts.shape[0] == 0:
        return None
    return np.median(pts, axis=0)
