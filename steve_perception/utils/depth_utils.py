#!/usr/bin/env python3
"""Utility functions for depth and point cloud processing."""

import numpy as np

def depth_to_points(
    mask: np.ndarray,
    depth_m: np.ndarray,
    intr,
    T_cb: np.ndarray,
    stride: int = 2,
    z_min: float = 0.04,
    z_max: float = 5.0,
) -> np.ndarray:
    """Convert depth image + mask to 3D points in base frame.
    
    Args:
        mask: HxW bool array indicating valid pixels
        depth_m: HxW float32 depth in meters
        intr: Camera intrinsics (fx, fy, cx, cy)
        T_cb: 4x4 transform from camera to base frame
        stride: Downsampling factor
        z_min, z_max: Depth filtering bounds in meters
    """
    ys, xs = np.where(mask[::stride, ::stride])
    xs = xs * stride
    ys = ys * stride

    if xs.size == 0:
        return np.empty((0, 3), np.float32)

    d = depth_m[ys, xs]
    # Filter by depth range and validity
    valid = np.isfinite(d) & (d > z_min) & (d < z_max)
    xs = xs[valid]
    ys = ys[valid]
    d = d[valid]

    if d.size == 0:
        return np.empty((0, 3), np.float32)

    X = (xs - intr.cx) * d / intr.fx
    Y = (ys - intr.cy) * d / intr.fy
    Z = d

    # Transform from camera frame to base frame
    pts_cam = np.stack([X, Y, Z, np.ones_like(Z)], axis=1)  # Nx4 homogeneous
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
    """Compute median 3D centroid of valid masked depth points."""
    pts = depth_to_points(mask, depth_m, intr, T_cb, stride=stride, z_min=z_min, z_max=z_max)
    if pts.shape[0] == 0:
        return None
    return np.median(pts, axis=0)
