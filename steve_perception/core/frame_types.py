# steve_perception/core/frame_types.py
from dataclasses import dataclass
from typing import Optional, Dict, Any
import numpy as np
from builtin_interfaces.msg import Time
from sensor_msgs.msg import Image, CameraInfo

@dataclass
class Intrinsics:
    """Camera intrinsic parameters."""
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

@dataclass
class RGBDFrame:
    """Synchronized RGB-D frame with TF and camera info."""
    # Timing and coordinate frames
    stamp: float
    stamp_msg: Time
    frame_id: str
    base_frame: str
    # Image data (NumPy arrays)
    rgb: np.ndarray
    depth_m: np.ndarray
    K: np.ndarray
    intr: Intrinsics
    # Original ROS messages for republishing
    rgb_msg: Optional[Image] = None
    depth_msg: Optional[Image] = None
    camera_info_msg: Optional[CameraInfo] = None
    seg_masks: Optional[Dict[str, np.ndarray]] = None
    meta: Optional[Dict[str, Any]] = None
