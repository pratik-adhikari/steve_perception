# steve_perception/core/frame_types.py

from dataclasses import dataclass
from typing import Optional, Dict, Any

import numpy as np

from builtin_interfaces.msg import Time #AG: needed for precise ROS time typing
from sensor_msgs.msg import Image, CameraInfo #AG: needed for storing original messages


@dataclass
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int


@dataclass
class RGBDFrame:
    """
    The *internal* synchronized frame object produced by CameraAgent.

    This is what gets passed into:
        cb_frame(camera_name: str, frame: RGBDFrame)

    Keep this stable. Everything else (mapping, semantic mapping, logging)
    should adapt to THIS, not to raw ROS topics.
    """

    # Timing / TF
    stamp: float               # seconds (legacy/convenience)
    stamp_msg: Time            #AG: added ROS time for accurate TF/republishing
    frame_id: str
    base_frame: str

    # Numerical payload used by perception code
    rgb: np.ndarray            # OpenCV image (usually uint8, BGR)
    depth_m: np.ndarray        # depth in meters (float32/float64)
    K: np.ndarray              # 3x3 camera intrinsic matrix
    intr: Intrinsics           # convenience wrapper

    # Optional: keep original ROS messages (useful for republishing / debugging)
    #AG: storing original messages allows robust republishing in CameraAgent
    rgb_msg: Optional[Image] = None
    depth_msg: Optional[Image] = None
    camera_info_msg: Optional[CameraInfo] = None
    
    # Extension hooks
    seg_masks: Optional[Dict[str, np.ndarray]] = None
    meta: Optional[Dict[str, Any]] = None
