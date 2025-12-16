#!/usr/bin/env python3

"""steve_perception.core.camera_agent

One CameraAgent == one RGB-D camera.

Responsibilities (keep it strict):
1) Subscribe to (RGB, Depth, CameraInfo) and synchronize them.
2) Gate output on TF availability at the *same timestamp* (prevents TF/time drift issues).
3) Optionally publish a single RTAB-Map-compatible RGBDImage topic.

Non-goals:
- Starting camera drivers (Gazebo / real drivers do that).
- Owning its own TF listener (PerceptionNode owns one, we reuse its buffer).
"""

from __future__ import annotations

from dataclasses import dataclass
import threading
from typing import Optional

import numpy as np
from cv_bridge import CvBridge
from message_filters import Subscriber, ApproximateTimeSynchronizer

from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time

from sensor_msgs.msg import CameraInfo, Image
from rtabmap_msgs.msg import RGBDImage

from tf2_ros import Buffer

from steve_perception.core.frame_types import RGBDFrame, Intrinsics


@dataclass(frozen=True)
class CameraAgentConfig:
    rgb_topic: str
    depth_topic: str
    info_topic: str
    frame_id: str
    base_frame: str

    # Depth handling
    depth_scale: float = 1.0
    depth_min: float = 0.0
    depth_max: float = 40.0

    # Synchronization
    queue_size: int = 30
    slop: float = 0.02

    # TF gating
    wait_for_tf_sec: float = 0.2

    # Output
    publish_rgbd: bool = False
    rgbd_topic: Optional[str] = None


class CameraAgent:
    """Synchronize + (optionally) republish a unified RGBD stream."""

    def __init__(self, node: Node, name: str, tf_buffer: Buffer, cfg: CameraAgentConfig):
        self.node = node
        self.log = node.get_logger()
        self.name = name
        self.tf_buffer = tf_buffer
        self.cfg = cfg

        self._enabled = True
        self._lock = threading.Lock()
        self._latest_frame: Optional[RGBDFrame] = None

        self._bridge = CvBridge()

        # --- Subscriptions (message_filters) ---
        self._rgb_sub = Subscriber(self.node, Image, self.cfg.rgb_topic)
        self._depth_sub = Subscriber(self.node, Image, self.cfg.depth_topic)
        self._info_sub = Subscriber(self.node, CameraInfo, self.cfg.info_topic)

        self._ats = ApproximateTimeSynchronizer(
            [self._rgb_sub, self._depth_sub, self._info_sub],
            queue_size=int(self.cfg.queue_size),
            slop=float(self.cfg.slop),
        )
        self._ats.registerCallback(self._cb)

        # --- Optional output publisher (RGBDImage for RTAB-Map) ---
        self._pub_rgbd: Optional[object] = None
        if self.cfg.publish_rgbd:
            out_topic = self.cfg.rgbd_topic or f"/steve_perception/{self.name}/rgbd_image"
            self._pub_rgbd = self.node.create_publisher(RGBDImage, out_topic, 10)
            self.log.info(f"[CameraAgent:{self.name}] publishing RGBDImage -> {out_topic}")
        else:
            self.log.info(f"[CameraAgent:{self.name}] RGBDImage publishing disabled")

        self.log.info(
            f"[CameraAgent:{self.name}] sync: rgb={self.cfg.rgb_topic} depth={self.cfg.depth_topic} info={self.cfg.info_topic}"
        )

    # -----------------------
    # Control
    # -----------------------
    def start(self) -> None:
        self._enabled = True
        self.log.info(f"[CameraAgent:{self.name}] enabled")

    def stop(self) -> None:
        self._enabled = False
        self.log.info(f"[CameraAgent:{self.name}] disabled")

    # -----------------------
    # Data access
    # -----------------------
    def get_latest_frame(self) -> Optional[RGBDFrame]:
        with self._lock:
            return self._latest_frame

    # -----------------------
    # Internal
    # -----------------------
    def _cb(self, rgb_msg: Image, depth_msg: Image, info_msg: CameraInfo) -> None:
        if not self._enabled:
            return

        # 1) TF gating at the *message timestamp*.
        stamp = rgb_msg.header.stamp
        lookup_time = Time.from_msg(stamp)
        try:
            self.tf_buffer.lookup_transform(
                self.cfg.base_frame,
                self.cfg.frame_id,
                lookup_time,
                timeout=Duration(seconds=float(self.cfg.wait_for_tf_sec)),
            )
        except Exception as e:
            # Do NOT publish frames that RTAB can't transform consistently.
            self.log.warn(
                f"[CameraAgent:{self.name}] drop frame: TF {self.cfg.base_frame}<-{self.cfg.frame_id} @ stamp missing ({e})"
            )
            return

        # 2) Publish RTAB-Map RGBDImage (raw msgs) if enabled.
        if self._pub_rgbd is not None:
            out = RGBDImage()
            out.header = rgb_msg.header
            out.rgb = rgb_msg
            out.depth = depth_msg
            out.rgb_camera_info = info_msg
            out.depth_camera_info = info_msg
            self._pub_rgbd.publish(out)

        # 3) Build internal RGBDFrame (for future semantic mapping / debugging).
        # NOTE: mapping itself does NOT need these numpy conversions, but other WP1 tasks will.
        rgb_np = self._bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="bgr8")
        depth_np = self._bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough").astype(np.float32)

        # Depth normalization:
        # - If depth is uint16 or looks like millimeters, convert to meters.
        if depth_np.dtype == np.uint16 or float(np.nanmax(depth_np)) > 50.0:
            depth_np *= 0.001
        depth_np *= float(self.cfg.depth_scale)

        # Clip invalid / out-of-band values.
        depth_np[~np.isfinite(depth_np)] = 0.0
        if self.cfg.depth_min > 0.0 or self.cfg.depth_max < 1e9:
            dmin = float(self.cfg.depth_min)
            dmax = float(self.cfg.depth_max)
            if dmax <= dmin:
                # Fail safe: don't kill all depth.
                dmax = dmin + 1.0
            depth_np[(depth_np < dmin) | (depth_np > dmax)] = 0.0

        K = np.array(info_msg.k, dtype=np.float32).reshape(3, 3)
        intr = Intrinsics(
            fx=float(K[0, 0]),
            fy=float(K[1, 1]),
            cx=float(K[0, 2]),
            cy=float(K[1, 2]),
            width=int(info_msg.width) if info_msg.width else int(rgb_np.shape[1]),
            height=int(info_msg.height) if info_msg.height else int(rgb_np.shape[0]),
        )

        stamp_sec = float(stamp.sec) + float(stamp.nanosec) * 1e-9
        frame = RGBDFrame(
            stamp=stamp_sec,
            stamp_msg=stamp,
            frame_id=self.cfg.frame_id,
            base_frame=self.cfg.base_frame,
            rgb=rgb_np,
            depth_m=depth_np,
            K=K,
            intr=intr,
            rgb_msg=rgb_msg,
            depth_msg=depth_msg,
            camera_info_msg=info_msg,
        )

        with self._lock:
            self._latest_frame = frame
