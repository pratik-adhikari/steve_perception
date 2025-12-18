"""CameraAgent: synchronize RGB + Depth + CameraInfo and publish RTAB-Map RGBDImage."""

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
    # Input topics and frames
    rgb_topic: str
    depth_topic: str
    info_topic: str
    frame_id: str
    base_frame: str
    # Depth processing
    depth_scale: float = 1.0
    depth_min: float = 0.0
    depth_max: float = 40.0
    # Message synchronization
    queue_size: int = 30
    slop: float = 0.02
    # TF timing parameters
    wait_for_tf_sec: float = 0.2
    tf_delay_sec: float = 0.05  # compensate for TF/image timestamp jitter
    tf_tolerance_sec: float = 0.05
    # Output control
    publish_rgbd_allowed: bool = False
    start_publishing: bool = False
    rgbd_topic: Optional[str] = None

class CameraAgent:
    """Synchronize + (optionally) republish a unified RGBD stream."""

    def __init__(self, node: Node, name: str, tf_buffer: Buffer, cfg: CameraAgentConfig):
        """Initialize camera agent with subscriptions and synchronizer."""
        self.node = node
        self.log = node.get_logger()
        self.name = name
        self.tf_buffer = tf_buffer
        self.cfg = cfg
        self._publish_allowed = bool(cfg.publish_rgbd_allowed)
        self._publish_enabled = False
        self._enabled = True
        self._lock = threading.Lock()
        self._latest_frame: Optional[RGBDFrame] = None
        self._bridge = CvBridge()

        # Setup message_filters subscriptions
        self._rgb_sub = Subscriber(self.node, Image, self.cfg.rgb_topic)
        self._depth_sub = Subscriber(self.node, Image, self.cfg.depth_topic)
        self._info_sub = Subscriber(self.node, CameraInfo, self.cfg.info_topic)
        self._ats = ApproximateTimeSynchronizer(
            [self._rgb_sub, self._depth_sub, self._info_sub],
            queue_size=int(self.cfg.queue_size),
            slop=float(self.cfg.slop),
        )
        self._ats.registerCallback(self._cb)

        # Publisher created on-demand when publishing is enabled
        self._pub_rgbd = None
        self.log.info(f"[CameraAgent:{self.name}] sync rgb='{self.cfg.rgb_topic}' depth='{self.cfg.depth_topic}' info='{self.cfg.info_topic}'")

        if bool(self.cfg.start_publishing):
            self.set_publishing(True)

    def start(self) -> None:
        """Enable frame processing."""
        self._enabled = True
        self.log.info(f"[CameraAgent:{self.name}] enabled")

    def stop(self) -> None:
        """Disable frame processing."""
        self._enabled = False
        self.log.info(f"[CameraAgent:{self.name}] disabled")

    def set_publishing(self, enabled: bool) -> None:
        """Enable/disable RGBDImage publishing (respects config allow-list)."""
        want = bool(enabled) and self._publish_allowed
        if want and self._pub_rgbd is None:
            out_topic = self.cfg.rgbd_topic or f"/steve_perception/{self.name}/rgbd_image"
            self._pub_rgbd = self.node.create_publisher(RGBDImage, out_topic, 10)
            self.log.info(f"[CameraAgent:{self.name}] publishing RGBDImage -> {out_topic}")
        self._publish_enabled = want

    def get_latest_frame(self) -> Optional[RGBDFrame]:
        """Thread-safe access to latest synchronized frame."""
        with self._lock:
            return self._latest_frame

    def _cb(self, rgb_msg: Image, depth_msg: Image, info_msg: CameraInfo) -> None:
        """Synchronized callback for RGB+Depth+CameraInfo."""
        if not self._enabled:
            return
        if not self._publish_enabled:
            return

        stamp = rgb_msg.header.stamp
        t0 = Time.from_msg(stamp)
        delay = Duration(seconds=float(self.cfg.tf_delay_sec))
        tol = float(self.cfg.tf_tolerance_sec)

        # Try multiple timestamps to find valid TF
        candidates = [t0 - delay]
        if tol > 0.0:
            candidates += [
                (t0 - delay) - Duration(seconds=tol),
                (t0 - delay) + Duration(seconds=tol),
            ]

        ok = False
        last_err: Optional[Exception] = None
        for tt in candidates:
            try:
                # Check if TF is available at this timestamp
                self.tf_buffer.lookup_transform(
                    self.cfg.base_frame,
                    self.cfg.frame_id,
                    tt,
                    timeout=Duration(seconds=float(self.cfg.wait_for_tf_sec)),
                )
                ok = True
                break
            except Exception as e:
                last_err = e

        if not ok:
            self.log.warn(f"[CameraAgent:{self.name}] drop frame: TF {self.cfg.base_frame}<-{self.cfg.frame_id} near stamp missing ({last_err})")
            return

        # Publish RGBDImage message if enabled
        if self._pub_rgbd is not None:
            out = RGBDImage()
            out.header = rgb_msg.header
            out.rgb = rgb_msg
            out.depth = depth_msg
            out.rgb_camera_info = info_msg
            out.depth_camera_info = info_msg
            self._pub_rgbd.publish(out)

        rgb_np = self._bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="bgr8")
        depth_np = self._bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough").astype(np.float32)

        # Convert depth to meters if needed
        if depth_np.dtype == np.uint16 or float(np.nanmax(depth_np)) > 50.0:
            depth_np *= 0.001
        depth_np *= float(self.cfg.depth_scale)

        depth_np[~np.isfinite(depth_np)] = 0.0
        dmin = float(self.cfg.depth_min)
        dmax = float(self.cfg.depth_max)
        if dmax > dmin:
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
