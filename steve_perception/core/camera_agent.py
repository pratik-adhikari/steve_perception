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
from rclpy.callback_groups import ReentrantCallbackGroup
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import Buffer
from steve_perception.core.frame_types import RGBDFrame, Intrinsics
from steve_perception.utils.tf_utils import transform_to_matrix
from rclpy.qos import qos_profile_sensor_data

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
    # TF lookup timeout (seconds)
    wait_for_tf_sec: float = 0.2

class CameraAgent:
    """Synchronize + (optionally) republish a unified RGBD stream."""

    def __init__(self, node: Node, name: str, tf_buffer: Buffer, cfg: CameraAgentConfig, on_frame=None):
        """Initialize camera agent with subscriptions and synchronizer."""
        self.node = node
        self.log = node.get_logger()
        self.name = name
        self.tf_buffer = tf_buffer
        self.cfg = cfg
        self._on_frame = on_frame
        self._enabled = True
        self._lock = threading.Lock()
        self._latest_frame: Optional[RGBDFrame] = None
        self._bridge = CvBridge()

        # Latest CameraInfo is cached (intrinsics don't need time sync as its mostly constant or low frequency).
        self._latest_info_msg: Optional[CameraInfo] = None

        # Setup message_filters subscriptions (RGB + Depth)
        # CameraInfo is cached via a regular subscription.
        # Use ReentrantCallbackGroup to allow TF lookups (which block) to not starve other callbacks
        self._cb_group = ReentrantCallbackGroup()

        self._rgb_sub = Subscriber(
            self.node, Image, self.cfg.rgb_topic,
            qos_profile=qos_profile_sensor_data, callback_group=self._cb_group
        )
        self._depth_sub = Subscriber(
            self.node, Image, self.cfg.depth_topic,
            qos_profile=qos_profile_sensor_data, callback_group=self._cb_group
        )

        self._info_sub_ros = self.node.create_subscription(
            CameraInfo, self.cfg.info_topic, self._on_info,
            qos_profile=qos_profile_sensor_data, callback_group=self._cb_group
        )
        self._ats = ApproximateTimeSynchronizer(
            [self._rgb_sub, self._depth_sub],
            queue_size=int(self.cfg.queue_size),
            slop=float(self.cfg.slop),
        )
        
        self._ats.registerCallback(self._cb)
        self.log.info(f"[CameraAgent:{self.name}] sync rgb='{self.cfg.rgb_topic}' depth='{self.cfg.depth_topic}' info='{self.cfg.info_topic}'")

        # --- Debug Instrumentation ---
        self._debug_rgb_count = 0
        self._debug_depth_count = 0
        self._debug_sync_count = 0
        self._debug_timer_period = 30.0
        self._debug_timer = self.node.create_timer(self._debug_timer_period, self._debug_timer_cb)

        # Register debug callbacks to inspect raw rates
        # message_filters.Subscriber is a SimpleFilter; we can attach multiple callbacks.
        self._rgb_sub.registerCallback(self._debug_rgb_hook)
        self._depth_sub.registerCallback(self._debug_depth_hook)

    def _debug_rgb_hook(self, msg):
        self._debug_rgb_count += 1

    def _debug_depth_hook(self, msg):
        self._debug_depth_count += 1

    def _debug_timer_cb(self):
        rgb_rate = self._debug_rgb_count / self._debug_timer_period
        depth_rate = self._debug_depth_count / self._debug_timer_period
        sync_rate = self._debug_sync_count / self._debug_timer_period
        
        self.log.info(
            f"[CameraAgent:{self.name}] RATES (Hz) -> RGB: {rgb_rate:.1f}, "
            f"Depth: {depth_rate:.1f}, Sync: {sync_rate:.1f}"
        )
        
        self._debug_rgb_count = 0
        self._debug_depth_count = 0
        self._debug_sync_count = 0

    def get_latest_frame(self) -> Optional[RGBDFrame]:
        """Thread-safe access to latest synchronized frame."""
        with self._lock:
            return self._latest_frame

    def _on_info(self, msg: CameraInfo) -> None:
        """Cache the most recent CameraInfo."""
        with self._lock:
            self._latest_info_msg = msg

    def _cb(self, rgb_msg: Image, depth_msg: Image) -> None:
        """Synchronized callback for RGB+Depth (CameraInfo is cached)."""
        self._debug_sync_count += 1
        if not self._enabled:
            return
        # Always process frames so downstream code can use get_latest_frame(),
        # even if RGBDImage publishing is disabled.

        with self._lock:
            info_msg = self._latest_info_msg

        if info_msg is None:
            self.log.warn(
                f"[CameraAgent:{self.name}] drop frame: no CameraInfo received yet on {self.cfg.info_topic}"
            )
            return

        # Use cached intrinsics but stamp them at the RGB timestamp for consistency.
        info_for_out = CameraInfo()
        info_for_out.header = rgb_msg.header
        info_for_out.height = info_msg.height
        info_for_out.width = info_msg.width
        info_for_out.distortion_model = info_msg.distortion_model
        info_for_out.d = info_msg.d
        info_for_out.k = info_msg.k
        info_for_out.r = info_msg.r
        info_for_out.p = info_msg.p
        info_for_out.binning_x = info_msg.binning_x
        info_for_out.binning_y = info_msg.binning_y
        info_for_out.roi = info_msg.roi

        stamp = rgb_msg.header.stamp
        t0 = Time.from_msg(stamp)

        # For a moving camera frame, we must ensure TF exists at the *exact* image timestamp.
        try:
            tf = self.tf_buffer.lookup_transform(
                self.cfg.base_frame,
                self.cfg.frame_id,
                t0,
                timeout=Duration(seconds=float(self.cfg.wait_for_tf_sec)),
            )
        except Exception as e:
            self.log.warn(
                f"[CameraAgent:{self.name}] drop frame: TF {self.cfg.base_frame}<-{self.cfg.frame_id} at stamp missing ({e})"
            )
            return

        T_base_cam = transform_to_matrix(tf.transform)

        rgb_np = self._bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="bgr8")
        depth_np = self._bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough").astype(np.float32)

        # Standard ROS Rep 118:
        # - 16UC1 is depth in millimeters.
        # - 32FC1 is depth in meters.
        if depth_msg.encoding == '16UC1':
             depth_np *= 0.001
        
        # Apply user-defined scaling (e.g. for correction)
        depth_np *= float(self.cfg.depth_scale)

        depth_np[~np.isfinite(depth_np)] = 0.0
        dmin = float(self.cfg.depth_min)
        dmax = float(self.cfg.depth_max)
        if dmax > dmin:
            depth_np[(depth_np < dmin) | (depth_np > dmax)] = 0.0

        K = np.array(info_for_out.k, dtype=np.float32).reshape(3, 3)
        intr = Intrinsics(
            fx=float(K[0, 0]),
            fy=float(K[1, 1]),
            cx=float(K[0, 2]),
            cy=float(K[1, 2]),
            width=int(info_for_out.width) if info_for_out.width else int(rgb_np.shape[1]),
            height=int(info_for_out.height) if info_for_out.height else int(rgb_np.shape[0]),
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
            T_base_cam=T_base_cam,
            rgb_msg=rgb_msg,
            depth_msg=depth_msg,
            camera_info_msg=info_for_out,
        )

        with self._lock:
            self._latest_frame = frame

        if self._on_frame is not None:
            self._on_frame(frame)
