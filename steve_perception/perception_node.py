"""Perception entry node.

This node owns the CameraAgents (RGB+Depth sync + TF lookup + cached latest frame).

It can optionally publish RTAB-Map compatible `rtabmap_msgs/RGBDImage` topics via
`RtabmapBridge`.

Configuration is read from steve.yaml (path passed as a ROS parameter).
"""

from __future__ import annotations

from typing import Dict, List

import rclpy
import yaml
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener

from steve_perception.bridges.rtabmap_bridge import RtabmapBridge
from steve_perception.core.camera_agent import CameraAgent, CameraAgentConfig
from steve_perception.utils.config_utils import resolve_pkg_config_path


class PerceptionNode(Node):
    """Starts CameraAgents and optionally publishes RGBDImage topics."""

    def __init__(self) -> None:
        """Initialize perception node: load config, create TF buffer, start camera agents."""
        super().__init__("steve_perception")

# --- Parameters ---
        # NOTE: commit(2) will make this param accept absolute paths as well.
        self.declare_parameter("config_file", "steve.yaml")
        # Default to a non-empty list of strings to help type inference, then filter it out
        self.declare_parameter("enabled_cameras", [""])
        self.declare_parameter("publish_rgbd", False)

        config_file = str(self.get_parameter("config_file").value)
        enabled_cameras = [c for c in self.get_parameter("enabled_cameras").value if c]
        publish_rgbd = bool(self.get_parameter("publish_rgbd").value)

        cfg_path = resolve_pkg_config_path("steve_perception", config_file)
        self.get_logger().info(f"Loading perception config: {cfg_path}")

        with open(cfg_path, "r") as f:
            cfg = yaml.safe_load(f) or {}

        camera_profiles: Dict[str, dict] = cfg.get("camera_profiles", {}) or {}
        if not camera_profiles:
            raise RuntimeError(f"No 'camera_profiles' found in {cfg_path}")

        selected: List[str]
        if enabled_cameras:
            selected = [c for c in enabled_cameras if c in camera_profiles]
            missing = [c for c in enabled_cameras if c not in camera_profiles]
            if missing:
                raise RuntimeError(
                    f"enabled_cameras contains unknown cameras {missing}. "
                    f"Known cameras: {sorted(camera_profiles.keys())}"
                )
        else:
            selected = list(camera_profiles.keys())

        self.get_logger().info(
            f"PerceptionNode starting cameras={selected} publish_rgbd={publish_rgbd}"
        )

        # --- Shared TF buffer/listener ---
        self.tf_buffer = Buffer(cache_time=Duration(seconds=10.0))
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # --- Optional RTAB-Map bridge ---
        self.bridge = None
        if publish_rgbd:
            selected_cfg = {name: camera_profiles[name] for name in selected}
            self.bridge = RtabmapBridge(self, selected_cfg)

        # --- Camera agents ---
        self.camera_agents: Dict[str, CameraAgent] = {}
        for cam_name in selected:
            cam_cfg = camera_profiles[cam_name]

            agent_cfg = CameraAgentConfig(
                rgb_topic=str(cam_cfg["rgb_topic"]),
                depth_topic=str(cam_cfg["depth_topic"]),
                info_topic=str(cam_cfg["camera_info_topic"]),
                frame_id=str(cam_cfg["frame_id"]),
                base_frame=str(cam_cfg.get("base_frame", "base_link")),
                depth_scale=float(cam_cfg.get("depth_scale", 1.0)),
                depth_min=float(cam_cfg.get("depth_min", 0.0)),
                depth_max=float(cam_cfg.get("depth_max", 40.0)),
                queue_size=int(cam_cfg.get("queue_size", 30)),
                slop=float(cam_cfg.get("slop", 0.02)),
                wait_for_tf_sec=float(cam_cfg.get("wait_for_tf_sec", 0.2)),
            )

            self.camera_agents[cam_name] = CameraAgent(
                node=self,
                name=cam_name,
                tf_buffer=self.tf_buffer,
                cfg=agent_cfg,
                on_frame=(self.bridge.on_frame if self.bridge is not None else None),
            )



def main(args=None) -> None:
    """Entry point: ros2 run steve_perception perception_node"""
    rclpy.init(args=args)
    node = PerceptionNode()
    executor = MultiThreadedExecutor()
    try:
        rclpy.spin(node, executor=executor)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass

if __name__ == "__main__":
    main()
