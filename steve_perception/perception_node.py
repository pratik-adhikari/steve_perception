"""steve_perception.perception_node

Owns the *shared* TF listener and constructs one :class:`CameraAgent` per
configured camera.

Key behavior (matches your requested workflow):
- `perception.launch.py` runs this node with `publish_rgbd:=false`.
  That means: it only subscribes/synchronizes internally, and does NOT create
  any extra "combined" RGBD topics.
- Mapping launches run this node with `publish_rgbd:=true`.
  Then each agent publishes exactly one RTAB-Map-compatible
  `rtabmap_msgs/RGBDImage` topic (already RGB+Depth+CameraInfo synced, and TF-gated).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import yaml

import rclpy
from rclpy.node import Node

from ament_index_python.packages import get_package_share_directory
from tf2_ros import Buffer, TransformListener

from steve_perception.core.camera_agent import CameraAgent, CameraAgentConfig


class PerceptionNode(Node):
    """Owns TF + spawns CameraAgents based on steve.yaml."""

    def __init__(self) -> None:
        super().__init__("steve_perception")

        # ----------------------
        # Parameters
        # ----------------------
        self.declare_parameter("config_file", "steve.yaml")
        self.declare_parameter("publish_rgbd", False)
        self.declare_parameter("enabled_cameras", [])  # string[]; empty => all cameras

        config_file = self.get_parameter("config_file").get_parameter_value().string_value
        publish_rgbd = self.get_parameter("publish_rgbd").get_parameter_value().bool_value
        # In rclpy, string_array_value is already a list of Python strings.
        enabled_cameras = list(self.get_parameter("enabled_cameras").get_parameter_value().string_array_value)

        # ----------------------
        # Load config
        # ----------------------
        pkg_share = Path(get_package_share_directory("steve_perception"))
        cfg_path = pkg_share / "config" / config_file

        self.get_logger().info(f"Loading perception config: {cfg_path}")
        with open(cfg_path, "r") as f:
            cfg = yaml.safe_load(f)

        camera_profiles: Dict[str, dict] = cfg.get("camera_profiles", {})
        if not camera_profiles:
            raise RuntimeError(f"No 'camera_profiles' found in {cfg_path}")

        # Filter cameras (if user specified a subset)
        selected: List[str]
        if enabled_cameras:
            selected = [c for c in enabled_cameras if c in camera_profiles]
            missing = [c for c in enabled_cameras if c not in camera_profiles]
            if missing:
                self.get_logger().warn(f"enabled_cameras contains unknown names: {missing}")
        else:
            selected = list(camera_profiles.keys())

        self.get_logger().info(
            f"PerceptionNode starting cameras={selected} publish_rgbd={publish_rgbd}"
        )

        # ----------------------
        # TF (single shared buffer)
        # ----------------------
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # ----------------------
        # CameraAgents
        # ----------------------
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
                publish_rgbd=bool(publish_rgbd),
                rgbd_topic=str(cam_cfg.get("rgbd_topic")) if cam_cfg.get("rgbd_topic") else None,
            )

            self.camera_agents[cam_name] = CameraAgent(
                node=self,
                name=cam_name,
                tf_buffer=self.tf_buffer,
                cfg=agent_cfg,
            )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PerceptionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
