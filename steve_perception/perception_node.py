"""ROS 2 node that creates CameraAgents and publishes RGBDImage topics.

Configuration is read from steve.yaml (path passed as a ROS parameter).
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, List
import yaml
import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import SetParametersResult
from rclpy.parameter import Parameter
from ament_index_python.packages import get_package_share_directory
from tf2_ros import Buffer, TransformListener
from rclpy.duration import Duration
from steve_perception.core.camera_agent import CameraAgent, CameraAgentConfig

# Perception pipeline entry point
class PerceptionNode(Node):
    """Starts CameraAgents and optionally publishes RGBDImage topics."""

    def __init__(self) -> None:
        """Initialize perception node: load config, create TF buffer, start camera agents."""
        super().__init__("steve_perception")

        # Load perception configuration
        self.declare_parameter("config_file", "steve.yaml")
        config_file  = self.get_parameter("config_file").value

        # Parse camera publishing parameter
        self.declare_parameter("publish_rgbd_cameras", "")
        requested_csv = str(self.get_parameter("publish_rgbd_cameras").value)
        requested = {s.strip() for s in requested_csv.split(",") if s.strip()}

        pkg_share = Path(get_package_share_directory("steve_perception"))
        cfg_path = pkg_share / "config" / config_file
        self.get_logger().info(f"Loading perception config: {cfg_path}")

        with open(cfg_path, "r") as f:
            cfg = yaml.safe_load(f)

        camera_profiles: Dict[str, dict] = cfg.get("camera_profiles", {})
        selected: List[str] = list(camera_profiles.keys())

        if not camera_profiles:
            raise RuntimeError(f"No 'camera_profiles' found in {cfg_path}")

        self.get_logger().info(
            f"PerceptionNode starting cameras={selected} publish_rgbd_cameras='{requested_csv}'"
        )

        # Create shared TF transform system
        self.tf_buffer = Buffer(cache_time=Duration(seconds=10.0))
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.camera_agents: Dict[str, CameraAgent] = {}

        # Create a CameraAgent for each camera profile
        for cam_name in selected:
            cam_cfg = camera_profiles[cam_name]
            # Determine if this camera should publish RGBDImage
            allowed = bool(cam_cfg.get("publish_rgbd", False))
            requested_this = (cam_name in requested) if requested else False
            publish_this = bool(allowed and requested_this)

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
                tf_delay_sec=float(cam_cfg.get("tf_delay_sec", 0.05)),
                tf_tolerance_sec=float(cam_cfg.get("tf_tolerance_sec", 0.05)),
                publish_rgbd_allowed=allowed,
                start_publishing=publish_this,
                rgbd_topic=str(cam_cfg.get("rgbd_topic")) if cam_cfg.get("rgbd_topic") else None,
            )

            self.camera_agents[cam_name] = CameraAgent(
                node=self,
                name=cam_name,
                tf_buffer=self.tf_buffer,
                cfg=agent_cfg,
            )

        # Enable parameter updates at runtime
        self.add_on_set_parameters_callback(self._on_set_parameters)

    def _on_set_parameters(self, params: list[Parameter]) -> SetParametersResult:
        """Handle runtime parameter changes for dynamic camera control."""
        for p in params:
            if p.name == "publish_rgbd_cameras":
                requested_csv = str(p.value)
                requested = {s.strip() for s in requested_csv.split(",") if s.strip()}
                for cam_name, agent in self.camera_agents.items():
                    requested_this = (cam_name in requested)
                    agent.set_publishing(requested_this)
        return SetParametersResult(successful=True)

def main(args=None) -> None:
    """Entry point: ros2 run steve_perception perception_node"""
    rclpy.init(args=args)
    node = PerceptionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
