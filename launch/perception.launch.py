#!/usr/bin/env python3
"""Launches PerceptionNode only."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution

def generate_launch_description() -> LaunchDescription:
    config_file = LaunchConfiguration("config_file")
    publish_rgbd_cameras = LaunchConfiguration("publish_rgbd_cameras")
    use_sim_time = LaunchConfiguration("use_sim_time")
    log_level = LaunchConfiguration("log_level")

    config_path = PathJoinSubstitution(
        [FindPackageShare("steve_perception"), "config", config_file]
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            "config_file",
            default_value="steve.yaml",
            description="Perception configuration YAML in steve_perception/config.",
        ),
        DeclareLaunchArgument(
            "publish_rgbd_cameras",
            default_value="front",
            description="Comma-separated list of cameras for RGBDImage publishing.",
        ),
        DeclareLaunchArgument(
            "use_sim_time", default_value="true", description="Use Gazebo / simulation time."
        ),
        DeclareLaunchArgument(
            "log_level", default_value="info", description="ROS 2 log level."
        ),
        Node(
            package="steve_perception",
            executable="perception_node",
            name="perception_node",
            output="screen",
            parameters=[{
                "config_file": config_path,
                "publish_rgbd_cameras": publish_rgbd_cameras,
                "use_sim_time": use_sim_time,
            }],
            arguments=["--ros-args", "--log-level", log_level],
        ),
    ])
