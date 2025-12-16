#!/usr/bin/env python3

"""Mapping (1 RGB-D camera) using CameraAgent-synchronized RGBDImage.

This launch:
1) Starts steve_perception/perception_node with `publish_rgbd:=true`.
   - CameraAgent publishes: /steve_perception/<cam>/rgbd_image
   - Frames are TF-gated at their timestamp.
2) Starts RTAB-Map RGB-D odometry + SLAM consuming that RGBDImage.

Configuration lives in: config/mapping_one_rgbd.yaml
"""

from __future__ import annotations

import os
import yaml

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from ament_index_python.packages import get_package_share_directory


def _load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def _launch_setup(context, *args, **kwargs):
    cfg_path = LaunchConfiguration("mapping_config").perform(context)
    cfg = _load_yaml(cfg_path)

    perception = cfg.get("perception", {})
    rtab = cfg.get("rtabmap", {})
    viz = cfg.get("viz", {})

    use_sim_time = bool(rtab.get("use_sim_time", True))
    delete_db_on_start = bool(rtab.get("delete_db_on_start", True))

    base_frame = str(rtab.get("base_frame", "base_link"))
    odom_frame = str(rtab.get("odom_frame", "odom"))
    map_frame = str(rtab.get("map_frame", "map"))

    rgbd_topic = str(rtab.get("rgbd_topic", "/steve_perception/front/rgbd_image"))
    wait_for_transform = float(rtab.get("wait_for_transform", 0.2))

    # INI file lives in steve_perception/config
    pkg_share = get_package_share_directory("steve_perception")
    ini_file = str(rtab.get("ini_file", "rtabmap_front_rgbd.ini"))
    ini_path = os.path.join(pkg_share, "config", ini_file)

    rtabmap_viz = bool(viz.get("rtabmap_viz", True))

    actions = []

    # --- Perception (publishes the RGBDImage topic) ---
    actions.append(
        Node(
            package="steve_perception",
            executable="perception_node",
            name="steve_perception",
            output="screen",
            parameters=[
                {
                    "use_sim_time": use_sim_time,
                    "config_file": str(perception.get("config_file", "steve.yaml")),
                    "publish_rgbd": bool(perception.get("publish_rgbd", True)),
                    "enabled_cameras": list(perception.get("enabled_cameras", ["front"])),
                }
            ],
        )
    )

    # --- RGB-D Odometry (creates /odom) ---
    actions.append(
        Node(
            package="rtabmap_odom",
            executable="rgbd_odometry",
            name="rgbd_odometry",
            output="screen",
            parameters=[
                {
                    "use_sim_time": use_sim_time,
                    "frame_id": base_frame,
                    "odom_frame_id": odom_frame,
                    "publish_tf": True,
                    "subscribe_rgbd": True,
                    # The RGBDImage is already synced by CameraAgent.
                    "approx_sync": False,
                    "wait_for_transform": wait_for_transform,
                }
            ],
            remappings=[
                ("rgbd_image", rgbd_topic),
                ("odom", "/odom"),
            ],
        )
    )

    # --- RTAB-Map SLAM ---
    slam_common = {
        "use_sim_time": use_sim_time,
        "frame_id": base_frame,
        "odom_frame_id": odom_frame,
        "map_frame_id": map_frame,
        "publish_tf": True,
        "subscribe_rgbd": True,
        "subscribe_odom": True,
        "approx_sync": False,
        "wait_for_transform": wait_for_transform,
        "config_path": ini_path,
    }

    actions.append(
        Node(
            condition=UnlessCondition(LaunchConfiguration("delete_db_on_start")),
            package="rtabmap_slam",
            executable="rtabmap",
            name="rtabmap",
            output="screen",
            parameters=[slam_common],
            remappings=[
                ("rgbd_image", rgbd_topic),
                ("odom", "/odom"),
            ],
        )
    )
    actions.append(
        Node(
            condition=IfCondition(LaunchConfiguration("delete_db_on_start")),
            package="rtabmap_slam",
            executable="rtabmap",
            name="rtabmap",
            output="screen",
            arguments=["-d"],
            parameters=[slam_common],
            remappings=[
                ("rgbd_image", rgbd_topic),
                ("odom", "/odom"),
            ],
        )
    )

    # --- Visualization (optional) ---
    actions.append(
        Node(
            condition=IfCondition(LaunchConfiguration("rtabmap_viz")),
            package="rtabmap_viz",
            executable="rtabmap_viz",
            name="rtabmap_viz",
            output="screen",
            parameters=[
                {
                    "use_sim_time": use_sim_time,
                    "frame_id": base_frame,
                    "odom_frame_id": odom_frame,
                    "subscribe_rgbd": True,
                    "subscribe_odom_info": True,
                    "approx_sync": False,
                }
            ],
            remappings=[
                ("rgbd_image", rgbd_topic),
                ("odom", "/odom"),
            ],
        )
    )

    # Defaults from YAML (so you don't have to repeat CLI args every time)
    # NOTE: LaunchConfiguration values are strings; we feed booleans through DeclareLaunchArgument defaults.
    # We set them here by returning no actions; the defaults are declared outside in generate_launch_description.
    # (The parsed `rtabmap_viz` is used only to set the default value of the launch arg.)

    return actions


def generate_launch_description():
    pkg_share = get_package_share_directory("steve_perception")
    default_cfg = os.path.join(pkg_share, "config", "mapping_one_rgbd.yaml")

    # Read YAML once to set sane default values for the CLI args (still overrideable at runtime)
    try:
        cfg = _load_yaml(default_cfg)
        delete_db_default = str(bool(cfg.get("rtabmap", {}).get("delete_db_on_start", True))).lower()
        viz_default = str(bool(cfg.get("viz", {}).get("rtabmap_viz", True))).lower()
    except Exception:
        delete_db_default = "true"
        viz_default = "true"

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "mapping_config",
                default_value=default_cfg,
                description="Path to a mapping_*.yaml profile (steve_perception/config).",
            ),
            DeclareLaunchArgument(
                "delete_db_on_start",
                default_value=delete_db_default,
                description="If true, start RTAB-Map with -d (wipes previous DB).",
            ),
            DeclareLaunchArgument(
                "rtabmap_viz",
                default_value=viz_default,
                description="Launch rtabmap_viz.",
            ),
            OpaqueFunction(function=_launch_setup),
        ]
    )
