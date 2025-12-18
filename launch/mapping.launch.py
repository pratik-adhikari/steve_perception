"""Launches PerceptionNode plus RTAB-Map mapping."""

from __future__ import annotations
import os
import re
import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, IncludeLaunchDescription, LogInfo
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition
from launch_ros.actions import Node

def _load_yaml(path: str) -> dict:
    """Load YAML configuration file."""
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}

def _rgbd_remaps(rgbd_topics: list[str]) -> list[tuple[str, str]]:
    """Build topic remapping rules for RTAB-Map RGBD inputs."""
    if len(rgbd_topics) == 1:
        return [("rgbd_image", rgbd_topics[0])]
    return [(f"rgbd_image{i}", t) for i, t in enumerate(rgbd_topics)]

def _rgbd_images_topic() -> str:
    """Multi-camera RGBDImages topic for rgbdx_sync output."""
    return "/steve_perception/rgbd_images"

def _csv(cams: list[str]) -> str:
    """Convert camera list to comma-separated string."""
    return ",".join([c.strip() for c in cams if str(c).strip()])

def _launch_setup(context, *args, **kwargs):
    """Build launch nodes based on mapping configuration."""
    cfg_path = LaunchConfiguration("mapping_config").perform(context)
    cfg = _load_yaml(cfg_path)

    # Parse configuration sections
    perception_cfg = cfg.get("perception", {}) or {}
    rtab = cfg.get("rtabmap", {}) or {}
    viz = cfg.get("viz", {}) or {}
    logging_cfg = cfg.get("logging", {}) or {}

    # Database directory setup
    output_dir = str(rtab.get("output_dir", "")).strip()
    if not output_dir:
        output_dir = os.path.join(os.path.expanduser("~"), ".ros", "steve_maps")
    output_dir = os.path.expanduser(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    database_name = str(rtab.get("database_name", "rtabmap.db")).strip()
    database_path = os.path.join(output_dir, database_name)

    # TF frames and timing
    use_sim_time = bool(rtab.get("use_sim_time", True))
    base_frame = str(rtab.get("base_frame", "base_link"))
    odom_frame = str(rtab.get("odom_frame", "odom"))
    map_frame = str(rtab.get("map_frame", "map"))
    wait_for_transform = float(rtab.get("wait_for_transform", 0.2))

    # RGBD topic configuration
    rgbd_topics = rtab.get("rgbd_topics", [])
    rgbd_topics = [str(t) for t in rgbd_topics]
    if not rgbd_topics:
        rgbd_topics = [str(rtab.get("rgbd_topic", "/steve_perception/front/rgbd_image"))]
    rgbd_topics = [str(t) for t in rgbd_topics]

    # Extract camera names from topics for perception node
    cams: list[str] = []
    pat = re.compile(r"^/steve_perception/([^/]+)/rgbd_image$")
    for t in rgbd_topics:
        m = pat.match(t)
        if m:
            cams.append(m.group(1))
    publish_rgbd_cameras_csv = ",".join(cams)

    # Multi-camera sync mode
    rgbd_cameras = len(rgbd_topics)
    use_rgbd_images_interface = rgbd_cameras > 1

    subscribe_scan = bool(rtab.get("subscribe_scan", False))
    scan_topic = str(rtab.get("scan_topic", "/scan"))

    pkg_share = get_package_share_directory("steve_perception")
    ini_file = str(rtab.get("ini_file", "rtabmap_front_rgbd.ini"))
    ini_path = os.path.join(pkg_share, "config", ini_file)

    # Logging levels per node
    lvl_perception = str(logging_cfg.get("steve_perception", logging_cfg.get("perception_node", "info")))
    lvl_odom = str(logging_cfg.get("rgbd_odometry", "info"))
    lvl_rtabmap = str(logging_cfg.get("rtabmap", "info"))
    lvl_viz = str(logging_cfg.get("rtabmap_viz", "warn"))

    # Perception node launch
    perception_launch = os.path.join(pkg_share, "launch", "perception.launch.py")
    perception_config_file = str(perception_cfg.get("config_file", "steve.yaml"))
    publish_rgbd_master = bool(perception_cfg.get("publish_rgbd", True))

    actions = [
        LogInfo(msg=f"[steve_perception] Mapping YAML: {cfg_path}"),
        LogInfo(msg=f"[steve_perception] RTAB-Map DB: {database_path}"),
        LogInfo(msg=f"[steve_perception] RTAB-Map INI: {ini_path}"),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(perception_launch),
            launch_arguments={
                "config_file": perception_config_file,
                "publish_rgbd_cameras": publish_rgbd_cameras_csv if publish_rgbd_master else "",
                "use_sim_time": str(use_sim_time).lower(),
                "log_level": lvl_perception,
            }.items(),
        ),
    ]

    # Add rgbdx_sync for multi-camera setups
    if use_rgbd_images_interface:
        actions.append(
            Node(
                package="rtabmap_sync",
                executable="rgbdx_sync",
                name="rgbdx_sync",
                output="screen",
                arguments=["--ros-args", "--log-level", logging_cfg.get("rgbdx_sync", "warn")],
                parameters=[{
                    "approx_sync": True,
                    "sync_queue_size": 30,
                    "topic_queue_size": 30,
                    "use_sim_time": use_sim_time,
                }],
                remappings=_rgbd_remaps(rgbd_topics) + [("rgbd_images", _rgbd_images_topic())],
            )
        )

    # Visual odometry configuration
    odom_params = {
        "use_sim_time": use_sim_time,
        "frame_id": base_frame,
        "odom_frame_id": odom_frame,
        "publish_tf": False,
        "subscribe_rgbd": True,
        "approx_sync": True,
        "sync_queue_size": 30,
        "topic_queue_size": 30,
        "wait_for_transform": wait_for_transform,
    }
    odom_rgbd_topic = rgbd_topics[0]

    actions.append(
        Node(
            package="rtabmap_odom",
            executable="rgbd_odometry",
            name="rgbd_odometry",
            output="screen",
            arguments=["--ros-args", "--log-level", lvl_odom],
            parameters=[odom_params],
            remappings=[("rgbd_image", odom_rgbd_topic), ("odom", "/odom")],
        )
    )

    # RTAB-Map SLAM configuration
    slam_params = {
        "use_sim_time": use_sim_time,
        "frame_id": base_frame,
        "odom_frame_id": odom_frame,
        "map_frame_id": map_frame,
        "publish_tf": True,
        "subscribe_rgbd": True,
        "subscribe_odom": True,
        "subscribe_scan": subscribe_scan,
        "approx_sync": True,
        "sync_queue_size": 30,
        "topic_queue_size": 30,
        "wait_for_transform": wait_for_transform,
        "config_path": ini_path,
        "database_path": database_path,
        "delete_db_on_start": LaunchConfiguration("delete_db_on_start"),
    }

    if use_rgbd_images_interface:
        slam_params["rgbd_cameras"] = 0
    elif rgbd_cameras > 1:
        slam_params["rgbd_cameras"] = rgbd_cameras

    actions.append(
        Node(
            package="rtabmap_slam",
            executable="rtabmap",
            name="rtabmap",
            output="screen",
            arguments=["--ros-args", "--log-level", lvl_rtabmap],
            parameters=[slam_params],
            remappings=(
                ([('rgbd_images', _rgbd_images_topic())] if use_rgbd_images_interface else _rgbd_remaps(rgbd_topics))
                + [("odom", "/odom"), ("scan", scan_topic)]
            ),
        )
    )

    # Visualization node configuration
    viz_params = {
        "use_sim_time": use_sim_time,
        "frame_id": base_frame,
        "odom_frame_id": odom_frame,
        "subscribe_rgbd": True,
        "subscribe_odom_info": True,
        "approx_sync": True,
        "sync_queue_size": 30,
    }

    if use_rgbd_images_interface:
        viz_params["rgbd_cameras"] = 0
    elif rgbd_cameras > 1:
        viz_params["rgbd_cameras"] = rgbd_cameras

    actions.append(
        Node(
            condition=IfCondition(LaunchConfiguration("rtabmap_viz")),
            package="rtabmap_viz",
            executable="rtabmap_viz",
            name="rtabmap_viz",
            output="screen",
            arguments=["--ros-args", "--log-level", lvl_viz],
            parameters=[viz_params],
            remappings=(
                ([('rgbd_images', _rgbd_images_topic())] if use_rgbd_images_interface else _rgbd_remaps(rgbd_topics))
                + [("odom", "/odom"), ("scan", scan_topic)]
            ),
        )
    )

    return actions

def generate_launch_description():
    """ROS 2 launch entry point."""
    pkg_share = get_package_share_directory("steve_perception")
    default_cfg = os.path.join(pkg_share, "config", "mapping.yaml")
    delete_db_default = "true"
    viz_default = "true"

    try:
        cfg = _load_yaml(default_cfg)
        delete_db_default = str(bool(cfg.get("rtabmap", {}).get("delete_db_on_start", True))).lower()
        viz_default = str(bool(cfg.get("viz", {}).get("rtabmap_viz", True))).lower()
    except Exception:
        pass

    return LaunchDescription([
        DeclareLaunchArgument(
            "mapping_config",
            default_value=default_cfg,
            description="Path to a mapping_*.yaml profile (steve_perception/config).",
        ),
        DeclareLaunchArgument(
            "delete_db_on_start",
            default_value=delete_db_default,
            description="If true, RTAB-Map deletes previous DB on startup.",
        ),
        DeclareLaunchArgument(
            "rtabmap_viz",
            default_value=viz_default,
            description="Launch rtabmap_viz.",
        ),
        OpaqueFunction(function=_launch_setup),
    ])
