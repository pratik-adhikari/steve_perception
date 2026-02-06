"""Launches PerceptionNode plus RTAB-Map mapping (single camera)."""

from __future__ import annotations
import os
import yaml
from pathlib import Path
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, LogInfo
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition
from launch_ros.actions import Node
import shutil

def _create_work_copy(src_path: str) -> str:
    """Create a working copy of the INI file to prevent overwriting the source."""
    if not os.path.exists(src_path):
        return src_path

    dirname, basename = os.path.split(src_path)
    name, ext = os.path.splitext(basename)
    dst_name = f"{name}_autosave{ext}"
    dst_path = os.path.join(dirname, dst_name)
    
    try:
        shutil.copy2(src_path, dst_path)
        return dst_path
    except Exception as e:
        print(f"[mapping] Failed to create work-copy: {e}. Using original.")
        return src_path

def _load_yaml(path: str) -> dict:
    """Load YAML configuration file."""
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}

def _pkg_share_from_this_launch() -> str:
    """Return package share directory based on this launch file location."""
    launch_dir = Path(__file__).resolve().parent
    return str(launch_dir.parent.resolve())

def _launch_setup(context, *args, **kwargs):
    """Build launch nodes based on mapping configuration."""
    cfg_path = LaunchConfiguration("mapping_config").perform(context)
    cfg = _load_yaml(cfg_path)

    perception_cfg = cfg.get("perception", {}) or {}
    rtab = cfg.get("rtabmap", {}) or {}
    logging_cfg = cfg.get("logging", {}) or {}

    # Queue sizes
    topic_q = int(rtab.get("topic_queue_size", 30))
    sync_q = int(rtab.get("sync_queue_size", 30))

    # Database setup
    output_dir = str(rtab.get("output_dir", "")).strip()
    if not output_dir:
        output_dir = os.path.join(os.path.expanduser("~"), ".ros", "steve_maps")
    output_dir = os.path.expanduser(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    database_name = str(rtab.get("database_name", "rtabmap.db")).strip()
    database_path = os.path.join(output_dir, database_name)

    # TF frames
    use_sim_time = bool(rtab.get("use_sim_time", True))
    base_frame = str(rtab.get("base_frame", "base_link"))
    odom_frame = str(rtab.get("odom_frame", "odom"))
    map_frame = str(rtab.get("map_frame", "map"))
    wait_for_transform = float(rtab.get("wait_for_transform", 0.2))

    # Single RGBD camera topic
    rgbd_topics = rtab.get("rgbd_topics", [])
    if not rgbd_topics or len(rgbd_topics) == 0:
        rgbd_topic = str(rtab.get("rgbd_topic", "/steve_perception/pan_tilt/rgbd_image"))
    else:
        rgbd_topic = str(rgbd_topics[0])  # Only use first camera

    # Lidar
    subscribe_scan = bool(rtab.get("subscribe_scan", False))
    scan_topic = str(rtab.get("scan_topic", "/scan"))

    # INI file
    pkg_share = _pkg_share_from_this_launch()
    ini_file = str(rtab.get("ini_file", "rtabmap_rgbd.ini"))
    ini_path = os.path.join(pkg_share, "config", ini_file)
    ini_path = _create_work_copy(ini_path)

    # Logging levels
    lvl_perception = str(logging_cfg.get("steve_perception", "info"))
    lvl_odom = str(logging_cfg.get("rgbd_odometry", "warn"))
    lvl_rtabmap = str(logging_cfg.get("rtabmap", "warn"))
    lvl_viz = str(logging_cfg.get("rtabmap_viz", "warn"))

    # Perception config
    perception_config_file = str(perception_cfg.get("config_file", "steve.yaml"))
    publish_rgbd = bool(perception_cfg.get("publish_rgbd", True))
    enabled_cameras = perception_cfg.get("enabled_cameras", ["pan_tilt"])

    actions = [
        LogInfo(msg=f"[steve_perception] Mapping YAML: {cfg_path}"),
        LogInfo(msg=f"[steve_perception] RTAB-Map DB: {database_path}"),
        LogInfo(msg=f"[steve_perception] RTAB-Map INI: {ini_path}"),
        LogInfo(msg=f"[steve_perception] RGBD Topic: {rgbd_topic}"),
        
        # Perception Node
        Node(
            package="steve_perception",
            executable="perception_node",
            name="steve_perception",
            output="screen",
            arguments=["--ros-args", "--log-level", lvl_perception],
            parameters=[{
                "config_file": perception_config_file,
                "use_sim_time": use_sim_time,
                "publish_rgbd": publish_rgbd,
                "enabled_cameras": enabled_cameras,
            }],
        ),

        # Visual Odometry
        Node(
            package="rtabmap_odom",
            executable="rgbd_odometry",
            name="rgbd_odometry",
            output="screen",
            arguments=["--ros-args", "--log-level", lvl_odom],
            parameters=[{
                "use_sim_time": use_sim_time,
                "frame_id": base_frame,
                "odom_frame_id": odom_frame,
                "publish_tf": False,
                "subscribe_rgbd": True,
                "approx_sync": True,
                "sync_queue_size": sync_q,
                "topic_queue_size": topic_q,
                "wait_for_transform": wait_for_transform,
            }],
            remappings=[
                ("rgbd_image", rgbd_topic),
                ("odom", "/odom"),
            ],
        ),

        # RTAB-Map SLAM
        Node(
            package="rtabmap_slam",
            executable="rtabmap",
            name="rtabmap",
            output="screen",
            arguments=["--ros-args", "--log-level", lvl_rtabmap],
            parameters=[{
                "use_sim_time": use_sim_time,
                "frame_id": base_frame,
                "odom_frame_id": odom_frame,
                "map_frame_id": map_frame,
                "publish_tf": True,
                "subscribe_rgbd": True,
                "subscribe_odom": True,
                "subscribe_scan": subscribe_scan,
                "approx_sync": True,
                "sync_queue_size": sync_q,
                "topic_queue_size": topic_q,
                "wait_for_transform": wait_for_transform,
                "config_path": ini_path,
                "database_path": database_path,
                "delete_db_on_start": LaunchConfiguration("delete_db_on_start"),
            }],
            remappings=[
                ("rgbd_image", rgbd_topic),
                ("odom", "/odom"),
                ("scan", scan_topic),
            ],
        ),

        # Visualization
        Node(
            condition=IfCondition(LaunchConfiguration("rtabmap_viz")),
            package="rtabmap_viz",
            executable="rtabmap_viz",
            name="rtabmap_viz",
            output="screen",
            arguments=["--ros-args", "--log-level", lvl_viz],
            parameters=[{
                "use_sim_time": use_sim_time,
                "frame_id": base_frame,
                "odom_frame_id": odom_frame,
                "subscribe_rgbd": True,
                "subscribe_odom_info": True,
                "approx_sync": True,
                "sync_queue_size": sync_q,
            }],
            remappings=[
                ("rgbd_image", rgbd_topic),
                ("odom", "/odom"),
                ("scan", scan_topic),
            ],
        ),
    ]

    return actions

def generate_launch_description():
    """ROS 2 launch entry point."""
    pkg_share = _pkg_share_from_this_launch()
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
            description="Path to mapping YAML config (steve_perception/config).",
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
