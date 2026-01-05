"""Launches PerceptionNode and RTAB-Map exclusively for the Pan-Tilt camera."""

from __future__ import annotations
import os
from pathlib import Path
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, LogInfo
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition
from launch_ros.actions import Node
import shutil
import yaml

def _create_work_copy(src_path: str) -> str:
    """Create a working copy of the INI file to prevent overwriting the source."""
    if not os.path.exists(src_path):
        return src_path # Let RTAB-Map handle missing file error or create default

    # Create a copy in the same directory but with _autosave suffix
    dirname, basename = os.path.split(src_path)
    name, ext = os.path.splitext(basename)
    dst_name = f"{name}_autosave{ext}"
    dst_path = os.path.join(dirname, dst_name)
    
    try:
        shutil.copy2(src_path, dst_path)
        print(f"[mapping_pan_tilt] Created config work-copy: {dst_path}")
        return dst_path
    except Exception as e:
        print(f"[mapping_pan_tilt] Failed to create work-copy: {e}. Using original.")
        return src_path
def _load_yaml(path: str) -> dict:
    """Load YAML configuration file."""
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}

def _pkg_share_from_this_launch() -> str:
    """Return `<pkg_share>/steve_perception`."""
    launch_dir = Path(__file__).resolve().parent
    return str(launch_dir.parent.resolve())

def _launch_setup(context, *args, **kwargs):
    """Build launch nodes for Pan-Tilt mapping."""
    pkg_share = _pkg_share_from_this_launch()

    # --- Configuration ---
    cfg_path = LaunchConfiguration("mapping_config").perform(context)
    cfg = _load_yaml(cfg_path)
    
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

    # Single Camera Setup for Pan-Tilt
    pan_tilt_rgb_topic = "/steve_perception/pan_tilt/rgbd_image"
    scan_topic = str(rtab.get("scan_topic", "/scan"))
    subscribe_scan = bool(rtab.get("subscribe_scan", False))

    # INI file
    ini_file = str(rtab.get("ini_file", "rtabmap_front_rgbd.ini"))
    
    # Logic to support "edit-in-place" without rebuilding:
    # 1. Check if it's an absolute path
    # 2. Check if it exists relative to the mapping_config file (e.g. in src/)
    # 3. Fallback to installed package share
    
    if os.path.isabs(ini_file) and os.path.exists(ini_file):
        ini_path = ini_file
    else:
        # Try finding it alongside the mapping_config file
        cfg_dir = os.path.dirname(cfg_path)
        potential_path = os.path.join(cfg_dir, ini_file)
        if os.path.exists(potential_path):
            ini_path = potential_path
        else:
            # Fallback
            ini_path = os.path.join(pkg_share, "config", ini_file)
            
    # Log which file we are using to be explicit
    print(f"[mapping_pan_tilt] Found source INI: {ini_path}")

    # Create work copy to prevent overwriting source
    work_ini_path = _create_work_copy(ini_path)
    print(f"[mapping_pan_tilt] RTAB-Map will use: {work_ini_path}")

    # Logging
    lvl_perception = str(logging_cfg.get("steve_perception", "info"))
    lvl_rtabmap = str(logging_cfg.get("rtabmap", "info"))
    lvl_viz = str(logging_cfg.get("rtabmap_viz", "warn"))

    actions = [
        LogInfo(msg=f"[steve_perception] Mapping Pan-Tilt Only mode"),
        LogInfo(msg=f"[steve_perception] DB: {database_path}"),
        
        # 1. Perception Node (Pan-Tilt only)
        Node(
            package="steve_perception",
            executable="perception_node",
            name="steve_perception",
            output="screen",
            arguments=["--ros-args", "--log-level", lvl_perception],
            parameters=[{
                "config_file": "steve.yaml",
                "use_sim_time": use_sim_time,
                "publish_rgbd": True,
                "enabled_cameras": ["pan_tilt"], 
            }],
        ),
    ]

    # 2. Pan-Tilt Control Node
    # Determine if we should suppress feedback logs based on launch arg
    # If log_pan_tilt_feedback is 'false', we append '--no-log-feedback'
    pan_tilt_node_args = []
    from launch.substitutions import PythonExpression
    
    should_log = LaunchConfiguration("log_pan_tilt_feedback").perform(context).lower() == "true"
    if not should_log:
        pan_tilt_node_args.append("--no-log-feedback")

    actions.append(
        Node(
            condition=IfCondition(LaunchConfiguration("pan_tilt_sweep")),
            package="steve_perception",
            executable="pan_tilt_control",
            name="pan_tilt_control",
            output="screen",
            arguments=pan_tilt_node_args, 
            parameters=[{
                "pan": LaunchConfiguration("pan_limit"),
                "tilt": LaunchConfiguration("tilt_limit"),
                "speed": LaunchConfiguration("pan_tilt_speed"),
                "sweep": LaunchConfiguration("pan_tilt_sweep")
            }]
        )
    )

    # 3. RTAB-Map Odometry (using rgbd_odometry)
    odom_params = {
        "use_sim_time": use_sim_time,
        "frame_id": base_frame,
        "odom_frame_id": odom_frame,
        "publish_tf": False, # Usually sim provides odom, or we let robot odom handle it. If visual odom is primary, set True.
        "subscribe_rgbd": True,
        "approx_sync": True,
        "sync_queue_size": sync_q,
        "topic_queue_size": topic_q,
        "wait_for_transform": wait_for_transform,
    }
    
    actions.append(
        Node(
            package="rtabmap_odom",
            executable="rgbd_odometry",
            name="rgbd_odometry",
            output="screen",
            parameters=[odom_params],
            remappings=[
                ("rgbd_image", pan_tilt_rgb_topic), 
                ("odom", "/odom")
            ],
        )
    )

    # 4. RTAB-Map SLAM
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
        "sync_queue_size": sync_q,
        "topic_queue_size": topic_q,
        "wait_for_transform": wait_for_transform,
        "config_path": work_ini_path,
        "database_path": database_path,
        "delete_db_on_start": LaunchConfiguration("delete_db_on_start"),
        "rgbd_cameras": 1, # Explicitly single camera
    }

    actions.append(
        Node(
            package="rtabmap_slam",
            executable="rtabmap",
            name="rtabmap",
            output="screen",
            arguments=["--ros-args", "--log-level", lvl_rtabmap],
            parameters=[slam_params],
            remappings=[
                ("rgbd_image", pan_tilt_rgb_topic),
                ("odom", "/odom"),
                ("scan", scan_topic)
            ],
        )
    )

    # 5. Visualization
    viz_params = {
        "use_sim_time": use_sim_time,
        "frame_id": base_frame,
        "odom_frame_id": odom_frame,
        "subscribe_rgbd": True,
        "subscribe_odom_info": True,
        "approx_sync": True,
        "sync_queue_size": sync_q,
        "rgbd_cameras": 1,
    }

    actions.append(
        Node(
            condition=IfCondition(LaunchConfiguration("rtabmap_viz")),
            package="rtabmap_viz",
            executable="rtabmap_viz",
            name="rtabmap_viz",
            output="screen",
            arguments=["--ros-args", "--log-level", lvl_viz],
            parameters=[viz_params],
            remappings=[
                ("rgbd_image", pan_tilt_rgb_topic),
                ("odom", "/odom"),
                ("scan", scan_topic)
            ],
        )
    )

    return actions

def generate_launch_description():
    pkg_share = _pkg_share_from_this_launch()
    default_cfg = os.path.join(pkg_share, "config", "mapping.yaml")

    return LaunchDescription([
        DeclareLaunchArgument(
            "mapping_config",
            default_value=default_cfg,
            description="Path to a mapping_*.yaml profile.",
        ),
        DeclareLaunchArgument(
            "delete_db_on_start",
            default_value="true",
            description="Delete previous DB on startup.",
        ),
        DeclareLaunchArgument(
            "rtabmap_viz",
            default_value="true",
            description="Launch RTAB-Map Visualization GUI.",
        ),
        DeclareLaunchArgument(
            "pan_limit",
            default_value="4.0",
            description="Pan limit/amplitude in degrees for scanner. (URDF Limit: -180 to 20 deg)",
        ),
        DeclareLaunchArgument(
            "tilt_limit",
            default_value="20.0",
            description="Tilt limit/amplitude in degrees for scanner. (URDF Limit: -180 to 20 deg)",
        ),
        DeclareLaunchArgument(
            "pan_tilt_speed",
            default_value="2.0",
            description="Movement speed in deg/s.",
        ),
        DeclareLaunchArgument(
            "pan_tilt_sweep",
            default_value="false",
            description="Enable continuous elliptical sweep.",
        ),
        DeclareLaunchArgument(
            "log_pan_tilt_feedback",
            default_value="false",
            description="Enable or disable feedback logging from pan-tilt controller.",
        ),
        OpaqueFunction(function=_launch_setup),
    ])
