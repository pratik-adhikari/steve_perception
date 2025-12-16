"""
rtabmap_front_static_rgbd.launch.py

Goal:
- Use the upstream, maintained "master" launch file:
    rtabmap_launch/rtabmap.launch.py
  but expose a clean interface for YOUR robot topics.

Core idea (important!):
- Upstream launch file = wiring harness (topics, relays, odom nodes, viz nodes).
- Your wrapper = sets ONLY arguments (topic names, toggles, frames, QoS).
- RTAB-Map tuning = goes into cfg (*.ini), NOT here.

How to think:
- If you want to change WHICH sensors are used -> change args here (subscribe_scan, rgbd_sync, visual_odometry, etc.)
- If you want to change HOW RTAB behaves internally -> change the .ini file (cfg).
- If you find yourself wanting to modify the upstream file -> stop. Wrap it harder.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource

from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    # --------- Your actual robot topics (front RGB-D camera) ----------
    # These match what you showed in `ros2 topic list`:
    # /front_camera/front_camera/image_raw
    # /front_camera/front_camera/depth/image_raw
    # /front_camera/front_camera/camera_info
    #
    # Note: If your depth camera_info is separate (some drivers do that),
    # RTAB's rgbd_sync uses ONLY the RGB camera_info (it assumes depth is registered).
    # If your depth is NOT registered, mapping will look "warped".
    rgb_topic_default = "/front_camera/front_camera/image_raw"
    depth_topic_default = "/front_camera/front_camera/depth/image_raw"
    camera_info_default = "/front_camera/front_camera/camera_info"

    # --------- Frames ----------
    # Choose base_link or base_footprint depending on what TF you have.
    # You showed base_link exists (and is parented to odom).
    frame_id_default = "base_link"
    map_frame_default = "map"
    odom_frame_default = "odom"

    # --------- Config file ----------
    # This is the RTAB-Map *.ini file (NOT YAML).
    # We'll provide it in steve_perception/config/rtabmap_front_static_rgbd.ini
    cfg_default = os.path.join(
        get_package_share_directory("steve_perception"),
        "config",
        "rtabmap_front_static_rgbd.ini"
    )

    # --------- Upstream launch ----------
    upstream = os.path.join(
        get_package_share_directory("rtabmap_launch"),
        "launch",
        "rtabmap.launch.py"
    )

    # --------- Launch arguments you may want to override from CLI ----------
    # We expose them so YOU can experiment without editing the file.
    declare_args = [
        DeclareLaunchArgument("use_sim_time", default_value="true",
                              description="Use Gazebo clock."),
        DeclareLaunchArgument("rtabmap_viz", default_value="true",
                              description="Start rtabmap_viz GUI."),
        DeclareLaunchArgument("rviz", default_value="false",
                              description="Start RViz2 (optional)."),

        DeclareLaunchArgument("frame_id", default_value=frame_id_default,
                              description="Robot base frame."),
        DeclareLaunchArgument("map_frame_id", default_value=map_frame_default,
                              description="Map frame."),
        DeclareLaunchArgument("odom_frame_id", default_value=odom_frame_default,
                              description="Odometry frame."),

        DeclareLaunchArgument("rgb_topic", default_value=rgb_topic_default,
                              description="Front camera RGB image."),
        DeclareLaunchArgument("depth_topic", default_value=depth_topic_default,
                              description="Front camera depth image."),
        DeclareLaunchArgument("camera_info_topic", default_value=camera_info_default,
                              description="Front camera RGB camera_info."),

        DeclareLaunchArgument("cfg", default_value=cfg_default,
                              description="RTAB-Map *.ini config path."),
    ]

    # --------- The actual decisions (these are the knobs you SHOULD understand) ----------
    #
    # 1) rgbd_sync:
    #    If true, upstream will run rtabmap_sync/rgbd_sync to produce /rgbd_image.
    #    This is usually the cleanest setup unless your camera already publishes RGBDImage.
    #
    # 2) subscribe_rgbd:
    #    Must match whether rgbd_sync is used.
    #
    # 3) visual_odometry vs icp_odometry:
    #    For your current “front RGB-D only” setup:
    #      - Start with visual_odometry=true, icp_odometry=false
    #    Later, if you add 2D lidar ICP odom:
    #      - visual_odometry=false, icp_odometry=true, subscribe_scan=true, scan_topic:=...
    #
    # 4) approx_sync:
    #    In simulation you can often use exact sync, but many camera pipelines are slightly jittery.
    #    Use approx_sync=true with a small max interval.
    #
    # 5) compressed:
    #    Don’t compress while debugging. Compression adds latency and hides timestamp issues.
    #

    upstream_args = {
        # Time
        "use_sim_time": LaunchConfiguration("use_sim_time"),

        # Visualizers
        "rtabmap_viz": LaunchConfiguration("rtabmap_viz"),
        "rviz": LaunchConfiguration("rviz"),

        # Frames
        "frame_id": LaunchConfiguration("frame_id"),
        "map_frame_id": LaunchConfiguration("map_frame_id"),
        "odom_frame_id": LaunchConfiguration("odom_frame_id"),

        # RGB-D inputs
        "rgb_topic": LaunchConfiguration("rgb_topic"),
        "depth_topic": LaunchConfiguration("depth_topic"),
        "camera_info_topic": LaunchConfiguration("camera_info_topic"),

        # We want upstream to create rgbd_image for us
        "rgbd_sync": "true",
        "subscribe_rgbd": "true",

        # Synchronization
        "approx_sync": "true",
        "approx_sync_max_interval": "0.02",   # (sec) tighten in sim; loosen if needed

        # Don’t use stereo mode
        "stereo": "false",

        # Odometry: start with visual odom (camera-based)
        "visual_odometry": "true",
        "icp_odometry": "false",

        # No lidar constraints in this "front-camera-only" version
        "subscribe_scan": "false",
        # "scan_topic": "/lidar_1/scan",  # (commented) enable later with icp_odometry

        # Don’t subscribe to scan_cloud either
        "subscribe_scan_cloud": "false",
        # "scan_cloud_topic": "/scan_cloud",  # (commented) you don't publish this now

        # No IMU right now
        # "imu_topic": "/imu/data",          # (commented) enable if you have IMU + correct TF

        # Data transport
        "compressed": "false",

        # Tuning file
        "cfg": LaunchConfiguration("cfg"),

        # Debug / quality-of-life
        "qos": "0",  # system default
        # "qos": "1", # (commented) force reliable if you see drops
        # "qos": "2", # (commented) best effort if camera driver is best-effort

        # Delete old db at each run while experimenting (otherwise you "merge" sessions)
        # NOTE: this is passed in RTAB's CLI args string.
        "args": "--delete_db_on_start",

        # Where DB is stored (keep it stable; if you switch robots/maps, change this)
        "database_path": "~/.ros/rtabmap_front_static.db",
    }

    include_upstream = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(upstream),
        launch_arguments=upstream_args.items()
    )

    return LaunchDescription(declare_args + [include_upstream])
