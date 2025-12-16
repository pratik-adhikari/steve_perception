#!/usr/bin/env python3
"""
rtabmap_front_rgbd.launch.py

Goal:
- Use the *official* rtabmap_launch/rtabmap.launch.py (the “master” launch)
- Provide only the minimal arguments needed for *our* front RGB-D camera setup

Why this exists:
- You want a stable upstream base (rtabmap_launch) so you don’t re-implement logic.
- You want a small, readable adapter you can own in steve_perception.

Key design:
- We DO NOT start any camera driver here.
  Gazebo already publishes:
    /front_camera/front_camera/image_raw
    /front_camera/front_camera/depth/image_raw
    /front_camera/front_camera/camera_info

- We DO NOT subscribe to /scan yet (kept as commented options).
- We DO use approximate sync initially (simulation timing jitter is common).

Where to tune SLAM behavior:
- Use the cfg INI file (rtabmap_front_rgbd.ini) instead of stuffing 30 params here.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # --- Topics (your Gazebo camera publishes these) ---
    rgb_topic_default = "/front_camera/front_camera/image_raw"
    depth_topic_default = "/front_camera/front_camera/depth/image_raw"
    camera_info_topic_default = "/front_camera/front_camera/camera_info"

    # You measured that all frame_ids are front_camera_optical_frame.
    # RTAB-Map's 'frame_id' should be the robot base frame, NOT the camera frame.
    # In your TF tree you have base_link, odom, map, etc.
    frame_id_default = "base_link"

    # Put your INI config in steve_perception/config
    default_cfg = PathJoinSubstitution([
        FindPackageShare("steve_perception"),
        "config",
        "rtabmap_front_rgbd.ini"
    ])

    # --- Include the upstream “master” launch file ---
    rtabmap_master_launch = PathJoinSubstitution([
        FindPackageShare("rtabmap_launch"),
        "launch",
        "rtabmap.launch.py"
    ])

    return LaunchDescription([
        # ===== Arguments you may want to override at runtime =====
        DeclareLaunchArgument("rgb_topic", default_value=rgb_topic_default),
        DeclareLaunchArgument("depth_topic", default_value=depth_topic_default),
        DeclareLaunchArgument("camera_info_topic", default_value=camera_info_topic_default),

        DeclareLaunchArgument("frame_id", default_value=frame_id_default),

        # If you play rosbags with /clock, set this true.
        DeclareLaunchArgument("use_sim_time", default_value="true"),

        # Where RTAB-Map stores the database (map). In sim you may want a temp path.
        DeclareLaunchArgument("database_path", default_value="~/.ros/rtabmap_front_rgbd.db"),

        # Our INI file: RTAB-Map internal parameters live here.
        DeclareLaunchArgument("cfg", default_value=default_cfg),

        # ===== The actual include of rtabmap_launch =====
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(rtabmap_master_launch),
            launch_arguments={
                # Core frames
                "frame_id": LaunchConfiguration("frame_id"),
                "map_frame_id": "map",
                "odom_frame_id": "odom",
                "publish_tf_map": "true",

                # Input topics (RGB-D)
                "rgb_topic": LaunchConfiguration("rgb_topic"),
                "depth_topic": LaunchConfiguration("depth_topic"),
                "camera_info_topic": LaunchConfiguration("camera_info_topic"),

                # We are using raw topics (not /compressed)
                "compressed": "false",

                # We are NOT using stereo
                "stereo": "false",

                # Let rtabmap_launch create rgbd_sync (recommended for clarity)
                # If you already have an rgbd_sync node elsewhere, set rgbd_sync:=false and subscribe_rgbd:=true.
                "rgbd_sync": "true",
                "subscribe_rgbd": "true",

                # Simulation timing jitter: start with approximate sync
                "approx_sync": "true",
                # If you see “dropped messages” or delay, increase slightly (e.g. 0.05)
                "approx_sync_max_interval": "0.02",

                # Odometry strategy:
                # Start simple: visual odometry from RGB-D
                "visual_odometry": "true",
                "icp_odometry": "false",
                "odom_topic": "odom",

                # LiDAR integration (NOT enabled yet — kept to teach you the knobs)
                # "subscribe_scan": "true",
                # "scan_topic": "/lidar_1/scan",
                # If you later turn on icp_odometry, you’ll often use scan here and disable visual_odometry.

                # Config INI
                "cfg": LaunchConfiguration("cfg"),

                # Visualizers (turn off if you want performance)
                "rtabmap_viz": "true",
                "rviz": "false",

                # Useful while iterating:
                # Set args to include -d if you want to delete db on start (clean runs).
                # WARNING: -d wipes previous mapping results.
                "args": "--delete_db_on_start",
                # You can also add RTAB-Map verbosity:
                # "args": "--delete_db_on_start --udebug",

                # Logging
                "log_level": "info",
            }.items(),
        ),
    ])
