#!/usr/bin/env python3
"""
Two RGB-D cameras + LiDAR ICP odometry + RTAB-Map SLAM.

WHY YOU GOT AN ERROR BEFORE:
- LaunchConfiguration() objects are *Substitutions* evaluated at launch-time.
- You cannot call .perform({}) with a plain dict or use them like normal strings.
- If you need conditional behavior, use ROS2 launch Conditions (IfCondition/UnlessCondition)
  and/or PythonExpression, not plain Python if-statements.

PIPELINE:
  front RGBD  -> rgbd_sync -> /rtabmap/rgbd_image0 \
                                               -> rtabmap (rgbd_cameras=2)
  rear  RGBD  -> rgbd_sync -> /rtabmap/rgbd_image1 /
  lidar scan  -> icp_odometry -> /odom ----------^

NOTES:
- Pan-tilt camera is STATIC here (no joint motion).
- We use LiDAR ICP odom to stabilize drift; visual odom is disabled.
"""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pkg_share = get_package_share_directory("steve_perception")
    rtabmap_ini = os.path.join(pkg_share, "config", "rtabmap_two_rgbd.ini")

    # --- Launch args (these are Substitutions, not strings) ---
    use_sim_time = LaunchConfiguration("use_sim_time")
    delete_db = LaunchConfiguration("delete_db_on_start")

    # --- Topics (hardcoded for now; robot-agnostic comes later) ---
    front_rgb   = "/front_camera/front_camera/image_raw"
    front_depth = "/front_camera/front_camera/depth/image_raw"
    front_info  = "/front_camera/front_camera/camera_info"

    rear_rgb    = "/pan_tilt_camera/pan_tilt_camera/image_raw"
    rear_depth  = "/pan_tilt_camera/pan_tilt_camera/depth/image_raw"
    rear_info   = "/pan_tilt_camera/pan_tilt_camera/camera_info"

    scan_topic  = "/lidar_1/scan"

    base_frame  = "base_link"
    odom_frame  = "odom"
    map_frame   = "map"

    return LaunchDescription([
        DeclareLaunchArgument(
            "use_sim_time", default_value="true",
            description="Use /clock (Gazebo) if true."
        ),
        DeclareLaunchArgument(
            "delete_db_on_start", default_value="true",
            description="Delete previous RTAB-Map DB on start (mapping/tuning)."
        ),

        # -----------------------------
        # RGBD Sync (Front)
        # -----------------------------
        Node(
            package="rtabmap_sync",
            executable="rgbd_sync",
            name="rgbd_sync_front",
            output="screen",
            parameters=[{
                "use_sim_time": use_sim_time,
                "approx_sync": True,
                "sync_queue_size": 30,
                "topic_queue_size": 30,
            }],
            remappings=[
                ("rgb/image", front_rgb),
                ("depth/image", front_depth),
                ("rgb/camera_info", front_info),
                ("rgbd_image", "/rtabmap/rgbd_image0"),
            ],
        ),

        # -----------------------------
        # RGBD Sync (Rear / PanTilt camera)
        # -----------------------------
        Node(
            package="rtabmap_sync",
            executable="rgbd_sync",
            name="rgbd_sync_rear",
            output="screen",
            parameters=[{
                "use_sim_time": use_sim_time,
                "approx_sync": True,
                "sync_queue_size": 30,
                "topic_queue_size": 30,
            }],
            remappings=[
                ("rgb/image", rear_rgb),
                ("depth/image", rear_depth),
                ("rgb/camera_info", rear_info),
                ("rgbd_image", "/rtabmap/rgbd_image1"),
            ],
        ),

        # -----------------------------
        # ICP Odometry from 2D LiDAR scan
        # -----------------------------
        Node(
            package="rtabmap_odom",
            executable="icp_odometry",
            name="icp_odometry",
            output="screen",
            parameters=[{
                "use_sim_time": use_sim_time,
                "frame_id": base_frame,
                "odom_frame_id": odom_frame,
                "publish_tf": True,

                "subscribe_scan": True,
                "subscribe_scan_cloud": False,

                # Planar ground robot assumption
                "Reg/Force3DoF": "true",

                # Conservative tuning (adjust later)
                "Icp/VoxelSize": "0.05",
                "Icp/MaxCorrespondenceDistance": "0.5",
                "Icp/Iterations": "10",
                "Icp/PointToPlane": "false",
            }],
            remappings=[
                ("scan", scan_topic),
                ("odom", "/odom"),
            ],
        ),

        # -----------------------------
        # RTAB-Map (Mapping) - WITHOUT deleting DB
        # -----------------------------
        Node(
            condition=UnlessCondition(delete_db),
            package="rtabmap_slam",
            executable="rtabmap",
            name="rtabmap",
            output="screen",
            parameters=[{
                "use_sim_time": use_sim_time,

                "frame_id": base_frame,
                "odom_frame_id": odom_frame,
                "map_frame_id": map_frame,
                "publish_tf": True,

                "subscribe_rgbd": True,
                "rgbd_cameras": 2,
                "subscribe_odom": True,

                "approx_sync": True,
                "sync_queue_size": 30,
                "topic_queue_size": 30,
                "wait_for_transform": 0.2,

                "config_path": rtabmap_ini,
            }],
            remappings=[
                ("rgbd_image0", "/rtabmap/rgbd_image0"),
                ("rgbd_image1", "/rtabmap/rgbd_image1"),
                ("odom", "/odom"),
            ],
        ),

        # -----------------------------
        # RTAB-Map (Mapping) - WITH deleting DB (-d)
        # -----------------------------
        Node(
            condition=IfCondition(delete_db),
            package="rtabmap_slam",
            executable="rtabmap",
            name="rtabmap",
            output="screen",
            arguments=["-d"],
            parameters=[{
                "use_sim_time": use_sim_time,

                "frame_id": base_frame,
                "odom_frame_id": odom_frame,
                "map_frame_id": map_frame,
                "publish_tf": True,

                "subscribe_rgbd": True,
                "rgbd_cameras": 2,
                "subscribe_odom": True,

                "approx_sync": True,
                "sync_queue_size": 30,
                "topic_queue_size": 30,
                "wait_for_transform": 0.2,

                "config_path": rtabmap_ini,
            }],
            remappings=[
                ("rgbd_image0", "/rtabmap/rgbd_image0"),
                ("rgbd_image1", "/rtabmap/rgbd_image1"),
                ("odom", "/odom"),
            ],
        ),

        # -----------------------------
        # RTAB-Map Viz (optional but useful)
        # -----------------------------
        Node(
            package="rtabmap_viz",
            executable="rtabmap_viz",
            name="rtabmap_viz",
            output="screen",
            parameters=[{
                "use_sim_time": use_sim_time,
                "frame_id": base_frame,
                "odom_frame_id": odom_frame,
                "subscribe_odom_info": True,
                "subscribe_rgbd": True,
                "rgbd_cameras": 2,
            }],
            remappings=[
                ("rgbd_image0", "/rtabmap/rgbd_image0"),
                ("rgbd_image1", "/rtabmap/rgbd_image1"),
                ("odom", "/odom"),
            ],
        ),
    ])
