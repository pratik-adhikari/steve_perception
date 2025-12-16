# Program dedicated to launch 2 realsense cameras. D405 were tested!
# (Please note that this program can manage maximum 4 depth cameras)
#
# Program modified by: Adrian Ricardez (https://github.com/adricort)
# Date: 07.07.2023
# Deutsches Zentrum für Luft- und Raumfahrt
#
# Requirements:
# Be sure that you did the build on your rtabmap workspace with the -DRTABMAP_SYNC_MULTI_RGBD=ON parameter.
#
# Launching the 2 realsense cameras (change your serial numbers):
#   $ ros2 launch realsense2_camera rs_multi_camera_launch.py pointcloud.enable1:=true pointcloud.enable2:=true filters:=colorizer align_depth:=true serial_no1:=_128422271521 serial_no2:=_128422272518
#
# Running the static publishers depending on the position of your cameras:
#   $ ros2 run tf2_ros static_transform_publisher --x 0.039 --y 0 --z 0 --yaw 0 --pitch 0 --roll 1.5708 --frame-id base_link --child-frame-id camera1_link
#   $ ros2 run tf2_ros static_transform_publisher --x -0 --y 0 --z 0.02 --yaw 1.5708 --pitch -1.5708 --roll 0 --frame-id base_link --child-frame-id camera2_link
#
#   $ ros2 launch rtabmap_examples rtabmap_D405x2.launch.py
#
# You should be able to visualize now, with the right rviz config, the camera's SLAM
# Have fun!

import os
import launch
import launch_ros
from ament_index_python.packages import get_package_share_directory

# ADDED: launch argument support so we can run correctly in Gazebo (/clock) vs real robot (wall time).
# WHY: Your logs show TF_OLD_DATA + massive delays because /clock exists but use_sim_time was False.
from launch.actions import DeclareLaunchArgument  # ADDED: expose launch arg to user
from launch.substitutions import LaunchConfiguration  # ADDED: use LaunchConfiguration('use_sim_time')


def generate_launch_description():

    # ADDED: use_sim_time launch arg (default true as you requested)
    # WHY: In simulation, Gazebo publishes /clock. All nodes must use sim time or TF/message filters break.
    use_sim_time = LaunchConfiguration('use_sim_time')
    declare_use_sim_time = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',  # ADDED: default true for sim use
        description='Use simulation time if true (Gazebo publishes /clock). Set false on real hardware.'
    )

    #AG: Perception pipeline is launched separately (ros2 launch steve_perception perception.launch.py)
    # perception_launch = ... (removed to keep separate)

    config_rviz = os.path.join(
            get_package_share_directory('rtabmap_examples'), 'config', 'slam_D405x2_config.rviz')

    rviz_node = launch_ros.actions.Node(
        package='rviz2', executable='rviz2', output='screen',
        arguments=[["-d"], [config_rviz]],
        # ADDED: RViz must also follow sim time to avoid TF cache warnings when /clock is used.
        parameters=[{
            "use_sim_time": use_sim_time  # ADDED: keep RViz aligned with /clock
        }]
    )

    # REMOVED: rgbd_sync nodes (Option A: CameraAgent publishes synchronized rgbd_image)
    #AG: CameraAgent is now the single sync authority, so we remove these duplicate sync nodes.

    # RGB-D odometry
    rgbd_odometry_node = launch_ros.actions.Node(
        package='rtabmap_odom', executable='rgbd_odometry', output="screen",
        parameters=[{
            "frame_id": 'base_link',
            "odom_frame_id": 'odom',
            "publish_tf": True,
            # "approx_sync": True, # No longer needed if we subscribe to single rgbd_image? 
            # Actually, subscribe_rgbd=True typically bypasses sync if there is only 1 input.
            # But keeping it doesn't hurt.
            "approx_sync": True, #AG: Set True for simulation robustness (fix WARN: did not receive data)
            "subscribe_rgbd": True,
            "use_sim_time": use_sim_time  # ADDED: odometry must use sim time or it will drop frames / TF_OLD_DATA
            }],
        remappings=[
            # ("rgbd_image", '/realsense_camera1/rgbd_image'),
            ("rgbd_image", '/wrist/rgbd_image'),  # CHANGED: Use wrist camera (mapped from steve.yaml 'wrist') #AG: Subscribe to CameraAgent output
            ("odom", 'odom')],
        arguments=["--delete_db_on_start", ''],
        prefix='',
        namespace='rtabmap'
    )

    # SLAM
    slam_node = launch_ros.actions.Node(
        package='rtabmap_slam', executable='rtabmap', output="screen",
        parameters=[{
            "rgbd_cameras": 1, #AG: Fix FATAL error: set to 1 until rtabmap_ros is recompiles with multi-rgbd
            # "subscribe_depth": True,
            "subscribe_depth": False,  # CHANGED: rtabmap warns subscribe_depth and subscribe_rgbd can't both be true; keep rgbd only.
            "subscribe_rgbd": True,
            # "subscribe_rgb": True,
            "subscribe_rgb": False,  # CHANGED: with subscribe_rgbd=true, separate rgb subscription is unnecessary and can confuse configs.
            "subscribe_odom_info": True,
            "frame_id": 'base_link',
            "map_frame_id": 'map',
            "publish_tf": True,
            "database_path": '~/.ros/rtabmap.db',
            "approx_sync": True, #AG: Set True for simulation robustness
            "Mem/IncrementalMemory": "true",
            "Mem/InitWMWithAllNodes": "true",
            "use_sim_time": use_sim_time  # ADDED: slam node must follow sim time to avoid TF_OLD_DATA & filter drops
        }],
        remappings=[
            # ("rgbd_image0", '/realsense_camera1/rgbd_image'),
            ("rgbd_image", '/wrist/rgbd_image'), #AG: Use single input for now
            # ("rgbd_image1", '/realsense_camera2/rgbd_image'),
            # ("rgbd_image1", '/pan_tilt/rgbd_image'),  # DISABLED: multi-camera sync not supported in binary
            ("odom", 'odom')],
        arguments=["--delete_db_on_start"],
        prefix='',
        namespace='rtabmap'
    )

    voxelcloud1_node = launch_ros.actions.Node(
        package='rtabmap_util', executable='point_cloud_xyzrgb', name='point_cloud_xyzrgb1', output='screen',
        parameters=[{
            "approx_sync": True, #AG: Simulation robustness
            "subscribe_rgbd": True, # Use the synced topic
            "use_sim_time": use_sim_time  # ADDED: keep pointcloud node aligned to sim time when /clock is used
        }],
        remappings=[
            ('rgbd_image', '/wrist/rgbd_image'), #AG: subscribe to CameraAgent output
            ('cloud', 'voxel_cloud1')],
        # (no namespace in original)
        namespace='front_camera'  # CHANGED: keep outputs grouped with the front camera.
    )

    voxelcloud2_node = launch_ros.actions.Node(
        package='rtabmap_util', executable='point_cloud_xyzrgb', name='point_cloud_xyzrgb2', output='screen',
        parameters=[{
            "approx_sync": True, #AG: Simulation robustness
            "subscribe_rgbd": True, # Use the synced topic
            "use_sim_time": use_sim_time  # ADDED: keep pointcloud node aligned to sim time when /clock is used
        }],
        remappings=[
            ('rgbd_image', '/pan_tilt/rgbd_image'), #AG: subscribe to CameraAgent output
            ('cloud', 'voxel_cloud2')],
        namespace='pan_tilt_camera'  # CHANGED: keep outputs grouped with the pan-tilt camera.
    )

    return launch.LaunchDescription(
        [
        declare_use_sim_time,  # ADDED: expose use_sim_time:=true/false (default true for sim)
        # perception_launch, #AG: Launched separately
        rviz_node,
        # rgbd_sync1_node, # REMOVED
        # rgbd_sync2_node, # REMOVED
        rgbd_odometry_node,
        slam_node,
        voxelcloud1_node,
        voxelcloud2_node
        ]
    )
