# Program dedicated to launch 2 realsense cameras. D405 were tested!
# (Please note that this program can manage maximum 4 depth cameras)

# Program modified by: Adrian Ricardez (https://github.com/adricort)
# Date: 07.07.2023
# Deutsches Zentrum für Luft- und Raumfahrt

# Requirements:
# Be sure that you did the build on your rtabmap workspace with the -DRTABMAP_SYNC_MULTI_RGBD=ON parameter.
# NOTE (CHANGED): In your build, multi-camera handling is done by rtabmap_sync/rgbdx_sync,
# so rtabmap is run with rgbd_cameras=0 and subscribes to RGBDImages produced by rgbdx_sync.

import os
import launch
import launch_ros
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():

    config_rviz = os.path.join(
        get_package_share_directory('rtabmap_examples'), 'config', 'slam_D405x2_config.rviz')

    rviz_node = launch_ros.actions.Node(
        package='rviz2', executable='rviz2', output='screen',
        arguments=[["-d"], [config_rviz]]
    )

    rgbd_sync1_node = launch_ros.actions.Node(
        package='rtabmap_sync', executable='rgbd_sync', name='rgbd_sync1', output="screen",
        parameters=[{
            # CHANGED: simulation has timestamp skew -> use approx sync
            "approx_sync": True,
            "use_sim_time": True,
            # CHANGED: reject crazy pairings (you saw up to 0.21s)
            "approx_sync_max_interval": 0.02
        }],
        remappings=[
            # CHANGED: camera1 -> FRONT Gazebo topics
            ("rgb/image", '/front_camera/front_camera/image_raw'),
            ("depth/image", '/front_camera/front_camera/depth/image_raw'),
            ("rgb/camera_info", '/front_camera/front_camera/camera_info'),
            ("rgbd_image", 'rgbd_image')],
        # CHANGED: namespace name
        namespace='front_camera'
    )

    rgbd_sync2_node = launch_ros.actions.Node(
        package='rtabmap_sync', executable='rgbd_sync', name='rgbd_sync2', output="screen",
        parameters=[{
            # CHANGED: simulation has timestamp skew -> use approx sync
            "approx_sync": True,
            "use_sim_time": True,
            # CHANGED: reject crazy pairings
            "approx_sync_max_interval": 0.02
        }],
        remappings=[
            # CHANGED: camera2 -> PAN-TILT Gazebo topics
            ("rgb/image", '/pan_tilt_camera/pan_tilt_camera/image_raw'),
            ("depth/image", '/pan_tilt_camera/pan_tilt_camera/depth/image_raw'),
            ("rgb/camera_info", '/pan_tilt_camera/pan_tilt_camera/camera_info'),
            ("rgbd_image", 'rgbd_image')],
        # CHANGED: namespace name
        namespace='pan_tilt_camera'
    )

    # CHANGED (NEW NODE): Merge two RGBDImage topics into one RGBDImages topic
    rgbdx_sync_node = launch_ros.actions.Node(
        package='rtabmap_sync', executable='rgbdx_sync', name='rgbdx_sync', output="screen",
        parameters=[{
            # CHANGED: must match your sim reality
            "approx_sync": True,
            "use_sim_time": True,
            "approx_sync_max_interval": 0.02,
            # You can tune these later if drops happen
            "topic_queue_size": 10,
            "sync_queue_size": 30
        }],
        remappings=[
            # CHANGED: inputs are the rgbd_image topics produced by the two rgbd_sync nodes
            ("rgbd_image0", "/front_camera/rgbd_image"),
            ("rgbd_image1", "/pan_tilt_camera/rgbd_image"),
            # CHANGED: output topic name (kept in rtabmap namespace for convenience)
            ("rgbd_images", "/rtabmap/rgbd_images"),
        ],
        namespace=''
    )

    # RGB-D odometry (keep on FRONT only — original design still valid)
    rgbd_odometry_node = launch_ros.actions.Node(
        package='rtabmap_odom', executable='rgbd_odometry', output="screen",
        parameters=[{
            "use_sim_time": True,
            "frame_id": 'base_link',
            "odom_frame_id": 'odom',
            "publish_tf": True,
            "approx_sync": True,
            "subscribe_rgbd": True,
        }],
        remappings=[
            # CHANGED: uses front_camera rgbd_image
            ("rgbd_image", '/front_camera/rgbd_image'),
            ("odom", 'odom')],
        arguments=["--delete_db_on_start", ''],
        prefix='',
        namespace='rtabmap'
    )

    # SLAM
    slam_node = launch_ros.actions.Node(
        package='rtabmap_slam', executable='rtabmap', output="screen",
        parameters=[{
            "use_sim_time": True,
            # CHANGED: IMPORTANT
            # Do NOT use rgbd_cameras=2 here, we feed merged RGBDImages instead
            "rgbd_cameras": 2,

            # CHANGED: We are not subscribing to raw RGB/Depth; we subscribe to RGBDImages
            "subscribe_rgbd": False,
            "subscribe_rgb": False,
            "subscribe_depth": False,

            # CHANGED: enable RGBDImages interface
            "subscribe_rgbd_images": False,

            "subscribe_odom_info": True,
            "frame_id": 'base_link',
            "map_frame_id": 'map',
            "publish_tf": True,
            "database_path": '~/.ros/rtabmap.db',
            "approx_sync": True,
            "Mem/IncrementalMemory": "true",
            "Mem/InitWMWithAllNodes": "true"
        }],
        remappings=[
            # CHANGED: rtabmap reads merged RGBDImages
            ("rgbd_images", "/rtabmap/rgbd_images"),
            ("odom", 'odom')],
        arguments=["--delete_db_on_start"],
        prefix='',
        namespace='rtabmap'
    )

    voxelcloud1_node = launch_ros.actions.Node(
        package='rtabmap_util', executable='point_cloud_xyzrgb', name='point_cloud_xyzrgb1', output='screen',
        parameters=[{
            "use_sim_time": True,
            "approx_sync": True,
        }],
        remappings=[
            # CHANGED: front topics
            ('rgb/image', '/front_camera/front_camera/image_raw'),
            ('depth/image', '/front_camera/front_camera/depth/image_raw'),
            ('rgb/camera_info', '/front_camera/front_camera/camera_info'),
            ('rgbd_image', 'rgbd_image'),
            ('cloud', 'voxel_cloud1')]
    )

    voxelcloud2_node = launch_ros.actions.Node(
        package='rtabmap_util', executable='point_cloud_xyzrgb', name='point_cloud_xyzrgb2', output='screen',
        parameters=[{
            "use_sim_time": True,
            "approx_sync": True,
        }],
        remappings=[
            # CHANGED: pan-tilt topics
            ('rgb/image', '/pan_tilt_camera/pan_tilt_camera/image_raw'),
            ('depth/image', '/pan_tilt_camera/pan_tilt_camera/depth/image_raw'),
            ('rgb/camera_info', '/pan_tilt_camera/pan_tilt_camera/camera_info'),
            ('rgbd_image', 'rgbd_image'),
            ('cloud', 'voxel_cloud2')]
    )

    return launch.LaunchDescription(
        [
            rviz_node,
            rgbd_sync1_node,
            rgbd_sync2_node,
            rgbdx_sync_node,       # CHANGED: new
            rgbd_odometry_node,
            slam_node,
            voxelcloud1_node,
            voxelcloud2_node
        ]
    )
