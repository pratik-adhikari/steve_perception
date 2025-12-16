from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    # rgb_topic = "/wrist_camera/wrist_camera/image_raw"
    # depth_topic = "/wrist_camera/wrist_camera/depth/image_raw"
    # rgb_info_topic = "/wrist_camera/wrist_camera/camera_info"
    # depth_info_topic = "/wrist_camera/wrist_camera/depth/camera_info"
    odom_topic = "/odom"
    rgb_topic = "/front_camera/front_camera/image_raw"
    depth_topic = "/front_camera/front_camera/depth/image_raw"
    rgb_info_topic = "/front_camera/front_camera/camera_info"
    depth_info_topic = "/front_camera/front_camera/depth/camera_info"

    odom_topic = "/icp_odometry/odom"

    frame_id = "base_link"

    return LaunchDescription([
        # 1) Sync RGB + Depth (+ their camera infos) into one RGBD message
        Node(
            package="rtabmap_sync",
            executable="rgbd_sync",
            name="rgbd_sync",
            output="screen",
            parameters=[{
                "approx_sync": True,
                "sync_queue_size": 30,
            }],
            remappings=[
                ("rgb/image", rgb_topic),
                ("depth/image", depth_topic),
                ("rgb/camera_info", rgb_info_topic),
                ("depth/camera_info", depth_info_topic),
                ("rgbd_image", "rgbd_image"),
            ],
        ),

        # 2) RTAB-Map SLAM consuming RGBDImage + wheel odom
        Node(
            package="rtabmap_slam",
            executable="rtabmap",
            name="rtabmap",
            output="screen",
            parameters=[{
                "frame_id": frame_id,
                "map_frame_id": "map",
                "odom_frame_id": "odom",
                "publish_tf": True,

                "subscribe_rgbd": True,
                "subscribe_odom": True,

                "approx_sync": True,
                "Rtabmap/DetectionRate": "2.0",
                "Mem/IncrementalMemory": "true",
                "Mem/InitWMWithAllNodes": "false",

                # Make it less eager to create nodes if you have low texture
                "RGBD/CreateOccupancyGrid": "false",
            }],
            remappings=[
                ("rgbd_image", "rgbd_image"),
                ("odom", odom_topic),
            ],
        ),
        
        Node(
            package="rtabmap_odom",
            executable="icp_odometry",
            name="icp_odometry",
            output="screen",
            parameters=[{
                "frame_id": "base_link",
                "odom_frame_id": "odom",
                "publish_tf": True,

                "subscribe_scan": True,
                "subscribe_scan_cloud": False,
                "scan_cloud_is_2d": True,

                # ICP tuning (start conservative)
                "Icp/MaxCorrespondenceDistance": "0.5",
                "Icp/VoxelSize": "0.05",
                "Icp/Iterations": "10",
                "Icp/PointToPlane": "false",

                # If your robot is planar, enforce 2D motion
                "Reg/Force3DoF": "true",
            }],
            remappings=[
                ("scan", "/lidar_1/scan"),
            ],
        ),


        # 3) Visualization
        Node(
            package="rtabmap_viz",
            executable="rtabmap_viz",
            name="rtabmap_viz",
            output="screen",
            parameters=[{
                "frame_id": frame_id,
                "map_frame_id": "map",
                "odom_frame_id": "odom",
                "subscribe_rgbd": True,
                "subscribe_odom_info": True,
                "approx_sync": True,
            }],
            remappings=[
                ("rgbd_image", "rgbd_image"),
                ("odom", odom_topic),
            ],
        ),
    ])
