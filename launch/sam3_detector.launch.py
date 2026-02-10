#!/usr/bin/env python3
"""
Launch file for SAM3 detector node

Usage:
    # Head camera with bottle detection
    ros2 launch steve_perception sam3_detector.launch.py camera:=head query:=bottle
    
    # Wrist camera with cup detection  
    ros2 launch steve_perception sam3_detector.launch.py camera:=wrist query:=cup
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Declare arguments
    camera_arg = DeclareLaunchArgument(
        'camera',
        default_value='head',
        description='Camera to use: head or wrist'
    )
    
    query_arg = DeclareLaunchArgument(
        'query',
        default_value='bottle',
        description='Object to detect (e.g., bottle, cup, can, milk)'
    )
    
    publish_rate_arg = DeclareLaunchArgument(
        'publish_rate',
        default_value='1.0',
        description='Detection rate in Hz'
    )
    
    # Get launch configurations
    camera = LaunchConfiguration('camera')
    query = LaunchConfiguration('query')
    publish_rate = LaunchConfiguration('publish_rate')
    
    # Map camera name to topic
    camera_topics = {
        'head': '/head_camera/color/image_raw',
        'wrist': '/wrist_camera/color/image_raw',
        'pantilt': '/pan_tilt_camera/color/image_raw',
    }
    
    # Create node
    sam3_detector_node = Node(
        package='steve_perception',
        executable='sam3_detector',
        name='sam3_detector',
        output='screen',
        parameters=[{
            'camera_topic': ['/', camera, '_camera/', camera, '_camera/image_raw'],  # /pan_tilt_camera/pan_tilt_camera/image_raw
            'query': query,
            'publish_rate': publish_rate,
            'sam3_url': 'http://localhost:5005',
            'visualize': True,
        }],
    )
    
    return LaunchDescription([
        camera_arg,
        query_arg,
        publish_rate_arg,
        sam3_detector_node,
    ])
