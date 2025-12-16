from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package="steve_perception",
            executable="perception_node",
            name="steve_perception",
            output="screen",
            parameters=[{
                # just the file name; perception_node will resolve it
                "config_file": "steve.yaml",
                # IMPORTANT: no combined RGBD topics unless a mapping launch enables it.
                "publish_rgbd": False,
                # Empty list => start all cameras defined in steve.yaml.
                "enabled_cameras": [],
            }],
        ),
    ])
