"""RViz for the hand-held grasp test (steve_perception/grasp_test, `handheld_grasp_test --live`):
robot_state_publisher with the Steve URDF (same xacro + arguments as steve_hardware_bringup) and
RViz with the grasp-test layout (fixed frame odom). The test script publishes /joint_states and
TF odom -> base_link itself, so do not run this next to the real robot's bringup.

    ros2 launch steve_perception grasp_test_rviz.launch.py
    ros2 run steve_perception handheld_grasp_test --live          # second terminal
"""

import os

import xacro
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _nodes(context):
    urdf = os.path.join(get_package_share_directory("steve_simulation"), "robots", "mmo_700", "mmo_700.urdf.xacro")
    description = xacro.process_file(urdf, mappings={
        "use_gazebo": "false",
        "arm_type": "ur5e",
        "include_wrist_camera": "true",
        "include_depth_camera": "false",
        "include_pan_tilt": "true",
        "arm_tool": "none",
        "gripper_hw": "none",
    }).toxml()
    nodes = [Node(package="robot_state_publisher", executable="robot_state_publisher",
                  name="grasp_test_robot_state_publisher", output="screen",
                  parameters=[{"robot_description": description}])]
    if LaunchConfiguration("rviz").perform(context).lower() == "true":
        cfg = os.path.join(get_package_share_directory("steve_perception"), "rviz", "grasp_test.rviz")
        nodes.append(Node(package="rviz2", executable="rviz2", name="grasp_test_rviz", output="screen",
                          arguments=["-d", cfg]))
    return nodes


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("rviz", default_value="true", description="start RViz"),
        OpaqueFunction(function=_nodes),
    ])
