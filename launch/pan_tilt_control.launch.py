"""Launches the Pan-Tilt camera control node."""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            "pan_goals",
            default_value="[4.0]",
            description="Pan goals: '[val]' for fixed, '[min, max]' for sweep (deg).",
        ),
        DeclareLaunchArgument(
            "pan_min",
            default_value="-280.0",
            description="Minimum pan angle (deg).",
        ),
        DeclareLaunchArgument(
            "pan_max",
            default_value="280.0",
            description="Maximum pan angle (deg).",
        ),
        DeclareLaunchArgument(
            "tilt_goals",
            default_value="[20.0]",
            description="Tilt goals: '[val]' for fixed, '[min, max]' for sweep (deg).",
        ),
        DeclareLaunchArgument(
            "tilt_min",
            default_value="-20.0",
            description="Minimum tilt angle (deg).",
        ),
        DeclareLaunchArgument(
            "tilt_max",
            default_value="20.0",
            description="Maximum tilt angle (deg).",
        ),
        DeclareLaunchArgument(
            "initial_pan",
            default_value="0.0",
            description="Initial pan angle to move to on startup (deg).",
        ),
        DeclareLaunchArgument(
            "initial_tilt",
            default_value="0.0",
            description="Initial tilt angle to move to on startup (deg).",
        ),
        DeclareLaunchArgument(
            "pan_tilt_speed",
            default_value="25.0",
            description="Movement speed in deg/s.",
        ),
        DeclareLaunchArgument(
            "log_pan_tilt_feedback",
            default_value="true",
            description="Enable or disable feedback logging from pan-tilt controller.",
        ),
        OpaqueFunction(function=launch_setup),
    ])

def launch_setup(context, *args, **kwargs):
    return [
        Node(
            package="steve_perception",
            executable="pan_tilt_control",
            name="pan_tilt_control",
            output="screen",
            parameters=[{
                "pan_goals": LaunchConfiguration("pan_goals"),
                "tilt_goals": LaunchConfiguration("tilt_goals"),
                "pan_min": LaunchConfiguration("pan_min"),
                "pan_max": LaunchConfiguration("pan_max"),
                "tilt_min": LaunchConfiguration("tilt_min"),
                "tilt_max": LaunchConfiguration("tilt_max"),
                "initial_pan": LaunchConfiguration("initial_pan"),
                "initial_tilt": LaunchConfiguration("initial_tilt"),
                "speed": LaunchConfiguration("pan_tilt_speed"),
                "log_feedback": LaunchConfiguration("log_pan_tilt_feedback"),
            }]
        )
    ]
