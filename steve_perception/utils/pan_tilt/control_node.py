#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from control_msgs.action import FollowJointTrajectory
import argparse
import sys
import math

from steve_perception.utils.pan_tilt.trajectory import create_single_goal_trajectory, create_elliptical_trajectory

class PanTiltController(Node):
    def __init__(self, pan_deg, tilt_deg, speed_deg_s, sweep=False):
        super().__init__('pan_tilt_controller_node')
        
        # Declare ROS parameters (without descriptors to allow loose typing from launch)
        self.declare_parameter('pan', float(pan_deg))
        self.declare_parameter('tilt', float(tilt_deg))
        self.declare_parameter('speed', float(speed_deg_s))
        self.declare_parameter('sweep', sweep)
        self.declare_parameter('log_feedback', False)
        
        # Read parameters
        # We explicitly cast to float to handle cases where '4' (int) is passed but we want float
        try:
            p_val = self.get_parameter('pan').value
            t_val = self.get_parameter('tilt').value
            s_val = self.get_parameter('speed').value
            
            self.pan_rad = math.radians(float(p_val))
            self.tilt_rad = math.radians(float(t_val))
            self.speed_rad_s = math.radians(float(s_val))
        except (TypeError, ValueError) as e:
            self.get_logger().error(f"Failed to parse parameters: {e}")
            raise e
            
        self.sweep_mode = bool(self.get_parameter('sweep').value)
        
        self._action_client = ActionClient(
            self, 
            FollowJointTrajectory, 
            '/pan_tilt_controller/follow_joint_trajectory'
        )
        
        # Mapping: TILT -> joint 0, PAN -> joint 1
        self.joint_names = ['pan_tilt_pan_motor_joint', 'pan_tilt_tilt_motor_joint']
        
        self.get_logger().info('Waiting for action server...')
        self._action_client.wait_for_server()
        self.get_logger().info('Action server available.')

        if self.sweep_mode:
            self.get_logger().info(f'Moving to start position of ellipse: Pan={math.degrees(self.pan_rad):.2f}, Tilt=0.0')
            self.send_single_goal(self.pan_rad, 0.0, is_approach=True)
        else:
            self.send_single_goal(self.pan_rad, self.tilt_rad)

    def send_single_goal(self, target_pan, target_tilt, is_approach=False):
        goal_msg = FollowJointTrajectory.Goal()
        goal_msg.trajectory.joint_names = self.joint_names
        
        points, duration = create_single_goal_trajectory(target_pan, target_tilt, self.speed_rad_s)
        goal_msg.trajectory.points = points
        
        self.get_logger().info(f'Sending goal: Pan={math.degrees(target_pan):.2f} deg, Tilt={math.degrees(target_tilt):.2f} deg, Duration={duration:.2f} s')
        
        self._send_goal_future = self._action_client.send_goal_async(
            goal_msg, 
            feedback_callback=self.feedback_callback
        )
        
        if is_approach:
            self._send_goal_future.add_done_callback(self.approach_response_callback)
        else:
            self._send_goal_future.add_done_callback(self.goal_response_callback)

    def send_ellipse_trajectory(self):
        goal_msg = FollowJointTrajectory.Goal()
        goal_msg.trajectory.joint_names = self.joint_names
        
        points, period = create_elliptical_trajectory(self.pan_rad, self.tilt_rad, self.speed_rad_s)
        goal_msg.trajectory.points = points
        
        self.get_logger().info(f'Sending elliptical trajectory: Period={period:.2f}s, Points={len(points)}')
        
        self._send_goal_future = self._action_client.send_goal_async(
            goal_msg, 
            feedback_callback=self.feedback_callback
        )
        self._send_goal_future.add_done_callback(self.ellipse_response_callback)

    def approach_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Approach goal rejected.')
            return
        
        self.get_logger().info('Approach goal accepted. Moving to start...')
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.approach_result_callback)

    def approach_result_callback(self, future):
        result = future.result().result
        if result.error_code == 0:
            self.get_logger().info('Reached start position. Starting Ellipse Loop.')
            self.send_ellipse_trajectory()
        else:
            self.get_logger().info(f'Approach failed with error code: {result.error_code}')
            rclpy.shutdown()

    def ellipse_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Ellipse goal rejected.')
            rclpy.shutdown()
            return

        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.ellipse_result_callback)

    def ellipse_result_callback(self, future):
        result = future.result().result
        if result.error_code == 0:
            self.get_logger().info('Ellipse cycle complete. Repeating...')
            self.send_ellipse_trajectory()
        else:
            self.get_logger().info(f'Ellipse failed: {result.error_code}')
            rclpy.shutdown()

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected :(')
            return
        self.get_logger().info('Goal accepted :)')
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        result = future.result().result
        self.get_logger().info(f'Goal finished with error_code: {result.error_code}')
        rclpy.shutdown()

    def feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback
        current_tilt_rad = feedback.actual.positions[0]
        current_pan_rad = feedback.actual.positions[1]
        
        if self.get_parameter('log_feedback').value:
            self.get_logger().info(
                f'Feedback: Pan={math.degrees(current_pan_rad):.2f} deg, Tilt={math.degrees(current_tilt_rad):.2f} deg'
            )


def main(args=None):
    parser = argparse.ArgumentParser(description='Control Pan-Tilt Camera with Feedback')
    parser.add_argument('--pan', type=float, default=0.0, help='Pan angle/amplitude (deg)')
    parser.add_argument('--tilt', type=float, default=0.0, help='Tilt angle/amplitude (deg)')
    parser.add_argument('--speed', type=float, default=10.0, help='Speed (deg/s)')
    parser.add_argument('--sweep', action='store_true', help='Enable elliptical sweep mode')
    parser.add_argument('--log', action='store_true', help='Enable feedback logging')
    
    ros_args = None
    if args is None:
        args = sys.argv[1:]
        
    parsed_args, unknown = parser.parse_known_args(args)

    rclpy.init(args=unknown)

    # Initialize with CLI args, which become default values for ROS params
    node = PanTiltController(
        parsed_args.pan, 
        parsed_args.tilt, 
        parsed_args.speed,
        sweep=parsed_args.sweep
    )
    # Update parameter manually since we didn't pass it to init
    node.set_parameters([rclpy.parameter.Parameter('log_feedback', value=parsed_args.log)])

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(e)
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
