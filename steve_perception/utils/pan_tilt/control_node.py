#!/usr/bin/env python3
"""ROS 2 Node for controlling the pan-tilt camera."""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from control_msgs.action import FollowJointTrajectory
import argparse
import sys
import math

from steve_perception.utils.pan_tilt.trajectory import create_single_goal_trajectory, create_elliptical_trajectory

class PanTiltController(Node):
    def __init__(self):
        super().__init__('pan_tilt_controller_node')
        
        # Parameters
        defaults = {
            'pan_min': -180.0, 'pan_max': 20.0,
            'tilt_min': -20.0, 'tilt_max': 20.0,
            'initial_pan': 0.0, 'initial_tilt': 0.0,
            'pan_goals': [0.0], 'tilt_goals': [0.0],
            'speed': 10.0, 'log_feedback': False
        }
        for name, val in defaults.items():
            self.declare_parameter(name, val)
            
        try:
            # Load and converting params
            p = {k: self.get_parameter(k).value for k in defaults}
            self.lims = {
                'pan': (math.radians(float(p['pan_min'])), math.radians(float(p['pan_max']))),
                'tilt': (math.radians(float(p['tilt_min'])), math.radians(float(p['tilt_max'])))
            }
            self.speed_rad_s = math.radians(float(p['speed']))
            self.log_fb = p['log_feedback']

            # Parse Goals
            p_goals = self._ensure_list(p['pan_goals'])
            t_goals = self._ensure_list(p['tilt_goals'])
            
            self.pan_c, self.pan_a = self._parse_mode(p_goals, "Pan")
            self.tilt_c, self.tilt_a = self._parse_mode(t_goals, "Tilt")
            self.sweep = (self.pan_a > 1e-6 or self.tilt_a > 1e-6)

            # Clamp targets (if fixed) and init
            if not self.sweep:
                self.pan_c = self._clamp(self.pan_c, 'pan')
                self.tilt_c = self._clamp(self.tilt_c, 'tilt')
                
            init_p = self._clamp(math.radians(float(p['initial_pan'])), 'pan')
            init_t = self._clamp(math.radians(float(p['initial_tilt'])), 'tilt')
            
        except Exception as e:
            self.get_logger().error(f"Param Error: {e}")
            raise e

        self.cli = ActionClient(self, FollowJointTrajectory, '/pan_tilt_controller/follow_joint_trajectory')
        # Correct mapping based on URDF analysis:
        # Base Motor (Pan motion) is named '..._tilt_motor_joint'
        # Top Motor (Tilt motion) is named '..._pan_motor_joint'
        self.joint_names = ['pan_tilt_tilt_motor_joint', 'pan_tilt_pan_motor_joint']
        
        self.get_logger().info('Waiting for server...')
        self.cli.wait_for_server()
        
        self.last_log = self.get_clock().now()

        # Start
        self.get_logger().info(f'Moving to Init: P={math.degrees(init_p):.2f}, T={math.degrees(init_t):.2f}')
        self._send_traj(
            create_single_goal_trajectory(init_p, init_t, self.speed_rad_s)[0], 
            on_success=self.start_main
        )

    def start_main(self):
        self.get_logger().info('Init done. Starting Main.')
        if self.sweep:
            self._loop_sweep()
        else:
            self.get_logger().info(f'Holding Target: P={math.degrees(self.pan_c):.2f}, T={math.degrees(self.tilt_c):.2f}')
            self._send_traj(
                create_single_goal_trajectory(self.pan_c, self.tilt_c, self.speed_rad_s)[0],
                on_success=lambda: self.get_logger().info('Target Reached.')
            )

    def _loop_sweep(self):
        pts, dur = create_elliptical_trajectory(
            self.pan_c, self.pan_a, self.tilt_c, self.tilt_a, self.speed_rad_s
        )
        self.get_logger().info(f'Sweeping (T={dur:.1f}s)...')
        self._send_traj(pts, on_success=self._loop_sweep)

    def _send_traj(self, points, on_success):
        msg = FollowJointTrajectory.Goal()
        msg.trajectory.joint_names = self.joint_names
        msg.trajectory.points = points
        
        future = self.cli.send_goal_async(msg, feedback_callback=self._fb_cb)
        future.add_done_callback(lambda f: self._on_goal(f, on_success))

    def _on_goal(self, future, on_success):
        gh = future.result()
        if not gh.accepted:
            self.get_logger().warn('Goal Rejected')
            return
        gh.get_result_async().add_done_callback(lambda f: self._on_res(f, on_success))

    def _on_res(self, future, on_success):
        res = future.result().result
        if res.error_code == 0:
            on_success()
        else:
            self.get_logger().error(f'Goal Failed: {res.error_code}')

    def _fb_cb(self, msg):
        if not self.log_fb: return
        
        now = self.get_clock().now()
        if (now - self.last_log).nanoseconds < 1e9:
            return
        self.last_log = now

        act = msg.feedback.actual.positions
        des = msg.feedback.desired.positions
        
        # Joint 0 = Pan, Joint 1 = Tilt
        pc, tc = math.degrees(act[0]), math.degrees(act[1])
        pt, tt = (math.degrees(des[0]), math.degrees(des[1])) if des else (0.0, 0.0)
        
        self.get_logger().info(f'Status: Pan[C={pc:.1f}, T={pt:.1f}] Tilt[C={tc:.1f}, T={tt:.1f}]')

    def _clamp(self, val, axis):
        mn, mx = self.lims[axis]
        return max(mn, min(val, mx))

    def _ensure_list(self, val):
        if isinstance(val, str):
            try: return [float(x) for x in val.replace('[','').replace(']','').split(',') if x.strip()]
            except: return [0.0]
        return [float(x) for x in (val if isinstance(val, (list, tuple)) else [val])]

    def _parse_mode(self, goals, name):
        if not goals: return 0.0, 0.0
        if len(goals) == 1:
            self.get_logger().info(f"{name}: Fixed {goals[0]}")
            return math.radians(goals[0]), 0.0
        c = (goals[0] + goals[1]) / 2.0
        a = abs(goals[0] - goals[1]) / 2.0
        self.get_logger().info(f"{name}: Sweep {goals} (C={c:.1f}, A={a:.1f})")
        return math.radians(c), math.radians(a)


def main(args=None):
    rclpy.init(args=args)

    node = PanTiltController()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        node.get_logger().error(f"Exception: {e}")
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()

if __name__ == '__main__':
    main()
