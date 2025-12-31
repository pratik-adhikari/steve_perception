import math
from trajectory_msgs.msg import JointTrajectoryPoint
from builtin_interfaces.msg import Duration

def create_single_goal_trajectory(target_pan_rad, target_tilt_rad, speed_rad_s):
    """
    Creates a single point trajectory.
    Returns: (points_list, duration_in_seconds)
    """
    point = JointTrajectoryPoint()
    # Mapping: [Tilt, Pan]
    point.positions = [target_tilt_rad, target_pan_rad]
    point.velocities = [0.0, 0.0]
    
    # Duration logic
    dist = max(abs(target_pan_rad), abs(target_tilt_rad))
    if speed_rad_s <= 1e-6:
         duration = 2.0
    else:
         duration = dist / speed_rad_s
    duration = max(duration, 1.0) 

    sec = int(duration)
    nanosec = int((duration - sec) * 1e9)
    point.time_from_start = Duration(sec=sec, nanosec=nanosec)
    
    return [point], duration

def create_elliptical_trajectory(pan_amplitude_rad, tilt_amplitude_rad, speed_rad_s):
    """
    Creates an elliptical trajectory.
    Returns: (points_list, period_in_seconds)
    """
    A = pan_amplitude_rad
    B = tilt_amplitude_rad
    
    max_amplitude = max(abs(A), abs(B))
    if max_amplitude < 1e-6:
        omega = 1.0 
    else:
        omega = speed_rad_s / max_amplitude
        
    period = 2 * math.pi / omega
    
    # Generate points
    # Resolution: 10 points per second? Or fixed number?
    # Let's say 20 points per cycle or roughly 10Hz
    num_points = max(int(period * 10), 20) 
    time_step = period / num_points
    
    points = []
    for i in range(num_points + 1): # +1 to close the loop perfectly
        t = i * time_step
        # Parametric equations
        # Pan = A * cos(omega * t)
        # Tilt = B * sin(omega * t)
        
        p_val = A * math.cos(omega * t)
        t_val = B * math.sin(omega * t)
        
        # Velocities (derivatives)
        v_p = -A * omega * math.sin(omega * t)
        v_t = B * omega * math.cos(omega * t)
        
        point = JointTrajectoryPoint()
        # Mapping [Tilt, Pan]
        point.positions = [t_val, p_val]
        point.velocities = [v_t, v_p]
        
        # Time from start
        t_sec = int(t)
        t_nano = int((t - t_sec) * 1e9)
        point.time_from_start = Duration(sec=t_sec, nanosec=t_nano)
        
        points.append(point)
        
    return points, period
