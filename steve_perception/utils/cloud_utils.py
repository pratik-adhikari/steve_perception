#!/usr/bin/env python3
"""Utility functions for depth and point cloud processing."""

import numpy as np
from sensor_msgs.msg import PointCloud2, PointField
from builtin_interfaces.msg import Time as TimeMsg

def points_to_cloud2(points: np.ndarray, frame_id: str, stamp: TimeMsg) -> PointCloud2:
    """Pack Nx3 float32 points into PointCloud2 message.
    
    Args:
        points: Nx3 float32 array of 3D points
        frame_id: TF frame for the points
        stamp: ROS time for message header
    """
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must be an Nx3 array")

    cloud = PointCloud2()
    cloud.header.frame_id = frame_id
    cloud.header.stamp = stamp
    cloud.height = 1
    cloud.width = points.shape[0]
    cloud.is_bigendian = False
    cloud.is_dense = True

    # Define XYZ fields (float32)
    cloud.fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
    ]

    cloud.point_step = 12
    cloud.row_step = cloud.point_step * cloud.width
    cloud.data = points.astype(np.float32).tobytes()

    return cloud
