#!/usr/bin/env python3

# DIFFERENCE from tomato_perception.utils.ros_conversions:
# - tomato_perception has a huge file converting Open3D clouds, tomato-specific messages (TomatoTruss, etc.).
# - Here we implement ONLY a minimal helper to convert an Nx3 numpy array into a standard ROS2 PointCloud2.
# - No Open3D, no tomato-specific fields, no RGB/normals for now (MVP for WP1 mapping).

import numpy as np
from sensor_msgs.msg import PointCloud2, PointField
from builtin_interfaces.msg import Time as TimeMsg


def points_to_cloud2(
    points: np.ndarray,
    frame_id: str,
    stamp: TimeMsg,
) -> PointCloud2:
    """
    Pack Nx3 float32 points into a PointCloud2 message in ROS2.

    Parameters
    ----------
    points : Nx3 float32
        3D points in some frame (typically base_link).
    frame_id : str
        TF frame of the points.
    stamp : builtin_interfaces/Time
        Timestamp for the header (usually now or RGBD frame time).

    NOTE:
    - We only support XYZ with float32.
    - No colors, normals, or intensity in this minimal version.
    - This is enough to feed into mapping / visualization later.
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

    # 3 fields: x, y, z as float32
    cloud.fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
    ]

    cloud.point_step = 12  # 3 * 4 bytes
    cloud.row_step = cloud.point_step * cloud.width

    cloud.data = points.astype(np.float32).tobytes()

    return cloud
