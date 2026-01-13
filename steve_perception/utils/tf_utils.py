# steve_perception/utils/tf_utils.py
"""Utility functions for TF transformations."""
import numpy as np
from geometry_msgs.msg import Transform

# Patch numpy.float for transforms3d compatibility with numpy >= 1.20
if not hasattr(np, 'float'):
    np.float = float

from tf_transformations import quaternion_matrix

def transform_to_matrix(t: Transform) -> np.ndarray:
    """Convert geometry_msgs/Transform to 4x4 homogeneous matrix."""
    q = [t.rotation.x, t.rotation.y, t.rotation.z, t.rotation.w]
    T = quaternion_matrix(q)  # rotation matrix from quaternion
    T[0, 3] = t.translation.x
    T[1, 3] = t.translation.y
    T[2, 3] = t.translation.z
    return T
