#!/usr/bin/env python3
"""
Camera calibration and pose utilities.
"""

import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation as R
import yaml
import logging

logger = logging.getLogger(__name__)


def load_poses_from_file(poses_file: Path) -> dict:
    """Load camera poses from RTAB-Map poses.txt file."""
    poses = {}
    
    with open(poses_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 8:
                continue
            
            try:
                stamp = parts[0]
                tx, ty, tz = float(parts[1]), float(parts[2]), float(parts[3])
                qx, qy, qz, qw = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])
                
                rot = R.from_quat([qx, qy, qz, qw]).as_matrix()
                T = np.eye(4)
                T[:3, :3] = rot
                T[:3, 3] = [tx, ty, tz]
                poses[stamp] = T
            except (ValueError, IndexError):
                continue
    
    logger.info(f"Loaded {len(poses)} poses")
    return poses


def load_calibration(calib_file: Path) -> tuple:
    """Load camera calibration from YAML file. Returns (K_4x4, T_local_4x4)."""
    with open(calib_file, 'r') as f:
        content = f.read()
        if content.startswith("%YAML"):
            content = content.split('\n', 1)[1]
    
    data = yaml.safe_load(content)
    
    # Camera matrix
    K = np.array(data['camera_matrix']['data']).reshape(3, 3)
    K_4x4 = np.eye(4)
    K_4x4[:3, :3] = K
    
    # Local transform (base to camera)
    T_local = np.eye(4)
    if 'local_transform' in data:
        T_local_3x4 = np.array(data['local_transform']['data']).reshape(3, 4)
        T_local[:3, :] = T_local_3x4
    
    return K_4x4, T_local


def save_matrix(matrix: np.ndarray, output_path: Path) -> None:
    """Save matrix to text file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(output_path, matrix, fmt='%.6f')
