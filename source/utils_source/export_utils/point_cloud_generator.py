#!/usr/bin/env python3
"""
Point cloud generation from mesh for RGBD systems.
"""

import numpy as np
import open3d as o3d
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def sample_from_mesh(mesh_path: Path, num_samples: int = 1_000_000) -> o3d.geometry.PointCloud:
    """Sample dense points from mesh surface."""
    mesh = o3d.io.read_triangle_mesh(str(mesh_path), enable_post_processing=True)
    
    # Clean mesh
    v_before = len(mesh.vertices)
    mesh = mesh.remove_duplicated_vertices()
    mesh = mesh.remove_unreferenced_vertices()
    logger.info(f"Cleaned mesh: {v_before} → {len(mesh.vertices)} vertices")
    
    if not mesh.has_triangles():
        raise RuntimeError("Mesh has no faces")
    
    pcd = mesh.sample_points_uniformly(number_of_points=num_samples)
    logger.info(f"Sampled {len(pcd.points):,} points")
    return pcd


def downsample(pcd: o3d.geometry.PointCloud, voxel_size: float) -> o3d.geometry.PointCloud:
    """Downsample point cloud with voxel grid."""
    n_before = len(pcd.points)
    pcd_ds = pcd.voxel_down_sample(voxel_size=voxel_size)
    pcd_ds = pcd_ds.remove_non_finite_points()
    logger.info(f"Downsampled: {n_before:,} → {len(pcd_ds.points):,} points")
    return pcd_ds


def add_colors_if_missing(pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    """Add uniform color if point cloud has none."""
    if not pcd.has_colors():
        logger.warning("Point cloud has no colors, adding gray")
        pcd.paint_uniform_color([0.5, 0.5, 0.5])
    return pcd


def estimate_normals(pcd: o3d.geometry.PointCloud, radius: float = 0.1) -> o3d.geometry.PointCloud:
    """Estimate surface normals."""
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30))
    logger.info(f"Estimated normals for {len(pcd.points):,} points")
    return pcd


def save_pcd(pcd: o3d.geometry.PointCloud, output_path: Path) -> None:
    """Save point cloud as PCD."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(output_path), pcd)
    logger.info(f"Saved: {output_path.name} ({len(pcd.points):,} points)")


def load_pcd(pcd_path: Path) -> o3d.geometry.PointCloud:
    """Load point cloud from PCD."""
    pcd = o3d.io.read_point_cloud(str(pcd_path))
    logger.info(f"Loaded: {pcd_path.name} ({len(pcd.points):,} points)")
    return pcd


def create_point_cloud_pipeline(mesh_path: Path, output_path: Path, 
                                num_samples: int = 1_000_000, 
                                voxel_size: float = 0.01) -> bool:
    """Generate RGBD point cloud from mesh."""
    try:
        logger.info("Creating point cloud from mesh...")
        
        pcd = sample_from_mesh(mesh_path, num_samples)
        pcd = downsample(pcd, voxel_size)
        pcd = add_colors_if_missing(pcd)
        pcd = estimate_normals(pcd)
        save_pcd(pcd, output_path)
        
        logger.info("Point cloud complete")
        return True
    except Exception as e:
        logger.error(f"Point cloud failed: {e}")
        return False
