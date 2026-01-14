#!/usr/bin/env python3
"""
Mesh processing - load, clean, save to PLY.
"""

import shutil
import open3d as o3d
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def load_mesh(mesh_path: Path) -> o3d.geometry.TriangleMesh:
    """Load mesh from file (OBJ, PLY, STL, etc)."""
    mesh = o3d.io.read_triangle_mesh(str(mesh_path), enable_post_processing=True)
    logger.info(f"Loaded: {mesh_path.name} - {len(mesh.vertices):,} verts, {len(mesh.triangles):,} tris")
    return mesh


def clean_mesh(mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
    """Remove duplicate and unreferenced vertices."""
    v_before = len(mesh.vertices)
    mesh = mesh.remove_duplicated_vertices()
    mesh = mesh.remove_unreferenced_vertices()
    v_after = len(mesh.vertices)
    
    if v_before > v_after:
        logger.info(f"Cleaned: {v_before:,} → {v_after:,} vertices")
    
    return mesh


def save_mesh_ply(mesh: o3d.geometry.TriangleMesh, output_path: Path) -> None:
    """Save mesh to PLY format."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_triangle_mesh(str(output_path), mesh)
    logger.info(f"Saved: {output_path.name} ({len(mesh.vertices):,} verts)")


def save_mesh_obj(mesh: o3d.geometry.TriangleMesh, output_path: Path) -> None:
    """Save mesh to OBJ format."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_triangle_mesh(str(output_path), mesh)
    logger.info(f"Saved: {output_path.name}")


def copy_textures(src_dir: Path, dst_dir: Path) -> int:
    """Copy texture files (jpg, png, mtl)."""
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    
    patterns = ['*.jpg', '*.jpeg', '*.png', '*.mtl']
    count = 0
    
    for pattern in patterns:
        for src_file in Path(src_dir).glob(pattern):
            dst_file = dst_dir / src_file.name
            shutil.copy(src_file, dst_file)
            count += 1
    
    if count > 0:
        logger.info(f"Copied {count} texture files")
    
    return count
