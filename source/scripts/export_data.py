#!/usr/bin/env python3
"""Export data from RTAB-Map database."""

import argparse
import shutil
import sys
import subprocess
import numpy as np
import yaml
import datetime
import logging
import open3d as o3d # Required for mesh processing
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils_source.export_utils import point_cloud_generator, mesh_processor, calibration_utils, mesh_reconstruction


def setup_logger(output_dir):
    """Setup logging to file and stdout."""
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = log_dir / f"01_export_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s: %(message)s',
        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)]
    )
    return logging.getLogger(__name__)


def validate_inputs(input_db):
    """Validate input database and required tools."""
    if shutil.which("rtabmap-export") is None:
        raise RuntimeError("rtabmap-export not found! Install RTAB-Map")
    if not input_db.exists():
        print(f"Error: DB {input_db} not found.")
        sys.exit(1)
    if not input_db.is_file():
        print(f"Error: {input_db} is not a file.")
        sys.exit(1)


def get_relative_path(path, cwd):
    """Get relative path or fallback to absolute."""
    try:
        return path.relative_to(cwd)
    except ValueError:
        return path


def setup_directories(output_dir):
    """Create directory structure and return directory paths."""
    export_dir = output_dir / "export"
    metadata_dir = export_dir / "metadata"
    
    # Flat structure: color, depth, poses, textures, metadata
    # scan_dir/color/0.jpg
    # scan_dir/depth/0.png
    # scan_dir/poses/0.txt
    for d in [export_dir / "color", export_dir / "depth", export_dir / "poses", export_dir / "textures", metadata_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    raw_dir = output_dir / "raw_export"
    raw_dir.mkdir(exist_ok=True)
    
    return {'export': export_dir, 'metadata': metadata_dir, 'raw': raw_dir}


def run_cmd(logger, cmd, description):
    """Helper to run and log subprocess commands."""
    logger.info(description)
    try:
        result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.stderr:
            logger.debug(result.stderr.strip())
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed: {e}")
        raise


def export_rtabmap_data(logger, raw_dir, input_db):
    """Export images, poses, and mesh from RTAB-Map database."""
    run_cmd(logger, ["rtabmap-export", "--output_dir", str(raw_dir), "--images", "--poses", "--poses_format", "10", str(input_db)], 
            "Extracting images and poses...")
            
    # Pass 1: Textured Mesh (for visualization)
    # This creates mesh.obj + textures in raw_dir
    run_cmd(logger, ["rtabmap-export", "--output_dir", str(raw_dir), "--texture", str(input_db)], 
            "Extracting textured mesh for visualization...")

    # Pass 2: Vertex-Colored Mesh/Cloud (for processing)
    # We export to a subdir to avoid overwriting the textured mesh
    # Without --texture flag, rtabmap-export usually produces a PLY with vertex colors
    proc_dir = raw_dir / "processing"
    proc_dir.mkdir(exist_ok=True)
    run_cmd(logger, ["rtabmap-export", "--output_dir", str(proc_dir), str(input_db)], 
            "Extracting colored geometry for processing...")



def organize_frames(logger, raw_dir, export_dir):
    """Organize and process frames from raw export."""
    poses_file = raw_dir / "rtabmap_poses.txt"
    if not poses_file.exists():
        raise RuntimeError("rtabmap_poses.txt not found")
    poses = calibration_utils.load_poses_from_file(poses_file)
    
    rgb_src = raw_dir / "rtabmap_rgb" if (raw_dir / "rtabmap_rgb").exists() else raw_dir
    depth_src = raw_dir / "rtabmap_depth" if (raw_dir / "rtabmap_depth").exists() else raw_dir
    calib_src = raw_dir / "rtabmap_calib"
    
    images = sorted(list(rgb_src.glob("*.jpg")))
    logger.info(f"Found {len(images)} frames")
    
    count, first_intrinsic = 0, None
    stats = {'pose_skip': 0, 'missing_depth': 0, 'missing_calib': 0, 'intrinsic_mismatch': 0}
    
    for img_path in images:
        stamp = img_path.stem
        if stamp not in poses:
            stats['pose_skip'] += 1
            logger.warning(f"Skipping frame {stamp}: no pose")
            continue
        
        # ScanNet format: integer based, no padding (e.g. 0.jpg, 0.png, 0.txt)
        frame_idx = f"{count}"
        shutil.copy(img_path, export_dir / "color" / f"{frame_idx}.jpg")
        
        depth_cand = depth_src / f"{stamp}.png"
        if depth_cand.exists():
            shutil.copy(depth_cand, export_dir / "depth" / f"{frame_idx}.png")
        else:
            stats['missing_depth'] += 1
            
        calib_file = calib_src / f"{stamp}.yaml"
        K_4x4, T_local = np.eye(4), np.eye(4)
        
        if calib_file.exists():
            K_4x4, T_local = calibration_utils.load_calibration(calib_file)
            if first_intrinsic is None:
                first_intrinsic = K_4x4
            elif not np.allclose(K_4x4, first_intrinsic, atol=1e-6):
                stats['intrinsic_mismatch'] += 1
        else:
            stats['missing_calib'] += 1
            if first_intrinsic is not None:
                K_4x4 = first_intrinsic
        
        T_world_cam = poses[stamp] @ T_local
        # Save pose: poses/0.txt
        calibration_utils.save_matrix(T_world_cam, export_dir / "poses" / f"{frame_idx}.txt")
        # Note: We do NOT save per-frame intrinsics anymore
        
        count += 1
    
    # Save single intrinsics.txt at root
    calib_4x4 = first_intrinsic if first_intrinsic is not None else np.eye(4)
    calibration_utils.save_matrix(calib_4x4, export_dir / "intrinsics.txt")
    
    logger.info(f"Organized {count} frames")
    for key, val in stats.items():
        if val > 0:
            logger.warning(f"{val} frames with {key.replace('_', ' ')}")
    
    return count, stats


def process_mesh_and_pcd(logger, export_dir, raw_dir, max_points, voxel_size):
    """Process mesh and point cloud from raw export."""
    # Find textured mesh (Pass 1) in root of raw_dir
    mesh_candidates = list(raw_dir.glob("*.obj")) + list(raw_dir.glob("*.ply"))
    mesh_file = next((f for f in mesh_candidates if f.suffix == '.obj'), mesh_candidates[0] if mesh_candidates else None)
    
    # Find colored mesh (Pass 2) in processing/
    proc_dir = raw_dir / "processing"
    cloud_candidates = []
    if proc_dir.exists():
        cloud_candidates = list(proc_dir.glob("*.ply")) + list(proc_dir.glob("*.obj"))
    
    # Prefer colored mesh for cloud generation, fallback to textured mesh
    cloud_source = cloud_candidates[0] if cloud_candidates else mesh_file
    
    export_stats = {'mesh_found': False, 'mesh_vertices': 0, 'mesh_triangles': 0, 'point_cloud_points': 0}
    
    if not mesh_file:
        logger.warning("No mesh found")
        return export_stats
    
    logger.info(f"Found visual mesh: {mesh_file.name}")
    if cloud_source != mesh_file:
        logger.info(f"Found colored source for cloud: {cloud_source.name}")
        
    export_stats['mesh_found'] = True
    
    try:
        # Load the VISUAL mesh for stats and saving mesh_raw.ply (geometry)
        mesh = mesh_processor.load_mesh(mesh_file)
        mesh = mesh_processor.clean_mesh(mesh)
        export_stats['mesh_vertices'] = len(mesh.vertices)
        export_stats['mesh_triangles'] = len(mesh.triangles)
        # Save raw mesh to processing directory to avoid confusing OpenYOLO3D
        # (It globs *.ply in export_dir and might pick this uncolored mesh)
        mesh_processor.save_mesh_ply(mesh, raw_dir / "processing" / "mesh_geometry.ply")
        logger.info(f"Saved geometry mesh to processing dir: {len(mesh.vertices):,} vertices")
    except Exception as e:
        logger.error(f"Mesh processing failed: {e}")
    
    try:
        n_tex = mesh_processor.copy_textures(raw_dir, export_dir / "textures")
        if n_tex > 0:
            logger.info(f"Copied {n_tex} textures")
    except Exception as e:
        logger.warning(f"Texture copy failed: {e}")
    
    logger.info("Creating point cloud...")
    # OpenYOLO3D expects scene.ply
    scene_ply = export_dir / "scene.ply"
    cloud_ply = export_dir / "cloud.ply"
    
    # Use cloud_source (which presumably has vertex colors from Pass 2)
    if point_cloud_generator.create_point_cloud_pipeline(cloud_source, scene_ply, max_points, voxel_size):
        import open3d as o3d
        pcd = o3d.io.read_point_cloud(str(scene_ply))
        export_stats['point_cloud_points'] = len(pcd.points)
        logger.info(f"Saved point cloud: {len(pcd.points):,} points")
        
    else:
        logger.error("Point cloud creation failed")
    
    # User requested specific naming convention for visualization
    try:
        vis_dir = export_dir / "visualization"
        vis_dir.mkdir(exist_ok=True)
        
        # 2. Save colored mesh as mesh.ply using Poisson Reconstruction
        # This guarantees a watertight surface with colors transferred from the point cloud.
        final_mesh_ply = vis_dir / "mesh.ply"
        logger.info("Generating high-quality mesh using Poisson Reconstruction (this may take a minute)...")
        
        # Use scene.ply (high density, colored cloud) as input
        mesh_reconstruction.reconstruct_mesh(str(scene_ply), str(final_mesh_ply), depth=9)
        
        logger.info(f"Saved reconstructed mesh to {final_mesh_ply.name}")
        
    except Exception as e:
        logger.warning(f"Failed to generate visualization mesh (Poisson): {e}")
    
    return export_stats


def save_export_metadata(logger, metadata_dir, export_stats, frame_stats, args, rel_input, rel_output):
    """Save export statistics to YAML metadata file."""
    export_stats.update({
        'timestamp': datetime.datetime.now().isoformat(),
        'input_db': str(rel_input),
        'output_dir': str(rel_output),
        'command_args': {'max_points': args.max_points, 'voxel_size': args.voxel_size, 'keep_temp': args.keep_temp},
        'warnings': {'skipped_frames': frame_stats['pose_skip'], 'missing_depth': frame_stats['missing_depth'], 
                     'missing_calibration': frame_stats['missing_calib'], 'intrinsic_mismatches': frame_stats['intrinsic_mismatch']}
    })
    
    try:
        export_info_file = metadata_dir / "export_info.yaml"
        with open(export_info_file, 'w') as f:
            yaml.dump(export_stats, f, default_flow_style=False, sort_keys=False)
        logger.info("Saved metadata")
    except Exception as e:
        logger.error(f"Failed to save metadata: {e}")

def main():
    parser = argparse.ArgumentParser(description="Export Data from RTAB-Map DB")
    parser.add_argument("input_db", help="Path to rtabmap.db")
    parser.add_argument("--output_dir", default="data/pipeline_output", help="Output directory")
    parser.add_argument("--max_points", type=int, default=50_000_000, help="Max points to sample from mesh")
    parser.add_argument("--voxel_size", type=float, default=0.001, help="Voxel size (m) for downsampling (Default: 0.001m for high res)")
    parser.add_argument("--keep_temp", action="store_true", help="Keep raw export folder for debugging")
    
    args = parser.parse_args()
    input_db = Path(args.input_db).resolve()
    output_dir = Path(args.output_dir).resolve()
    
    validate_inputs(input_db)
    logger = setup_logger(output_dir)
    
    cwd = Path.cwd()
    rel_output = get_relative_path(output_dir, cwd)
    rel_input = get_relative_path(input_db, cwd)
    
    logger.info(f"Exporting from {rel_input}")
    logger.info(f"Output to {rel_output}")
    
    dirs = setup_directories(output_dir)
    logger.info("Directories ready")
    
    export_rtabmap_data(logger, dirs['raw'], input_db)
    
    logger.info("Organizing frames...")
    # dirs['frames'] was removed, passing dirs['export'] instead
    frame_count, frame_stats = organize_frames(logger, dirs['raw'], dirs['export'])
    
    logger.info("Processing mesh and point cloud...")
    mesh_stats = process_mesh_and_pcd(logger, dirs['export'], dirs['raw'], args.max_points, args.voxel_size)
    
    if args.keep_temp:
        rel_raw = get_relative_path(dirs['raw'], cwd)
        logger.info(f"Kept temp folder: {rel_raw}")
    else:
        shutil.rmtree(dirs['raw'])
        logger.info("Removed temp files")
    
    export_stats = {'total_frames': frame_count, 'voxel_size': args.voxel_size, 'max_samples': args.max_points}
    export_stats.update(mesh_stats)
    save_export_metadata(logger, dirs['metadata'], export_stats, frame_stats, args, rel_input, rel_output)
    
    logger.info(f"\nDone! Output in {rel_output}")


if __name__ == "__main__":
    main()
