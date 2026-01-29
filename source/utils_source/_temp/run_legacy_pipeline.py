#!/usr/bin/env python3
"""
Legacy Pipeline Orchestrator.
Runs the distributed Mask3D + YOLO Drawer pipeline using the original Docker containers.
"""

import os
import sys
import subprocess
import time
import numpy as np
import argparse
import shutil
from pathlib import Path
import open3d as o3d

def run_cmd(cmd, bg=False, wait=True):
    print(f"Running: {' '.join(cmd)}")
    if bg:
        return subprocess.Popen(cmd)
    else:
        subprocess.run(cmd, check=True)


def patch_config(config_path, scan_name):
    """
    Temporarily patch config.yaml to point to our scan.
    Returns: backup_path (or None if failed)
    """
    config_path = Path(config_path)
    if not config_path.exists():
        print(f"Error: Config not found at {config_path}")
        return None
        
    backup_path = config_path.parent / "config.yaml.bak"
    shutil.copy2(config_path, backup_path)
    
    with open(config_path, 'r') as f:
        lines = f.readlines()
        
    new_lines = []
    for line in lines:
        if "high_res:" in line:
            new_lines.append(f'  high_res: "{scan_name}"\n')
        elif "low_res:" in line:
            new_lines.append(f'  low_res: "{scan_name}"\n')
        else:
            new_lines.append(line)
            
    with open(config_path, 'w') as f:
        f.writelines(new_lines)
    
    print("Patched config.yaml")
    return backup_path

def restore_config(config_path, backup_path):
    if backup_path and backup_path.exists():
        shutil.move(backup_path, config_path)
        print("Restored config.yaml")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_ply", required=True, help="Path to scene.ply (high quality)")
    args = parser.parse_args()
    
    # Paths
    steve_ws = Path(os.getcwd()) # Assumes running from steve_perception
    compose_root = steve_ws.parent / "stretch-compose"
    
    if not compose_root.exists():
        print(f"Error: stretch-compose not found at {compose_root}")
        sys.exit(1)

    scan_name = "legacy_test_run"
    data_dir = compose_root / "data"
    scan_dir = data_dir / "ipad_scans" / scan_name
    scan_dir.mkdir(parents=True, exist_ok=True)
    
    # --- PROVISION DATA FOR LEGACY PIPELINE ---
    
    # 1. mesh_labeled.ply (Required by SceneGraph Preprocessing)
    target_labeled = scan_dir / "mesh_labeled.ply" 
    if not target_labeled.exists():
        print(f"Copying {args.input_ply} to {target_labeled}...")
        shutil.copy2(args.input_ply, target_labeled)

    # 2. mesh.ply (Required by Mask3D as 'pcd' source)
    target_mesh_ply = scan_dir / "mesh.ply"
    if not target_mesh_ply.exists():
        print(f"Copying {args.input_ply} to {target_mesh_ply}...")
        shutil.copy2(args.input_ply, target_mesh_ply)
        
    # 3. textured_output.obj (Required by Mask3D as 'mesh' source)
    target_obj = scan_dir / "textured_output.obj"
    if not target_obj.exists():
        print("Generating textured_output.obj from visualization/mesh.ply...")
        # Check input directory for visualization mesh
        input_dir = Path(args.input_ply).parent
        vis_mesh = input_dir / "visualization" / "mesh.ply"
        
        if not vis_mesh.exists():
             # Fallback: Use scene.ply (might be sparse/just points, but better than crashing)
             vis_mesh = Path(args.input_ply)
             
        try:
             mesh = o3d.io.read_triangle_mesh(str(vis_mesh))
             # If strictly point cloud, mesh might be empty?
             if len(mesh.vertices) == 0:
                 # Try reading as point cloud and converting to dummy mesh?
                 # Or just save an empty mesh?
                 pass
             o3d.io.write_triangle_mesh(str(target_obj), mesh)
             print(f"Saved {target_obj}")
        except Exception as e:
             print(f"Warning: Failed to create OBJ: {e}")
             pass

    # 4. aruco_pose.npy (Required to align coordinate systems)
    # Since our data is already from RTAB-Map (aligned), we provide Identity.
    target_pose = scan_dir / "aruco_pose.npy"
    if not target_pose.exists():
        print("Generating aruco_pose.npy (Identity)...")
        np.save(target_pose, np.eye(4))
        
    # 5. aligned_point_clouds/legacy_test_run/pose/icp_tform_ground.txt (Required for coordinate change)
    # Provides transform to Stretch Frame. We assume Identity (already aligned).
    aligned_dir = data_dir / "aligned_point_clouds" / scan_name / "pose"
    aligned_dir.mkdir(parents=True, exist_ok=True)
    tform_file = aligned_dir / "icp_tform_ground.txt"
    if not tform_file.exists():
        print("Generating icp_tform_ground.txt (Identity)...")
        with open(tform_file, 'w') as f:
            # Write 4x4 identity matrix
            eye = np.eye(4)
            for row in eye:
                f.write(" ".join(map(str, row)) + "\n")

    # Also Symlink `color` folder if it exists in input_ply parent?
    # Our export puts `scene.ply` in `export/`.
    # `color/` is in `raw/` or `data/`?
    # In my new export structure: `data/pipeline_output/export/scene.ply`, `data/pipeline_output/frames` (no, I removed frames dir).
    # Wait, `export_data.py`: `organize_frames` copies images to `export_dir/color`?
    # No, `organize_frames` calls `process_frames` which saves to `output_dir/color`.
    # Yes. `dirs['export']` contains `color`, `depth`, `poses`.
    
    # So `args.input_ply` (scene.ply) is in a folder that ALSO contains `color` and `poses`.
    # I need to symlink/copy `color`, `poses`, and `intrinsics.txt` to `scan_dir`.
    
    input_dir = Path(args.input_ply).parent
    
    for item in ["color", "poses", "intrinsics.txt"]:
        src = input_dir / item
        dst = scan_dir / item
        if src.exists():
            # Check if dst exists (including broken symlinks)
            if os.path.lexists(dst):
                 if os.path.islink(dst) or os.path.isfile(dst):
                     os.remove(dst)
                 else:
                     shutil.rmtree(dst)
            
            print(f"Linking {item}...")
            # Use symlink for speed
            try:
                os.symlink(src, dst)
            except OSError:
                 # Fallback to copy if symlink fails (e.g. cross-device)
                 if src.is_dir():
                     shutil.copytree(src, dst)
                 else:
                     shutil.copy2(src, dst)
        else:
            print(f"Warning: {item} not found in {input_dir}")
    
    # Ensure label mapping exists
    src_map = steve_ws / "data" / "mask3d_label_mapping.csv"
    dst_map = data_dir / "mask3d_label_mapping.csv"
    if src_map.exists() and not dst_map.exists():
        shutil.copy2(src_map, dst_map)
        
    # 2. Patch Config
    config_yaml = compose_root / "configs" / "config.yaml"
    backup = patch_config(config_yaml, scan_name)
    
    yolo_container_name = "legacy_yolo_service"
    
    try:
        # 3. Start YOLO
        print("Starting YOLO Service...")
        subprocess.run(["docker", "rm", "-f", yolo_container_name], stderr=subprocess.DEVNULL)
        
        # Note: Mounting data_dir just in case, though it runs as service
        subprocess.run([
            "docker", "run", 
            "--name", yolo_container_name,
            "-p", "5004:5004",
            "--gpus", "all",
            "-d", 
            "craiden/yolodrawer:v1.0",
            "python3", "app.py"
        ], check=True)
        
        print("Waiting 15s for YOLO...")
        time.sleep(15)
        
        # 4. Run Mask3D (One-off)
        print("Running Mask3D...")
        # Mask3D needs to see the data. We mount compose_root/data to /home/stretch/workspace/stretch-compose/data
        # to match the paths logic if possible, or just mount to /workspace and rely on relative paths?
        # The legacy command was:
        # -v /home:/home
        # --workspace /home/stretch/workspace/stretch-compose/data/ipad_scans/2025_08_21
        
        # We will mount `compose_root` to `/legacy_workspace` and pass that path.
        # But `mask3d.py` might be strict about paths?
        # Let's try mounting `data_dir` to `/data`.
        
        # Command: `python3 mask3d.py --seed 42 --workspace <scan_dir> --pcd`
        # We need to map `scan_dir` to something visible inside.
        
        abs_scan_dir = str(scan_dir.resolve())
        # We mount the HOST path of scan_dir to the SAME path inside container to avoid confusion?
        # Or simpler: mount `scan_dir` to `/input`.
        # args: `--workspace /input`
        
        # We assume mask3d.py is inside the image `rupalsaxena/mask3d_docker`
        # ENTRYPOINT is /bin/bash, so we must use -c "command"
        
        # Mask3D source is in stretch-compose/source/Mask3D
        mask3d_src = compose_root / "source" / "Mask3D"
        if not mask3d_src.exists():
             print(f"Error: Mask3D source not found at {mask3d_src}")
             raise FileNotFoundError("Mask3D source missing")
             
        mask3d_exec = f"python3 mask3d.py --seed 42 --workspace /input --pcd"
        
        abs_src = str(mask3d_src.resolve())
        
        # Check if Mask3D results already exist to skip re-run
        pred_mask_dir = scan_dir / "pred_mask"
        if pred_mask_dir.exists() and any(pred_mask_dir.iterdir()):
             print("Mask3D results found. Skipping execution.")
        else:
            print(f"Running Mask3D command: {mask3d_exec}")
            subprocess.run([
                "docker", "run",
                "--rm",
                "--gpus", "all",
                "-v", f"{abs_scan_dir}:/input",
                "-v", f"{abs_src}:/Mask3D_code",
                "-w", "/Mask3D_code",
                "rupalsaxena/mask3d_docker:latest",
                "-c", mask3d_exec
            ], check=True)
        
        # 5. Run SceneGraph Preprocessing
        print("Running SceneGraph Architecture...")
        env = os.environ.copy()
        env["PYTHONPATH"] = f"{compose_root}/source:{env.get('PYTHONPATH','')}"
        
        script = compose_root / "source" / "scripts" / "preprocessing_scripts" / "scenegraph_preprocessing.py"
        subprocess.run(["python3", str(script)], env=env, check=True)
        
        print("\nOriginal Pipeline Execution Completed Successfully!")
        print(f"Results should be in {scan_dir}")
        
    except Exception as e:
        print(f"Pipeline Failed: {e}")
    finally:
        print("Cleaning up...")
        restore_config(config_yaml, backup)
        subprocess.run(["docker", "rm", "-f", yolo_container_name], stderr=subprocess.DEVNULL)

if __name__ == "__main__":
    main()
