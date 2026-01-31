#!/usr/bin/env python3
"""
Run Full Perception Pipeline (Phases 1-3)
1. YOLO-Drawer (Detection + Lifting)
2. Mask3D (Segmentation in Docker)
3. Augmentation (Merge Drawers into Masks)
4. Scene Graph (Build Graph)
"""
import os
import sys
import subprocess
import argparse
from pathlib import Path

def run_command(cmd, cwd=None, env=None):
    print(f"\n[Pipeline] Running: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True, cwd=cwd, env=env)
    except subprocess.CalledProcessError as e:
        print(f"[Pipeline] Error running command: {e}")
        sys.exit(e.returncode)

def main():
    parser = argparse.ArgumentParser(description="Run Full Perception Pipeline")
    parser.add_argument("--data", default="data/pipeline_output", help="Pipeline output directory")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent.parent
    data_dir = (project_root / args.data).resolve()
    
    print(f"[Pipeline] Root: {project_root}")
    print(f"[Pipeline] Data: {data_dir}")
    
    # Setup Env
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{env.get('PYTHONPATH', '')}:{project_root}/source"
    
    # Phase 1: YOLO-Drawer
    print("\n=== Phase 1: YOLO-Drawer ===")
    try:
        run_command([sys.executable, "source/scripts/run_yolodrawer.py", "--data", str(data_dir)], cwd=project_root, env=env)
    except Exception as e:
        print(f"Phase 1 Failed: {e}")

    # Phase 2: Mask3D (Docker)
    print("\n=== Phase 2a: Mask3D (Docker) ===")
    try:
        # Note: We mount project_root to /workspace
        docker_cmd = [
            "docker", "run", "--rm", "--gpus", "all",
            "-v", f"{project_root}:/workspace",
            "-w", "/workspace",
            "-e", "PYTHONPATH=/workspace/source",
            "-e", "PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128",
            # "steve_perception:unified", # Using the image we verified
            "steve_perception:unified",
            "python3", "source/scripts/run_segmentation.py",
            "--data", args.data, # relative path matches inside container if pwd maps correctly
            "--model", "mask3d",
            "--vocab", "furniture"
        ]
        run_command(docker_cmd, cwd=project_root)
        
        # Phase 2b: Fix Permissions
        print("\n=== Phase 2b: Fix Permissions ===")
        uid = os.getuid()
        gid = os.getgid()
        fix_cmd = [
            "docker", "run", "--rm",
            "-v", f"{project_root}:/workspace",
            "craiden/yolodrawer:v1.0", "chown", "-R", f"{uid}:{gid}", f"/workspace/{args.data}/mask3d_output"
        ]
        run_command(fix_cmd, cwd=project_root)
    except Exception as e:
        print(f"Phase 2 Failed: {e}")

    # Phase 3: Combine Inferences
    print("\n=== Phase 3: Combine Inferences ===")
    combined_name = "combined_output"
    try:
        run_command([
            sys.executable, "source/scripts/combine_inferences.py", 
            "--data", str(data_dir),
            "--output", combined_name
        ], cwd=project_root, env=env)
    except Exception as e:
        print(f"Phase 3 Failed: {e}")
    
    # Phase 4: Scene Graph
    print("\n=== Phase 4: Scene Graph (Combined) ===")
    try:
        output_sg = data_dir / "generated_graph"
        cmd_sg = [
            sys.executable, "source/scripts/build_scene_graph.py",
            "--input", str(data_dir / combined_name),
            "--output", str(output_sg),
            "--min-confidence", "0.6" # User requested Mask3D > 60%
        ]
        run_command(cmd_sg, cwd=project_root, env=env)
    except Exception as e:
        print(f"Phase 4 Failed: {e}")
    
    print("\n[Pipeline] Main Execution finished. Running Diagnostics...")
    
    # Phase 5: Diagnostics (Individual Graphs)
    print("\n=== Phase 5: Diagnostics (Individual Graphs) ===")
    
    # 5a. Mask3D Only
    print("Generating Mask3D-only Graph (>60%)...")
    try:
        run_command([
            sys.executable, "source/scripts/build_scene_graph.py",
            "--input", str(data_dir / "mask3d_output"),
            "--output", str(data_dir / "generated_graph_mask3d"),
            "--no-drawers",
            "--min-confidence", "0.6"
        ], cwd=project_root, env=env)
    except Exception as e:
         print(f"Mask3D Diagnostic Failed (Likely upstream failure): {e}")
    
    # 5b. YOLO Only
    print("Generating YOLO-only Graph...")
    try:
        # First, creating YOLO-only combination
        run_command([
            sys.executable, "source/scripts/combine_inferences.py",
            "--data", str(data_dir),
            "--output", "yolo_output",
            "--only-drawers"
        ], cwd=project_root, env=env)
        
        run_command([
            sys.executable, "source/scripts/build_scene_graph.py",
            "--input", str(data_dir / "yolo_output"),
            "--output", str(data_dir / "generated_graph_yolo")
        ], cwd=project_root, env=env)
    except Exception as e:
         print(f"YOLO Diagnostic Failed: {e}")
    
    print("\n[Pipeline] Diagnostics complete!")

if __name__ == "__main__":
    main()
