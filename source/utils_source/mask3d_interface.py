"""
Mask3D Interface Wrapper
"""
import os
import sys
from pathlib import Path

# Add Mask3D library to path
mask3d_lib_path = os.path.join(os.path.dirname(__file__), '../models/lib/Mask3D')
if mask3d_lib_path not in sys.path:
    sys.path.insert(0, mask3d_lib_path)

# Import the actual run function
try:
    # Mute stdout during import to suppress "Hydra" or other logs if needed
    # But for now we just import
    from mask3d import run_mask3d as run_mask3d_lib
except ImportError as e:
    print(f"Failed to import Mask3D: {e}")
    run_mask3d_lib = None

def run_mask3d(
    config_path: str,
    scene_path: str,
    output_path: str,
    device: str = "cuda"
):
    """
    Run Mask3D inference on a scene.
    
    :param config_path: Ignored (Mask3D lib uses its own internal config/checkpoint path).
    :param scene_path: Path to input point cloud (PLY) or scene directory.
                       Mask3D expects a directory containing 'mesh.ply' or 'textured_output.obj'.
                       If scene_path is a file (e.g. scene.ply), we assume the parent dir is the workspace.
    :param output_path: Where to save results (not fully used by Mask3D lib which saves in scene_dir, 
                        but we can move them if needed).
    :param device: Device to run on
    """
    if run_mask3d_lib is None:
        raise ImportError("Mask3D library count not be imported.")

    scene_path_obj = Path(scene_path)
    workspace = scene_path_obj.parent if scene_path_obj.is_file() else scene_path_obj
    
    # Mask3D expects "mesh.ply" in workspace if pcd=True. 
    # Our export pipeline produces "scene.ply". 
    # We might need to symlink scene.ply -> mesh.ply if not exists
    mesh_ply = workspace / "mesh.ply"
    scene_ply = workspace / "scene.ply"
    
    # Check if scene.ply exists (from export)
    if not mesh_ply.exists():
        if scene_ply.exists():
            # Create symlink or copy
            # Symlink is safer/faster
            try:
                os.symlink(scene_ply, mesh_ply)
            except OSError:
                # If symlink fails (e.g. cross-device), copy? Or just print warning
                import shutil
                shutil.copy(scene_ply, mesh_ply)
        else:
             print(f"Warning: Neither mesh.ply nor scene.ply found in {workspace}")

    print(f"[Mask3D Interface] Running inference in {workspace}...")
    
    # Run inference
    # mask3d.py: run_mask3d(scene_dir, device, flip, pcd)
    # We use pcd=True to use mesh.ply (which we ensured exists)
    try:
        run_mask3d_lib(
            scene_dir=workspace,
            device=device,
            flip=False,
            pcd=True 
        )
    except Exception as e:
        print(f"Mask3D failed: {e}")
        return False
        
    return True
