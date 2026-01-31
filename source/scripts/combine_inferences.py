import os
import sys
import json
import glob
import numpy as np
import open3d as o3d
from pathlib import Path
from tqdm import tqdm

# Add utils to path if needed, though we mostly use standard libs
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils_source.recursive_config import Config

def load_drawers(prior_path):
    with open(prior_path, 'r') as f:
        data = json.load(f)
    return data

import argparse
import shutil
import datetime
import logging

def setup_logger(data_dir):
    log_dir = data_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = log_dir / f"combine_inferences_{timestamp}.log"
    
    logger = logging.getLogger("CombineInferences")
    logger.setLevel(logging.INFO)
    
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    return logger

def main():
    parser = argparse.ArgumentParser(description="Combine Mask3D and YOLO-Drawer Outputs")
    parser.add_argument("--data", required=True, help="Path to data directory containing 'mask3d_output' and 'drawer_priors'")
    parser.add_argument("--output", default="combined_output", help="Name of the output folder to create (inside --data)")
    parser.add_argument("--only-drawers", action="store_true", help="If set, ignores Mask3D output and builds dataset from drawers only")
    args = parser.parse_args()

    # Paths
    data_dir = Path(args.data).resolve()
    logger = setup_logger(data_dir)
    
    mask3d_src_dir = data_dir / "mask3d_output"
    prior_dir = data_dir / "drawer_priors"
    combined_dir = data_dir / args.output
    
    logger.info(f"Starting Combination Pipeline (Only Drawers: {args.only_drawers})")
    logger.info(f"Data Dir: {data_dir}")
    logger.info(f"Output:   {combined_dir}")
    
    # Pre-flight
    if not args.only_drawers and not mask3d_src_dir.exists():
        logger.error(f"Mask3D output not found at: {mask3d_src_dir}")
        return
    if not prior_dir.exists():
        logger.error(f"Drawer priors not found at: {prior_dir}")
        return

    boxes_file = prior_dir / "boxes_3d.json"
    if not boxes_file.exists():
         logger.error(f"boxes_3d.json not found in {prior_dir}")
         return

    # Create Combined Directory
    logger.info(f"Creating output at: {combined_dir}")
    if combined_dir.exists():
        shutil.rmtree(combined_dir)
    
    if args.only_drawers:
        combined_dir.mkdir(parents=True, exist_ok=True)
        # We need the mesh/scene ply for visualization/structure
        export_scene = data_dir / "export" / "scene.ply"
        if export_scene.exists():
             shutil.copy2(export_scene, combined_dir / "mesh.ply")
        else:
             logger.warning(f"Could not find scene.ply in export to copy to {combined_dir}")
             # We let downstream handle missing mesh if possible, but augment needs it
    else:
        shutil.copytree(mask3d_src_dir, combined_dir)
    
    # Now Augment inside combined_dir
    pred_mask_dir = combined_dir / "pred_mask"
    pred_mask_dir.mkdir(parents=True, exist_ok=True)
    
    # We need scene.ply to define mask size (num points)
    scene_file = combined_dir / "mesh.ply"
    if not scene_file.exists():
        # Fallback to export/scene.ply
        scene_file = data_dir / "export" / "scene.ply"
        
    if not scene_file.exists():
        logger.error(f"Could not find scene point cloud at {scene_file} or inside combined output.")
        return

    # Load Scene
    logger.info(f"Loading scene from {scene_file}...")
    pcd = o3d.io.read_point_cloud(str(scene_file))
    points = o3d.utility.Vector3dVector(np.asarray(pcd.points))
    
    # Load Drawers
    drawers = load_drawers(boxes_file)
    logger.info(f"Loaded {len(drawers)} drawer priors.")
    
    # Check existing masks index
    existing_masks = glob.glob(str(pred_mask_dir / "*.txt"))
    start_idx = 0
    if existing_masks:
        indices = [int(Path(f).stem) for f in existing_masks if Path(f).stem.isdigit()]
        if indices:
            start_idx = max(indices) + 1
            
    logger.info(f"Starting augmentation at index {start_idx:03d}")
    
    new_entries = []
    
    # Mapping
    # Mapping to ScanNet200 IDs
    CLASS_MAP = {
        "cabinet": 7,            # ScanNet200 ID for cabinet
        "cabinet door": 7,       # Map door to cabinet base
        "refrigerator": 24,      # ScanNet200 ID for refrigerator
        "refrigerator door": 24, # Map door to fridge base
        "drawer": 25,            # ScanNet200 ID for 'kitchen cabinet' (close enough) OR custom reserved ID
        "door": 14               # ScanNet200 ID for door
    }
    # Note: ID 25 is "kitchen cabinet" in ScanNet200, which is reasonable for drawers if we want valid semantic meaning.
    # However, SceneGraph might treat ID specially if we define a mapping.
    # Let's stick to valid ScanNet200 IDs to avoid "ID not found" if mapping is missing.
    # 25 is "kitchen cabinet". 
    # Or we can use a custom ID like 999 if we provide a mapping CSV.
    # Let's use the USER APPROVED plan: 25.


    
    for i, drawer in tqdm(enumerate(drawers)):
        center = np.array(drawer['center'])
        extent = np.array(drawer['extent'])
        R = np.array(drawer['rotation'])
        label_text = drawer['label']
        score = drawer['score']
        
        obb = o3d.geometry.OrientedBoundingBox(center, R, extent)
        indices = obb.get_point_indices_within_bounding_box(points)
        
        if len(indices) == 0:
            continue
            
        full_mask = np.zeros(len(pcd.points), dtype=int)
        full_mask[indices] = 1
        
        mask_filename = f"{start_idx + i:03d}.txt"
        mask_path = pred_mask_dir / mask_filename
        np.savetxt(str(mask_path), full_mask, fmt='%d')
        
        class_id = CLASS_MAP.get(label_text, 3)
        new_entries.append(f"pred_mask/{mask_filename} {class_id} {score:.4f}")

    # Append to predictions files
    pred_file = combined_dir / "predictions_drawers.txt"
    with open(pred_file, "w") as f:
        for entry in new_entries:
            f.write(entry + "\n")
            
    main_pred = combined_dir / "predictions.txt"
    with open(main_pred, "a") as f:
        for entry in new_entries:
            f.write(entry + "\n")
                
    logger.info(f"Augmented {len(new_entries)} masks in {combined_dir}.")
    
    # Save explicit mapping to ensure Scene Graph Builder interprets them correctly
    mapping_path = combined_dir / "mask3d_label_mapping.csv"
    with open(mapping_path, "w") as f:
        f.write("id,category\n")
        f.write("7,cabinet\n")
        f.write("24,refrigerator\n")
        f.write("25,drawer\n")
        f.write("14,door\n")

        # Add other common ScanNet200 labels we might want consistent
        f.write("3,floor\n") # Keep 3 as floor just in case (though we don't output it)
        f.write("70,washing machine\n") # Keep 70 as washing machine to document the conflict we solved
    logger.info(f"Saved label mapping to {mapping_path}")
    
    # Save Dimensions Mapping (Fix for 2D Square Drawers)
    # Map from integer index (start_idx + i) to YOLO 3D Box Data
    dimensions_map = {}
    for i, drawer in enumerate(drawers):
        mask_id = start_idx + i
        dimensions_map[mask_id] = {
            "center": drawer['center'],
            "extent": drawer['extent'],
            "rotation": drawer['rotation'],
            "label": drawer['label'],
            "source_imgs": drawer.get('source_imgs', [])
        }
    
    dims_path = combined_dir / "drawer_dimensions.json"
    with open(dims_path, "w") as f:
        json.dump(dimensions_map, f, indent=4)
    logger.info(f"Saved drawer dimensions to {dims_path}")


if __name__ == "__main__":
    main()
