import os
import glob
import json
import cv2
import numpy as np
import open3d as o3d
from tqdm import tqdm
import argparse
import datetime
import logging
from pathlib import Path

# Reusing existing utils
from utils_source.recursive_config import Config
from utils_source.preprocessing_utils.drawer_integration import register_drawers

def setup_logger(data_dir):
    log_dir = data_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = log_dir / f"run_yolodrawer_{timestamp}.log"
    
    logger = logging.getLogger("DrawerDetection")
    logger.setLevel(logging.INFO)
    
    fh = logging.FileHandler(str(log_file))
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
    parser = argparse.ArgumentParser(description="Run YOLO-Drawer 2D Detection & 3D Lifting (Integrated Fusion)")
    parser.add_argument("--data", required=True, help="Path to data directory (containing 'export' folder)")
    args = parser.parse_args()

    data_dir = Path(args.data).resolve()
    export_dir = data_dir / "export"
    output_dir = data_dir / "drawer_priors"
    debug_dir = output_dir / "debug_images"
    
    logger = setup_logger(data_dir)
    logger.info("Starting YOLO-Drawer Pipeline (Refactored with Fusion)")
    logger.info(f"Input: {export_dir}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Debug: {debug_dir}")
    
    # Pre-flight
    if not export_dir.exists():
        logger.error(f"Export directory not found at {export_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)
    
    # Check for mesh (required for register_drawers)
    if not (export_dir / "mesh.ply").exists() and not (export_dir / "mesh_labeled.ply").exists() and not (export_dir / "scene.ply").exists():
        logger.error("No mesh found (mesh.ply/scene.ply/mesh_labeled.ply) in export directory! 3D lifting requires a mesh.")
        return

    # Run Integrated Drawer Registration
    try:
        # returns list of dicts: {'bbox': o3d_obj, 'confidence': float, 'label': str, 'indices': np.array}
        logger.info("Running detection, lifting, and clustering (this may take a while)...")
        drawers = register_drawers(str(export_dir), vis_block=True, debug_output_dir=str(debug_dir))
        logger.info(f"Clustering complete. Found {len(drawers)} unique objects.")
    except Exception as e:
        logger.error(f"Failed to register drawers: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return

    # Serialize to JSON
    json_entries = []
    
    for item in drawers:
        bbox = item['bbox']
        conf = item['confidence']
        label = item['label']
        
        # Convert Open3D bbox to serializable format
        center = bbox.center.tolist()
        extent = bbox.extent.tolist()
        rotation = bbox.R.tolist()
        source_img = item.get('source_img', "unknown")
        
        json_entries.append({
            "center": center,
            "extent": extent,
            "rotation": rotation,
            "label": label,
            "score": float(conf),
            "source_imgs": item.get('source_imgs', [])
        })
        
    output_json = output_dir / "boxes_3d.json"
    with open(output_json, "w") as f:
        json.dump(json_entries, f, indent=4)
        
    logger.info(f"Saved {len(json_entries)} detections to {output_json}")
    logger.info(f"Visualizations saved to {debug_dir}")
    logger.info("Pipeline Step 1 Complete.")

if __name__ == "__main__":
    main()
