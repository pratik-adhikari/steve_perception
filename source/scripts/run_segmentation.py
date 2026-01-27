#!/usr/bin/env python3
"""
Unified segmentation pipeline CLI.
Single entry point for all segmentation models.
"""
import argparse
import sys
import os

# CRITICAL: Ensure source directory is in python path BEFORE any local imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import yaml
import shutil
import numpy as np

from models.adapters.base_adapter import SegmentationResult
from models.adapters.openyolo3d_adapter import OpenYolo3DAdapter
from models.adapters.mask3d_adapter import Mask3DAdapter
from utils_source.vocabulary import load_vocabulary

def save_label_mapping(vocabulary: list, output_dir: str):
    """Save vocabulary to CSV for SceneGraph builder."""
    import pandas as pd
    csv_path = os.path.join(output_dir, 'mask3d_label_mapping.csv')
    
    # Create DataFrame with 'id' and 'category' columns
    df = pd.DataFrame({
        'id': range(len(vocabulary)),
        'category': vocabulary
    })
    
    # Save to CSV
    df.to_csv(csv_path, index=False)
    print(f"[Pipeline] Saved label mapping to {csv_path}")

def save_scenegraph_format(result: SegmentationResult, output_dir: str, config):
    """Save results in SceneGraph-compatible format."""
    mask3d_dir = os.path.join(output_dir, 'mask3d_output')
    masks_dir = os.path.join(mask3d_dir, 'masks')
    os.makedirs(masks_dir, exist_ok=True)
    
    mesh_dst = os.path.join(output_dir, 'mesh.ply')
    mesh_labeled_dst = os.path.join(mask3d_dir, 'mesh_labeled.ply')
    
    if result.mesh_path and os.path.exists(result.mesh_path):
        if not os.path.exists(mesh_dst):
            shutil.copy2(result.mesh_path, mesh_dst)
        if not os.path.exists(mesh_labeled_dst):
            shutil.copy2(result.mesh_path, mesh_labeled_dst)
    
    conf_thresh = config['inference'].get('conf_threshold', 0.1)
    
    lines = []
    for i in range(len(result.scores)):
        if result.scores[i] < conf_thresh:
            continue
        
        mask = result.masks[:, i].astype(int)
        mask_filename = f"{i}.txt"
        mask_path = os.path.join(masks_dir, mask_filename)
        np.savetxt(mask_path, mask, fmt='%d')
        
        class_idx = int(result.classes[i])
        rel_path = os.path.join("masks", mask_filename)
        lines.append(f"{rel_path} {class_idx} {result.scores[i]:.4f}")
    
    pred_path = os.path.join(mask3d_dir, 'predictions.txt')
    with open(pred_path, 'w') as f:
        f.write('\n'.join(lines))
    
    print(f"[Pipeline] Saved {len(lines)} masks for SceneGraph to {mask3d_dir}")

def save_detailed_objects(result: SegmentationResult, output_dir: str, vocabulary, config):
    """Save individual object point clouds and metadata."""
    import open3d as o3d
    import json
    
    objects_dir = os.path.join(output_dir, 'objects')
    os.makedirs(objects_dir, exist_ok=True)
    
    conf_thresh = config['inference'].get('conf_threshold', 0.1)
    metadata = {}
    
    for i in range(len(result.scores)):
        if result.scores[i] < conf_thresh:
            continue
        
        class_idx = int(result.classes[i])
        class_name = vocabulary[class_idx] if class_idx < len(vocabulary) else f"class_{class_idx}"
        mask = result.masks[:, i].astype(bool)
        
        obj_points = result.points[mask]
        if len(obj_points) < 10:
            continue
        
        centroid = obj_points.mean(axis=0).tolist()
        min_bb = obj_points.min(axis=0)
        max_bb = obj_points.max(axis=0)
        dimensions = (max_bb - min_bb).tolist()
        
        obj_pcd = o3d.geometry.PointCloud()
        obj_pcd.points = o3d.utility.Vector3dVector(obj_points)
        if result.colors is not None:
            obj_pcd.colors = o3d.utility.Vector3dVector(result.colors[mask])
        
        ply_path = os.path.join(objects_dir, f"{i}_{class_name.replace(' ', '_')}.ply")
        o3d.io.write_point_cloud(ply_path, obj_pcd)
        
        metadata[str(i)] = {
            "class_id": class_idx,
            "class_name": class_name,
            "confidence": float(result.scores[i]),
            "centroid": centroid,
            "dimensions": dimensions,
            "num_points": int(mask.sum())
        }
    
    json_path = os.path.join(objects_dir, 'predictions.json')
    with open(json_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    
    print(f"[Pipeline] Saved {len(metadata)} objects to {objects_dir}")

import datetime
import logging

def setup_logger(output_dir):
    """Setup logging to file and stdout."""
    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = os.path.join(log_dir, f"segmentation_{timestamp}.log")
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s: %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ],
        force=True
    )
    return logging.getLogger(__name__), log_file

def main():
    parser = argparse.ArgumentParser(
        description='Run 3D segmentation with OpenYOLO3D or Mask3D',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_segmentation.py --data data/pipeline_output --model openyolo3d --vocab furniture
        """
    )
    
    parser.add_argument('--data', required=True, help='Path to data directory containing "export" folder')
    parser.add_argument('--model', choices=['openyolo3d', 'mask3d'], help='Model to use (overrides config)')
    parser.add_argument('--vocab', help='Vocabulary mode: lvis, coco, furniture (overrides config)')
    parser.add_argument('--config', help='Path to custom config file')
    parser.add_argument('--frame-step', type=int, help='Frame step for OpenYOLO3D (overrides config)')
    parser.add_argument('--conf-threshold', type=float, help='Confidence threshold (overrides config)')
    
    args = parser.parse_args()
    
    # Path Derivation
    data_dir = os.path.abspath(args.data)
    scene_dir = os.path.join(data_dir, 'export')
    
    # Validation
    if not os.path.isdir(scene_dir):
        print(f"[ERROR] Export directory not found: {scene_dir}")
        print(f"       Expected structure: {args.data}/export/")
        sys.exit(1)
        
    # Load config (early load for model name defaults)
    config_path = args.config
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), '../configs/segmentation.yaml')
        
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    # Determine model name for output folder
    model_name = args.model if args.model else config['model']['name']
    output_dir = os.path.join(data_dir, f"{model_name}_output")
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup Logger
    logger, log_file = setup_logger(data_dir)
    logger.info(f"Starting Segmentation Pipeline")
    logger.info(f"Data Directory: {data_dir}")
    logger.info(f"Input Scene:    {scene_dir}")
    logger.info(f"Output Dir:     {output_dir}")
    logger.info(f"Log File:       {log_file}")
    
    # Apply Overrides
    if args.model:
        config['model']['name'] = args.model
        logger.info(f"Override model: {args.model}")
    
    if args.vocab:
        config['vocabulary']['mode'] = args.vocab
        logger.info(f"Override vocabulary: {args.vocab}")
    
    if args.frame_step:
        config['inference']['frame_step'] = args.frame_step
        logger.info(f"Override frame_step: {args.frame_step}")
    
    if args.conf_threshold:
        config['inference']['conf_threshold'] = args.conf_threshold
        logger.info(f"Override conf_threshold: {args.conf_threshold}")
        
    # Initialize adapter
    if model_name == 'openyolo3d':
        adapter = OpenYolo3DAdapter(config)
    elif model_name == 'mask3d':
        adapter = Mask3DAdapter(config)
    else:
        logger.error(f"Unknown model: {model_name}")
        return 1
    
    logger.info(f"Initialized {model_name} adapter")
    
    # Vocabulary
    vocab_mode = config['vocabulary']['mode']
    custom_classes = config['vocabulary'].get('custom_classes', None)
    vocabulary = load_vocabulary(vocab_mode, custom_classes)
    
    logger.info(f"Using vocabulary: {vocab_mode} ({len(vocabulary)} classes)")
    
    # Run Inference
    try:
        # Capture stdout/stderr to redirect library output to logger
        import io
        import contextlib
        
        # Create string buffers  
        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()
        
        logger.info("="*60)
        logger.info("Starting model inference (capturing all output)...")
        logger.info("="*60)
        
        # Capture all print statements from the library
        with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
            result = adapter.predict(scene_dir, vocabulary, output_dir)
        
        # Log captured output
        stdout_content = stdout_buffer.getvalue()
        stderr_content = stderr_buffer.getvalue()
        
        if stdout_content:
            logger.info("="*60)
            logger.info("Captured stdout from model:")
            logger.info("="*60)
            for line in stdout_content.split('\n'):
                if line.strip():
                    logger.info(f"  {line}")
        
        if stderr_content:
            logger.warning("="*60)
            logger.warning("Captured stderr from model:")
            logger.warning("="*60)
            for line in stderr_content.split('\n'):
                if line.strip():
                    logger.warning(f"  {line}")
        
        logger.info("="*60)
        logger.info("Segmentation complete!")
        logger.info(f"Instances detected: {len(result.scores)}")
        logger.info("="*60)
        
        # Save Label Mapping (Critical for SceneGraph)
        save_label_mapping(vocabulary, output_dir)
        
        # Save Outputs
        if config['output'].get('save_scenegraph_format', True):
            save_scenegraph_format(result, output_dir, config)
            
        if config['output'].get('save_detailed_objects', True):
            save_detailed_objects(result, output_dir, vocabulary, config)
            
        # Note: Scene Graph generation removed per user request.
        # It is now a separate step to be run on the data folder.
            
        return 0
    except Exception as e:
        logger.error(f"Segmentation failed: {e}", exc_info=True)
        return 1

if __name__ == '__main__':
    sys.exit(main())
