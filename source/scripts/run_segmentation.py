#!/usr/bin/env python3
"""
Unified segmentation pipeline CLI.
Single entry point for all segmentation models.
"""
import argparse
import sys
import os
import yaml
import shutil
import numpy as np

# Ensure source directory is in python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models.adapters.base_adapter import SegmentationResult
from models.adapters.openyolo3d_adapter import OpenYolo3DAdapter
from models.adapters.mask3d_adapter import Mask3DAdapter
from utils.vocabulary import load_vocabulary

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

def main():
    parser = argparse.ArgumentParser(
        description='Run 3D segmentation with OpenYOLO3D or Mask3D',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_segmentation.py --scene scene/ --output results/ --model openyolo3d --vocab furniture
  python run_segmentation.py --scene scene/ --output results/ --model mask3d --config custom.yaml
        """
    )
    
    parser.add_argument('--scene', required=True, help='Path to scene directory')
    parser.add_argument('--output', required=True, help='Output directory')
    parser.add_argument('--model', choices=['openyolo3d', 'mask3d'], help='Model to use (overrides config)')
    parser.add_argument('--vocab', help='Vocabulary mode: lvis, coco, furniture (overrides config)')
    parser.add_argument('--config', help='Path to custom config file')
    parser.add_argument('--frame-step', type=int, help='Frame step for OpenYOLO3D (overrides config)')
    parser.add_argument('--conf-threshold', type=float, help='Confidence threshold (overrides config)')
    
    args = parser.parse_args()
    
    # Load config
    config_path = args.config
    if config_path is None:
        # Default to source/configs/segmentation.yaml relative to this script
        config_path = os.path.join(os.path.dirname(__file__), '../configs/segmentation.yaml')
        
    print(f"[CLI] Loading config from {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Overrides
    if args.model:
        config['model']['name'] = args.model
        print(f"[CLI] Overriding model: {args.model}")
    
    if args.vocab:
        config['vocabulary']['mode'] = args.vocab
        print(f"[CLI] Overriding vocabulary: {args.vocab}")
    
    if args.frame_step:
        config['inference']['frame_step'] = args.frame_step
        print(f"[CLI] Overriding frame_step: {args.frame_step}")
    
    if args.conf_threshold:
        config['inference']['conf_threshold'] = args.conf_threshold
        print(f"[CLI] Overriding conf_threshold: {args.conf_threshold}")
        
    # Initialize adapter
    model_name = config['model']['name']
    if model_name == 'openyolo3d':
        adapter = OpenYolo3DAdapter(config)
    elif model_name == 'mask3d':
        adapter = Mask3DAdapter(config)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    print(f"[Pipeline] Initialized {model_name} adapter")
    
    # Vocabulary
    vocab_mode = config['vocabulary']['mode']
    custom_classes = config['vocabulary'].get('custom_classes', None)
    vocabulary = load_vocabulary(vocab_mode, custom_classes)
    
    print(f"[Pipeline] Using vocabulary: {vocab_mode} ({len(vocabulary)} classes)")
    print(f"[Pipeline] Scene: {args.scene}")
    print(f"[Pipeline] Output: {args.output}")
    
    # Run Inference
    try:
        result = adapter.predict(args.scene, vocabulary, args.output)
        
        print(f"\n[SUCCESS] Segmentation complete!")
        print(f"  Instances: {len(result.scores)}")
        
        # Save Outputs
        if config['output'].get('save_scenegraph_format', True):
            save_scenegraph_format(result, args.output, config)
            
        if config['output'].get('save_detailed_objects', True):
            save_detailed_objects(result, args.output, vocabulary, config)
            
        return 0
    except Exception as e:
        print(f"\n[ERROR] Segmentation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main())
