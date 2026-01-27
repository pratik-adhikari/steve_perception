#!/usr/bin/env python3
"""
Scene Graph Generation Script

Adapted from: Spot-Light repository (source/scenegraph/)
Date: 2026-01-17
Purpose: Generate scene graph from unified segmentation pipeline output

This script reads the standardized output from run_segmentation.py and creates
a scene graph with spatial relationships, generating graph.json, furniture.json,
and individual object files.
"""
import argparse
import sys
import os
import json
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils_source.preprocessing_utils.scene_graph import SceneGraph
from utils_source.scannet_200_labels import CLASS_LABELS_200, VALID_CLASS_IDS_200


def load_label_mapping(scan_dir: str) -> dict:
    """
    Load label mapping from CSV if available, otherwise use ScanNet200 labels.
    
    :param scan_dir: Directory containing the scan data
    :return: Dictionary mapping class IDs to category names
    """
    csv_path = os.path.join(scan_dir, 'mask3d_label_mapping.csv')
    
    if os.path.exists(csv_path):
        print(f"[SceneGraph] Loading label mapping from {csv_path}")
        label_map = pd.read_csv(csv_path, usecols=['id', 'category'])
        return pd.Series(label_map['category'].values, index=label_map['id']).to_dict()
    else:
        print(f"[SceneGraph] Using default ScanNet200 label mapping")
        # Create mapping from VALID_CLASS_IDS_200 to CLASS_LABELS_200
        return dict(zip(VALID_CLASS_IDS_200, CLASS_LABELS_200))


def main():
    parser = argparse.ArgumentParser(
        description='Generate scene graph from segmentation results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python build_scene_graph.py --input results/ --output results/scene_graph/
  python build_scene_graph.py --input results/ --output results/ --min-confidence 0.3
        """
    )
    
    parser.add_argument('--input', required=True,
                       help='Input directory containing mask3d_output/predictions.txt')
    parser.add_argument('--output', required=True,
                       help='Output directory for scene graph files')
    parser.add_argument('--min-confidence', type=float, default=0.1,
                       help='Minimum confidence threshold for nodes (default: 0.1)')
    parser.add_argument('--immovable', nargs='*', default=[],
                       help='List of object categories to mark as immovable')
    parser.add_argument('--no-drawers', action='store_true',
                       help='Disable drawer detection')
    parser.add_argument('--visualize', action='store_true',
                       help='Visualize the scene graph after creation')
    
    args = parser.parse_args()
    
    # Validate input directory
    mask3d_dir = os.path.join(args.input, 'mask3d_output')
    predictions_file = os.path.join(mask3d_dir, 'predictions.txt')
    
    if not os.path.exists(predictions_file):
        print(f"[ERROR] predictions.txt not found at {predictions_file}")
        print(f"[ERROR] Make sure you run run_segmentation.py first")
        return 1
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Load label mapping
    label_mapping = load_label_mapping(args.input)
    
    # Define immovable objects (furniture)
    if not args.immovable:
        immovable = ["table", "chair", "sofa", "bed", "desk", "shelving", "cabinet", 
                    "bookshelf", "counter", "armchair", "shelf", "end table"]
    else:
        immovable = args.immovable
    
    print(f"[SceneGraph] Creating scene graph...")
    print(f"  Input: {args.input}")
    print(f"  Output: {args.output}")
    print(f"  Min confidence: {args.min_confidence}")
    print(f"  Immovable categories: {immovable}")
    
    # Initialize scene graph
    scene_graph = SceneGraph(
        label_mapping=label_mapping,
        min_confidence=args.min_confidence,
        immovable=immovable
    )
    
    # Build scene graph from predictions
    try:
        scene_graph.build(mask3d_dir, drawers=not args.no_drawers)
        print(f"[SceneGraph] Built graph with {len(scene_graph.nodes)} nodes")
    except Exception as e:
        print(f"[ERROR] Failed to build scene graph: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Apply color palette
    scene_graph.color_with_ibm_palette()
    
    # Save outputs
    try:
        # Save full graph
        graph_path = os.path.join(args.output, 'graph.json')
        scene_graph.save_full_graph_to_json(graph_path)
        print(f"[SceneGraph] Saved graph to {graph_path}")
        
        # Save furniture
        furniture_path = os.path.join(args.output, 'furniture.json')
        scene_graph.save_furniture_to_json(furniture_path)
        print(f"[SceneGraph] Saved furniture to {furniture_path}")
        
        # Save individual objects
        objects_dir = os.path.join(args.output, 'objects')
        scene_graph.save_objects_to_json(objects_dir)
        print(f"[SceneGraph] Saved objects to {objects_dir}")
        
        # Save drawers if present
        if not args.no_drawers:
            drawers_dir = os.path.join(args.output, 'drawers')
            scene_graph.save_drawers_to_json(drawers_dir)
            print(f"[SceneGraph] Saved drawers to {drawers_dir}")
        
        print(f"\n[SUCCESS] Scene graph generation complete!")
        print(f"  Nodes: {len(scene_graph.nodes)}")
        print(f"  Connections: {len(scene_graph.outgoing)}")
        
    except Exception as e:
        print(f"[ERROR] Failed to save scene graph: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Optional visualization
    if args.visualize:
        try:
            print(f"\n[SceneGraph] Launching visualization...")
            scene_graph.visualize(labels=True, connections=True, centroids=True)
        except Exception as e:
            print(f"[WARNING] Visualization failed: {e}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
