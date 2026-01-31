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
        return None

def get_default_label_mapping() -> dict:
    print(f"[SceneGraph] Using default ScanNet200 label mapping")
    return dict(zip(VALID_CLASS_IDS_200, CLASS_LABELS_200))


import datetime
import logging

def setup_logger(output_path):
    # Try to find common logs folder (../logs relative to output)
    # If output is 'data/pipeline_output/generated_graph', parent is 'data/pipeline_output'
    # logs should be 'data/pipeline_output/logs'
    
    out_path = Path(output_path).resolve()
    parent = out_path.parent
    log_dir = parent / "logs"
    
    # If parent doesn't look like a pipeline root, just log locally?
    # But let's force creation.
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = log_dir / f"scene_graph_{timestamp}.log"
    
    logger = logging.getLogger("SceneGraph")
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
    
    # Setup Logger
    logger = setup_logger(args.output)
    
    # Validate input directory
    # User provides the main output folder (e.g. data/pipeline_output/combined_output)
    # Wait, the script historically appended 'mask3d_output'.
    # OLD CODE: mask3d_dir = os.path.join(args.input, 'mask3d_output')
    # BUT if input IS the combined output (which mimics mask3d_output structure somewhat), maybe we shouldn't append?
    # CombineInferences puts stuff DIRECTLY in combined_output/pred_mask, output/predictions.txt
    # So combined_output structure IS:
    #   combined_output/predictions.txt
    #   combined_output/pred_mask/
    #   combined_output/mesh.ply
    
    # BUT build_scene_graph logic (lines 76-77):
    # mask3d_dir = os.path.join(args.input, 'mask3d_output')
    # predictions_file = os.path.join(mask3d_dir, 'predictions.txt')
    
    # If we run Step 3: combine_inferences output 'combined_output'
    # It contains predictions.txt directly.
    # So if we pass --input combined_output, the old script will look for combined_output/mask3d_output/predictions.txt
    # THIS WILL FAIL.
    
    # I should check if predictions.txt exists directly in args.input, verify, and fallback.
    
    mask3d_dir = args.input
    predictions_file = os.path.join(mask3d_dir, 'predictions.txt')
    
    if not os.path.exists(predictions_file):
        # Try legacy structure
        mask3d_dir = os.path.join(args.input, 'mask3d_output')
        predictions_file = os.path.join(mask3d_dir, 'predictions.txt')
        
    if not os.path.exists(predictions_file):
        logger.error(f"predictions.txt not found at {predictions_file} or in {args.input}")
        logger.error(f"Make sure you run run_segmentation.py first")
        return 1
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Load label mapping
    label_mapping = load_label_mapping(args.input)
    
    if not label_mapping:
        logger.warning("Label mapping CSV not found. Using default ScanNet200 mapping.")
        label_mapping = get_default_label_mapping()

    if not label_mapping:
         logger.error("Failed to load any label mapping.")
         return 1
    
    # Define immovable objects (furniture)
    if not args.immovable:
        immovable = ["table", "chair", "sofa", "bed", "desk", "shelving", "cabinet", 
                    "bookshelf", "counter", "armchair", "shelf", "end table",
                    "refrigerator", "stove", "oven", "washing machine", "dishwasher", "fireplace", "door"]

    else:
        immovable = args.immovable
    
    logger.info(f"Creating scene graph...")
    logger.info(f"  Input: {mask3d_dir}")
    logger.info(f"  Output: {args.output}")
    logger.info(f"  Min confidence: {args.min_confidence}")
    logger.info(f"  Immovable categories: {immovable}")
    
    # Initialize scene graph
    scene_graph = SceneGraph(
        label_mapping=label_mapping,
        min_confidence=args.min_confidence,
        immovable=immovable
    )
    
    # Build scene graph from predictions
    try:
        scene_graph.build(mask3d_dir, drawers=not args.no_drawers)
        logger.info(f"Built graph with {len(scene_graph.nodes)} nodes")
    except Exception as e:
        logger.error(f"Failed to build scene graph: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1
    
    # Apply color palette
    # scene_graph.color_with_ibm_palette()
    
    # Save outputs
    try:
        # Save full graph
        graph_path = os.path.join(args.output, 'graph.json')
        scene_graph.save_full_graph_to_json(graph_path)
        logger.info(f"Saved graph to {graph_path}")
        
        # Save furniture
        furniture_path = os.path.join(args.output, 'furniture.json')
        scene_graph.save_furniture_to_json(furniture_path)
        logger.info(f"Saved furniture to {furniture_path}")
        
        # Save individual objects
        objects_dir = os.path.join(args.output, 'objects')
        scene_graph.save_objects_to_json(objects_dir)
        logger.info(f"Saved objects to {objects_dir}")
        
        # Save drawers if present
        if not args.no_drawers:
            drawers_dir = os.path.join(args.output, 'drawers')
            scene_graph.save_drawers_to_json(drawers_dir)
            logger.info(f"Saved drawers to {drawers_dir}")
        
        logger.info(f"\n[SUCCESS] Scene graph generation complete!")
        logger.info(f"  Nodes: {len(scene_graph.nodes)}")
        logger.info(f"  Connections: {len(scene_graph.outgoing)}")
        
    except Exception as e:
        logger.error(f"Failed to save scene graph: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1
    
    # Generate Interactive HTML Visualization
    try:
        from utils_source.vis import visualize_scene_graph_interactive
        html_path = os.path.join(args.output, "scene_graph_interactive.html")
        visualize_scene_graph_interactive(scene_graph, html_path)
    except Exception as e:
        import traceback
        traceback.print_exc()
        logger.warning(f"Interactive visualization failed: {e}")

    # Optional visualization (GUI)
    if args.visualize:
        try:
            logger.info(f"\nLaunching visualization...")
            scene_graph.visualize(labels=True, connections=True, centroids=True)
        except Exception as e:
            logger.warning(f"Visualization failed: {e}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
