"""
export_utils.py

Export utilities for OpenYOLO3D wrapper.
Contains functions to save detection results in various formats.
"""
import os
import json
import numpy as np
import open3d as o3d
import shutil
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../..'))
from utils.vocabulary import load_vocabulary


def save_detailed_results(model, prediction, output_path: str, text_prompts: list, conf_thresh: float = 0.1):
    """
    Save individual object point clouds and metadata JSON.
    
    # ---------------------------------------------------------------------------------------------------------
    # WHY: The default OpenYOLO3D output is a single PLY. For analysis, we want:
    # - Separate PLY per detected object (for debugging, visualization)
    # - JSON with class names, confidence scores, centroids, bounding boxes
    # ---------------------------------------------------------------------------------------------------------
    
    :param model: OpenYolo3D instance (has world2cam.mesh, etc.)
    :param prediction: Dict {scene: (masks, classes, scores)} or Tuple (masks, classes, scores)
    :param output_path: Base output path (e.g., output/result.ply)
    :param text_prompts: List of class names
    :param conf_thresh: Minimum confidence to save
    """
    # Handle prediction as dict or tuple
    if isinstance(prediction, dict):
        # OpenYOLO3D returns {scene_name: (masks, classes, scores)}
        prediction = list(prediction.values())[0]
    masks, classes, scores = prediction
    output_dir = os.path.dirname(output_path)
    detailed_dir = os.path.join(output_dir, "objects")
    os.makedirs(detailed_dir, exist_ok=True)
    
    # Load mesh points
    mesh_path = model.world2cam.mesh
    pcd = o3d.io.read_point_cloud(mesh_path)
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors) if pcd.has_colors() else None
    
    metadata = {}
    num_instances = classes.shape[0]
    
    for i in range(num_instances):
        score = float(scores[i].item())
        if score < conf_thresh:
            continue
        
        class_idx = int(classes[i].item())
        class_name = text_prompts[class_idx] if class_idx < len(text_prompts) else f"class_{class_idx}"
        mask = masks[:, i].bool().cpu().numpy()
        
        obj_points = points[mask]
        if len(obj_points) < 10:
            continue
        
        # Compute geometry
        centroid = obj_points.mean(axis=0).tolist()
        min_bb = obj_points.min(axis=0)
        max_bb = obj_points.max(axis=0)
        dimensions = (max_bb - min_bb).tolist()
        
        # Save PLY
        obj_pcd = o3d.geometry.PointCloud()
        obj_pcd.points = o3d.utility.Vector3dVector(obj_points)
        if colors is not None:
            obj_pcd.colors = o3d.utility.Vector3dVector(colors[mask])
        
        ply_path = os.path.join(detailed_dir, f"{i}_{class_name.replace(' ', '_')}.ply")
        o3d.io.write_point_cloud(ply_path, obj_pcd)
        
        metadata[str(i)] = {
            "class_id": class_idx,
            "class_name": class_name,
            "confidence": score,
            "centroid": centroid,
            "dimensions": dimensions,
            "num_points": int(mask.sum())
        }
    
    # Save JSON
    json_path = os.path.join(detailed_dir, "predictions.json")
    with open(json_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    
    print(f"[Export] Saved {len(metadata)} objects to {detailed_dir}")


def save_scenegraph_structure(model, prediction, output_path: str, conf_thresh: float = 0.1):
    """
    Save output in format expected by SceneGraph.build().
    
    # ---------------------------------------------------------------------------------------------------------
    # WHY: The SceneGraph class (scene_graph.py) expects a specific directory structure:
    #   scan_dir/
    #     mesh.ply                    <- Copy of input mesh
    #     mask3d_output/
    #       predictions.txt           <- "masks/0.txt class_id confidence" per line
    #       mesh_labeled.ply          <- Copy of mesh (for compatibility)
    #       masks/
    #         0.txt, 1.txt, ...       <- Binary mask (0/1 per point)
    #
    # This was reverse-engineered from SceneGraph.build() which reads:
    #   - predictions.txt to get mask paths, class IDs, and confidence scores
    #   - masks/*.txt to get per-point binary masks
    #   - mesh_labeled.ply to get point coordinates
    # ---------------------------------------------------------------------------------------------------------
    
    :param model: OpenYolo3D instance
    :param prediction: Dict {scene: (masks, classes, scores)} or Tuple (masks, classes, scores)
    :param output_path: Base output path
    :param conf_thresh: Minimum confidence to save
    """
    # Handle prediction as dict or tuple
    if isinstance(prediction, dict):
        # OpenYOLO3D returns {scene_name: (masks, classes, scores)}
        prediction = list(prediction.values())[0]
    masks, classes, scores = prediction
    output_dir = os.path.dirname(output_path)
    
    mask3d_dir = os.path.join(output_dir, "mask3d_output")
    masks_dir = os.path.join(mask3d_dir, "masks")
    os.makedirs(masks_dir, exist_ok=True)
    
    # Copy mesh
    src_mesh = model.world2cam.mesh
    dst_mesh = os.path.join(output_dir, "mesh.ply")
    dst_labeled = os.path.join(mask3d_dir, "mesh_labeled.ply")
    
    if not os.path.exists(dst_mesh):
        shutil.copy2(src_mesh, dst_mesh)
    if not os.path.exists(dst_labeled):
        shutil.copy2(src_mesh, dst_labeled)
    
    # Write masks and predictions.txt
    lines = []
    num_instances = classes.shape[0]
    count = 0
    
    for i in range(num_instances):
        score = float(scores[i].item())
        if score < conf_thresh:
            continue
        
        mask = masks[:, i].bool().cpu().numpy().astype(int)
        mask_filename = f"{i}.txt"
        mask_path = os.path.join(masks_dir, mask_filename)
        np.savetxt(mask_path, mask, fmt='%d')
        
        class_idx = int(classes[i].item())
        rel_path = os.path.join("masks", mask_filename)
        lines.append(f"{rel_path} {class_idx} {score:.4f}")
        count += 1
    
    pred_path = os.path.join(mask3d_dir, "predictions.txt")
    with open(pred_path, 'w') as f:
        f.write("\n".join(lines))
    
    print(f"[Export] Saved {count} masks for SceneGraph to {mask3d_dir}")
