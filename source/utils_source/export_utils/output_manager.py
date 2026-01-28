"""
Output manager for handling multiple segmentation output formats.
Centralizes all file I/O operations for cleaner code organization.
"""
import os
import json
import numpy as np
import pandas as pd
import logging
from typing import List, Dict, Any
from models.adapters.base_adapter import SegmentationResult


class OutputManager:
    """Manages all output generation for segmentation results."""
    
    def __init__(self, output_dir: str, config: Dict[str, Any]):
        """
        Initialize output manager.
        
        Args:
            output_dir: Base directory for all outputs
            config: Configuration dictionary
        """
        self.output_dir = output_dir
        self.config = config
        self.logger = logging.getLogger(__name__)
        os.makedirs(output_dir, exist_ok=True)
    
    def save_all(self, result: SegmentationResult, vocabulary: List[str]):
        """
        Save all configured output formats.
        
        Args:
            result: Segmentation result object
            vocabulary: List of class names
        """
        # Save label mapping (always needed for SceneGraph)
        self._save_label_mapping(vocabulary)
        
        # Save optional formats based on config
        if self.config['output'].get('save_scenegraph_format', True):
            self._save_scenegraph_format(result)
            
        if self.config['output'].get('save_detailed_objects', True):
            self._save_detailed_objects(result, vocabulary)
            
        # Always save visualization if requested
        if self.config['output'].get('save_visualization', True):
            self._save_colored_visualization(result)

    def _save_colored_visualization(self, result: SegmentationResult):
        """Save a colored visualization (mesh or point cloud) of the segmentation."""
        import open3d as o3d
        import matplotlib.pyplot as plt
        try:
            from scipy.spatial import cKDTree
        except ImportError:
            cKDTree = None
            self.logger.warning("scipy not found, fallback to slow point cloud visualization")
        
        output_path = os.path.join(self.output_dir, "mesh_labeled.ply")
        
        # 1. Determine High-Quality Mesh Path
        # Expecting: output_dir=.../openyolo3d_output -> parent -> export/visualization/raw_mesh.ply
        # We use raw_mesh.ply because it is guaranteed to have geometry (faces), whereas mesh.ply might be a dense cloud
        parent_dir = os.path.dirname(self.output_dir.rstrip('/'))
        mesh_path = os.path.join(parent_dir, "export", "visualization", "raw_mesh.ply")
        
        has_mesh = os.path.exists(mesh_path)
        if has_mesh:
            self.logger.info(f"Found high-quality mesh: {mesh_path}")
        else:
            self.logger.warning(f"High-quality mesh not found at {mesh_path}, falling back to point cloud")
            
        # 2. Prepare Colors for Segmentation Points
        if result.points is None:
            return

        # Default to white for unsegmented/background
        point_colors = np.ones((len(result.points), 3)) 
        
        # Mark unsegmented points with a sentinel if we want strict white background 
        # (Already initialized to white (1,1,1))
        
        # Get colormap
        cmap = plt.get_cmap("tab20")
        
        # Overlay masks
        # Sort by score so high confidence overwrites low confidence
        sorted_indices = np.argsort(result.scores)
        conf_thresh = self.config['inference'].get('conf_threshold', 0.1)
        
        # Keep track of which points are actually segmented
        segmented_mask = np.zeros(len(result.points), dtype=bool)
        
        for idx in sorted_indices:
            if result.scores[idx] < conf_thresh:
                continue
                
            mask = result.masks[:, idx].astype(bool)
            color = cmap(idx % 20)[:3]
            point_colors[mask] = color
            segmented_mask[mask] = True
            
        # 3. Transfer to Mesh (if available) or Save Point Cloud
        if has_mesh and cKDTree is not None:
            try:
                mesh = o3d.io.read_triangle_mesh(mesh_path)
                vertices = np.asarray(mesh.vertices)
                
                self.logger.info(f"Building KDTree for {len(result.points)} points...")
                tree = cKDTree(result.points)
                
                self.logger.info(f"Querying nearest neighbors for {len(vertices)} vertices...")
                # Query nearest neighbor for all vertices
                dists, indices = tree.query(vertices, k=1, workers=-1)
                
                # Transfer colors
                # Initialize mesh colors to White
                mesh_colors = np.ones((len(vertices), 3))
                
                # Check distance threshold? 
                # If the mesh vertex is too far from any point, keep it white.
                # Assuming typical voxel sizes, 0.1m is a safe upper bound for "too far".
                valid_mask = dists < 0.1 
                
                # Only transfer color if the nearest neighbor was actually segmented
                # indices gives us the index in result.points
                neighbor_segmented = segmented_mask[indices]
                
                # Combine: Valid neighbor AND neighbor is segmented
                transfer_mask = valid_mask & neighbor_segmented
                
                # Apply colors where applicable
                mesh_colors[transfer_mask] = point_colors[indices[transfer_mask]]
                
                mesh.vertex_colors = o3d.utility.Vector3dVector(mesh_colors)
                o3d.io.write_triangle_mesh(output_path, mesh)
                self.logger.info(f"Saved high-quality labeled mesh to {output_path}")
                return
                
            except Exception as e:
                self.logger.error(f"Failed to process high-quality mesh: {e}")
                self.logger.info("Falling back to point cloud dump")
        
        # Fallback: Save the labeled point cloud directly
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(result.points)
        pcd.colors = o3d.utility.Vector3dVector(point_colors)
        
        o3d.io.write_point_cloud(output_path, pcd)
        self.logger.info(f"Saved colored visualization (point cloud) to {output_path}")
    
    def _save_label_mapping(self, vocabulary: List[str]):
        """Save vocabulary to CSV for SceneGraph builder."""
        csv_path = os.path.join(self.output_dir, 'mask3d_label_mapping.csv')
        
        df = pd.DataFrame({
            'id': range(len(vocabulary)),
            'category': vocabulary
        })
        
        df.to_csv(csv_path, index=False)
        self.logger.info(f"Saved label mapping to {csv_path}")
    
    def _save_scenegraph_format(self, result: SegmentationResult):
        """Save results in SceneGraph-compatible format."""
        import shutil
        
        mask3d_dir = os.path.join(self.output_dir, 'mask3d_output')
        masks_dir = os.path.join(mask3d_dir, 'masks')
        os.makedirs(masks_dir, exist_ok=True)
        
        # Copy mesh files
        mesh_dst = os.path.join(self.output_dir, 'mesh.ply')
        mesh_labeled_dst = os.path.join(mask3d_dir, 'mesh_labeled.ply')
        
        if result.mesh_path and os.path.exists(result.mesh_path):
            if not os.path.exists(mesh_dst):
                shutil.copy2(result.mesh_path, mesh_dst)
            if not os.path.exists(mesh_labeled_dst):
                shutil.copy2(result.mesh_path, mesh_labeled_dst)
        
        # Save masks and create predictions.txt
        conf_thresh = self.config['inference'].get('conf_threshold', 0.1)
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
        
        self.logger.info(f"Saved {len(lines)} masks for SceneGraph to {mask3d_dir}")
    
    def _save_detailed_objects(self, result: SegmentationResult, vocabulary: List[str]):
        """Save individual object point clouds and metadata."""
        import open3d as o3d
        
        objects_dir = os.path.join(self.output_dir, 'objects')
        os.makedirs(objects_dir, exist_ok=True)
        
        conf_thresh = self.config['inference'].get('conf_threshold', 0.1)
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
            
            # Compute object properties
            centroid = obj_points.mean(axis=0).tolist()
            min_bb = obj_points.min(axis=0)
            max_bb = obj_points.max(axis=0)
            dimensions = (max_bb - min_bb).tolist()
            
            # Create point cloud
            obj_pcd = o3d.geometry.PointCloud()
            obj_pcd.points = o3d.utility.Vector3dVector(obj_points)
            if result.colors is not None:
                obj_pcd.colors = o3d.utility.Vector3dVector(result.colors[mask])
            
            # Save PLY file
            ply_path = os.path.join(objects_dir, f"{i}_{class_name.replace(' ', '_')}.ply")
            o3d.io.write_point_cloud(ply_path, obj_pcd)
            
            # Store metadata
            metadata[str(i)] = {
                "class_id": class_idx,
                "class_name": class_name,
                "confidence": float(result.scores[i]),
                "centroid": centroid,
                "dimensions": dimensions,
                "num_points": int(mask.sum())
            }
        
        # Save metadata JSON
        json_path = os.path.join(objects_dir, 'predictions.json')
        with open(json_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        
        self.logger.info(f"Saved {len(metadata)} objects to {objects_dir}")
