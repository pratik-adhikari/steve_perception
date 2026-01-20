"""
Mask3D adapter for unified pipeline.
Wrapper around existing Mask3D interface.
"""
import sys
import os
import numpy as np
from typing import Dict, Any, List

from utils_source.mask3d_interface import run_mask3d

from models.adapters.base_adapter import BaseSegmentationAdapter, SegmentationResult
from utils_source.gpu_monitor import GPUMonitor, GPUEstimator


class Mask3DAdapter(BaseSegmentationAdapter):
    """Adapter for Mask3D model."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
    def get_model_name(self) -> str:
        return "mask3d"
    
    def predict(self, scene_path: str, vocabulary: List[str], output_dir: str) -> SegmentationResult:
        """
        Run Mask3D segmentation.
        
        Args:
            scene_path: Path to scene directory with mesh.ply
            vocabulary: List of class names (note: Mask3D is not open-vocabulary)
            output_dir: Output directory path
            
        Returns:
            SegmentationResult with standardized outputs
        """
        os.makedirs(output_dir, exist_ok=True)
        
        voxel_size = self.config.get('mask3d', {}).get('voxel_size', 0.02)
        conf_thresh = self.config.get('inference', {}).get('conf_threshold', 0.55)
        
        print(f"[Mask3D] Running with voxel_size={voxel_size}, conf={conf_thresh}")
        
        run_mask3d(
            scene_dir=scene_path,
            device='cuda:0',
            voxel_size=voxel_size,
            threshold=conf_thresh,
            no_rotation=True
        )
        
        mask3d_output_dir = os.path.join(scene_path, 'mask3d_output')
        pred_file = os.path.join(mask3d_output_dir, 'predictions.txt')
        
        import open3d as o3d
        mesh_path = os.path.join(scene_path, 'mesh.ply')
        pcd = o3d.io.read_point_cloud(mesh_path)
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors) if pcd.has_colors() else None
        
        masks_list = []
        classes_list = []
        scores_list = []
        
        with open(pred_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                    mask_file = os.path.join(mask3d_output_dir, parts[0])
                    class_id = int(parts[1])
                    score = float(parts[2])
                    
                    mask = np.loadtxt(mask_file, dtype=bool)
                    
                    masks_list.append(mask)
                    classes_list.append(class_id)
                    scores_list.append(score)
        
        if len(masks_list) == 0:
            print("[Warning] No masks found in Mask3D output")
            num_points = len(points)
            masks_np = np.zeros((num_points, 0), dtype=bool)
            classes_np = np.array([], dtype=int)
            scores_np = np.array([], dtype=float)
        else:
            masks_np = np.column_stack(masks_list)
            classes_np = np.array(classes_list)
            scores_np = np.array(scores_list)
        
        return SegmentationResult(
            masks=masks_np,
            classes=classes_np,
            scores=scores_np,
            points=points,
            colors=colors,
            mesh_path=mesh_path,
            metadata={
                'model': 'mask3d',
                'voxel_size': voxel_size,
                'conf_threshold': conf_thresh
            }
        )
