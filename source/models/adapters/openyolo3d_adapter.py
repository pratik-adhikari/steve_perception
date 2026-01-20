"""
OpenYOLO3D adapter for unified pipeline.
Thin wrapper around existing OpenYOLO3D interface.
"""
import sys
import os
import numpy as np
import torch
from typing import Dict, Any, List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../wrappers/openyolo3d'))
from interface import OpenYolo3DInterface

from models.adapters.base_adapter import BaseSegmentationAdapter, SegmentationResult
from utils_source.gpu_monitor import GPUMonitor, GPUEstimator


class OpenYolo3DAdapter(BaseSegmentationAdapter):
    """Adapter for OpenYOLO3D model."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.interface = None
        
    def get_model_name(self) -> str:
        return "openyolo3d"
    
    def predict(self, scene_path: str, vocabulary: List[str], output_dir: str) -> SegmentationResult:
        """
        Run OpenYOLO3D segmentation.
        
        Args:
            scene_path: Path to scene directory with mesh.ply and images
            vocabulary: List of class names to detect
            output_dir: Output directory path
            
        Returns:
            SegmentationResult with standardized outputs
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Get config path - resolve relative to project root
        config_path = self.config.get('openyolo3d', {}).get('config_path', 'source/models/lib/OpenYOLO3D/pretrained/config.yaml')
        
        # If path is relative, resolve it from project root (2 levels up from this file's directory)
        if not os.path.isabs(config_path):
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
            config_path = os.path.join(project_root, config_path)
        
        frame_step = self.config.get('inference', {}).get('frame_step', 10)
        conf_thresh = self.config.get('inference', {}).get('conf_threshold', 0.1)
        depth_scale = self.config.get('inference', {}).get('depth_scale', 1000.0)
        
        print(f"[OpenYOLO3D] Initializing with frame_step={frame_step}, conf={conf_thresh}")
        print(f"[OpenYOLO3D] Config path: {config_path}")
        
        if self.interface is None:
            self.interface = OpenYolo3DInterface(config_path)
        
        prediction = self.interface.predict(
            scene_path=scene_path,
            text_prompts=vocabulary,
            depth_scale=depth_scale,
            frame_step=frame_step
        )
        
        if isinstance(prediction, dict):
            # Handle dictionary return (masks, classes, scores are keys)
            if 'masks' in prediction:
                masks = prediction['masks']
                classes = prediction['classes']
                scores = prediction['scores']
            else:
                # Fallback for {scene_name: tuple} structure if applicable
                prediction = next(iter(prediction.values()))
                masks, classes, scores = prediction
        else:
            masks, classes, scores = prediction
        
        import open3d as o3d
        mesh_path = os.path.join(scene_path, 'mesh.ply')
        if not os.path.exists(mesh_path):
            # Fallback to scene.ply (ScanNet standard / OpenYOLO3D input)
            mesh_path = os.path.join(scene_path, 'scene.ply')
            
        points = None
        colors = None
        
        if os.path.exists(mesh_path):
            pcd = o3d.io.read_point_cloud(mesh_path)
            points = np.asarray(pcd.points)
            colors = np.asarray(pcd.colors) if pcd.has_colors() else None
        
        # OpenYOLO3D returns (N, K), we keep it as (N, K)
        masks_np = masks.cpu().numpy()
        classes_np = classes.cpu().numpy()
        scores_np = scores.cpu().numpy()
        
        return SegmentationResult(
            masks=masks_np,
            classes=classes_np,
            scores=scores_np,
            points=points,
            colors=colors,
            mesh_path=mesh_path,
            metadata={
                'model': 'openyolo3d',
                'vocabulary': vocabulary,
                'frame_step': frame_step,
                'conf_threshold': conf_thresh
            }
        )
