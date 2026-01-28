"""
OpenYOLO3D adapter for unified pipeline.
Direct integration without wrapper layer for simplicity.
"""
import sys
import os
import numpy as np
import torch
import logging
from typing import Dict, Any, List

# Ensure Mask3D submodule is in path
mask3d_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../lib/OpenYOLO3D/models/Mask3D'))
if mask3d_path not in sys.path:
    sys.path.insert(0, mask3d_path)

# Add OpenYOLO3D library path
openyolo3d_lib_path = os.path.join(os.path.dirname(__file__), '../lib/OpenYOLO3D')
if os.path.exists(openyolo3d_lib_path) and openyolo3d_lib_path not in sys.path:
    sys.path.insert(0, openyolo3d_lib_path)

# Import directly from OpenYOLO3D library
from utils import OpenYolo3D

from models.adapters.base_adapter import BaseSegmentationAdapter, SegmentationResult
from utils_source.gpu_monitor import GPUMonitor, GPUEstimator
from utils_source.export_utils.logging_decorators import log_method_call


class OpenYolo3DAdapter(BaseSegmentationAdapter):
    """Adapter for OpenYOLO3D model with integrated interface."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model = None
        self.logger = logging.getLogger(__name__)
        
    def get_model_name(self) -> str:
        return "openyolo3d"
    
    @log_method_call
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
        
        self.logger.info(f"[OpenYOLO3D] Initializing with frame_step={frame_step}, conf={conf_thresh}")
        self.logger.info(f"[OpenYOLO3D] Config path: {config_path}")
        self.logger.info(f"[OpenYOLO3D] Vocabulary: {len(vocabulary)} classes")
        self.logger.info(f"[OpenYOLO3D] Depth scale: {depth_scale}")
        self.logger.info(f"[OpenYOLO3D] Scene path: {scene_path}")
        
        # Initialize model (lazy initialization)
        if self.model is None:
            self.model = OpenYolo3D(config_path)
            self.logger.info(f"[OpenYOLO3D] Model initialized")
        
        # Update frequency/frame_step in model config dynamically
        if hasattr(self.model, 'openyolo3d_config'):
            self.model.openyolo3d_config['openyolo3d']['frequency'] = frame_step
            self.logger.info(f"[OpenYOLO3D] Set frame_step/frequency = {frame_step}")
        
        self.logger.info(f"[OpenYOLO3D] Running prediction with {len(vocabulary)} text prompts")
        
        # Run prediction (merged from interface.py)
        prediction = self.model.predict(
            path_2_scene_data=scene_path,
            depth_scale=depth_scale,
            text=vocabulary,
            datatype="point cloud"
        )
        
        # Extract results from library return format
        scene_name = list(prediction.keys())[0]
        masks, classes, scores = prediction[scene_name]
        
        self.logger.info(f"[OpenYOLO3D] Prediction complete for scene: {scene_name}")
        self.logger.info(f"[OpenYOLO3D]   → Masks shape: {masks.shape}")
        self.logger.info(f"[OpenYOLO3D]   → Classes shape: {classes.shape}, unique: {classes.unique().tolist() if len(classes) > 0 else []}")
        
        # Handle empty results gracefully
        if len(scores) > 0:
            self.logger.info(f"[OpenYOLO3D]   → Scores shape: {scores.shape}, range: [{scores.min():.4f}, {scores.max():.4f}]")
            self.logger.info(f"[OpenYOLO3D]   → Total instances: {len(scores)}")
            
            # Log score distribution
            high_conf = (scores > 0.5).sum().item()
            med_conf = ((scores > 0.1) & (scores <= 0.5)).sum().item()
            low_conf = (scores <= 0.1).sum().item()
            self.logger.info(f"[OpenYOLO3D]   → Score distribution: >0.5: {high_conf}, 0.1-0.5: {med_conf}, <0.1: {low_conf}")
        else:
            self.logger.warning(f"[OpenYOLO3D]   → NO INSTANCES DETECTED! Scores shape: {scores.shape}")
            self.logger.warning(f"[OpenYOLO3D]   → This likely means either:")
            self.logger.warning(f"[OpenYOLO3D]        1. No 3D mask proposals were generated (Mask3D step failed)")
            self.logger.warning(f"[OpenYOLO3D]        2. No 2D bounding boxes were detected (YOLO-World step failed)")
            self.logger.warning(f"[OpenYOLO3D]        3. The 3D-2D matching process filtered out all instances")
        
        # Load point cloud data
        import open3d as o3d
        mesh_path = os.path.join(scene_path, 'scene.ply')
        if not os.path.exists(mesh_path):
            # Fallback to mesh.ply
            mesh_path = os.path.join(scene_path, 'mesh.ply')
            
        points = None
        colors = None
        
        if os.path.exists(mesh_path):
            pcd = o3d.io.read_point_cloud(mesh_path)
            points = np.asarray(pcd.points)
            colors = np.asarray(pcd.colors) if pcd.has_colors() else None
        
        # Convert to numpy
        masks_np = masks.cpu().numpy()
        classes_np = classes.cpu().numpy()
        scores_np = scores.cpu().numpy()
        
        self.logger.info(f"[OpenYOLO3D] Returning result:")
        self.logger.info(f"[OpenYOLO3D]   → masks: {masks_np.shape}, dtype: {masks_np.dtype}")
        self.logger.info(f"[OpenYOLO3D]   → classes: {classes_np.shape}, dtype: {classes_np.dtype}")
        self.logger.info(f"[OpenYOLO3D]   → scores: {scores_np.shape}, dtype: {scores_np.dtype}")
        self.logger.info(f"[OpenYOLO3D]   → points: {points.shape if points is not None else None}")
        self.logger.info(f"[OpenYOLO3D]   → Total instances: {len(scores_np)}")
        
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
