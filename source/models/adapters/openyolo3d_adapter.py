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
        
        # [Preprocess] Externalize coordinate handling as requested
        import open3d as o3d
        import tempfile
        
        # 1. Load Data (High Res if requested)
        preprocessing_config = self.config.get('preprocessing', {})
        load_high_res = preprocessing_config.get('load_high_res', False)
        
        # Default to scene.ply
        mesh_path = os.path.join(scene_path, 'scene.ply')
        
        # Helper to safely load and check colors
        def load_and_validate(path):
            if not os.path.exists(path): return None
            try:
                if path.endswith('.obj'):
                    m = o3d.io.read_triangle_mesh(path, enable_post_processing=True)
                    if m.has_vertices():
                         # If mesh has textures but no vertex colors, this might fail to capture color.
                         # Open3D read_triangle_mesh doesn't autosample texture to vertex color.
                         if not m.has_vertex_colors() and not m.has_textures():
                             return None # No color source
                         if m.has_vertex_colors():
                             p = o3d.geometry.PointCloud()
                             p.points = m.vertices
                             p.colors = m.vertex_colors
                             return p
                         # If texture, we'd need complex sampling. Skip for now.
                         return None
                
                # Try as point cloud (PLY)
                p = o3d.io.read_point_cloud(path)
                if p.has_points() and p.has_colors():
                    return p
            except:
                pass
            return None

        pcd = None
        if load_high_res:
             candidates = [
                os.path.join(scene_path, "textured_output.obj"),
                os.path.join(scene_path, "visualization/mesh.ply"),
                os.path.join(scene_path, "visualization/raw_mesh.ply")
             ]
             for cand in candidates:
                 pcd_cand = load_and_validate(cand)
                 if pcd_cand is not None:
                     self.logger.info(f"[OpenYOLO3D-Adapter] Found High-Res Source with Colors: {cand}")
                     mesh_path = cand
                     pcd = pcd_cand
                     break
                 else:
                     if os.path.exists(cand):
                         self.logger.warning(f"[OpenYOLO3D-Adapter] Candidate {cand} exists but has no readable colors. Skipping.")
        
        # Fallback to scene.ply if high-res failed or not requested
        if pcd is None:
             self.logger.info(f"[OpenYOLO3D-Adapter] Loading baseline {mesh_path}...")
             pcd = o3d.io.read_point_cloud(mesh_path)
        
        if not pcd.has_points():
             self.logger.error(f"[OpenYOLO3D-Adapter] Failed to load any point cloud from {mesh_path}!")
             return SegmentationResult(masks=np.array([]), classes=np.array([]), scores=np.array([]), points=None, colors=None, mesh_path=mesh_path)

        original_points = np.asarray(pcd.points)
        original_colors = np.asarray(pcd.colors) if pcd.has_colors() else np.ones_like(original_points) * 0.5
        
        if not pcd.has_colors():
            # This is critical for OpenYOLO3D
            self.logger.warning("[OpenYOLO3D-Adapter] WARNING: Input has no colors! Model performance will be degraded.")
            
        # Determine effective voxel size scaling
        target_voxel_size = float(preprocessing_config.get('target_voxel_size', 0.05))
        model_voxel_size = 0.02
        scale_factor = model_voxel_size / target_voxel_size
        
        # Prepare processed copy
        pcd_proc = o3d.geometry.PointCloud()
        pcd_proc.points = o3d.utility.Vector3dVector(original_points.copy())
        if pcd.has_colors():
            pcd_proc.colors = o3d.utility.Vector3dVector(original_colors.copy())
        else:
            # Assign gray if missing (silent fail handled by warning above)
            pcd_proc.colors = o3d.utility.Vector3dVector(original_colors.copy())
        
        # 2. Apply Transformations
        # Center
        if preprocessing_config.get('center_coordinates', True):
            mean = np.mean(original_points, axis=0)
            self.logger.info(f"[OpenYOLO3D-Adapter] Centering coordinates (Mean: {mean})")
            pcd_proc.translate(-mean)
            
        # Rotate Z-up to Model Space (if needed)
        # Assuming model expects Y-up or different convention. 
        # Standard Mask3D transform: Rx(90) * Rz(270)
        if preprocessing_config.get('rotate_z_up', True):
            self.logger.info(f"[OpenYOLO3D-Adapter] Rotating Z-up to Model Space")
            # R_x_90 (Z->Y, Y->-Z)
            R_x = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
            pcd_proc.rotate(R_x, center=(0,0,0))
            
            # R_z_270 (X->Y, Y->-X)
            R_z = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
            pcd_proc.rotate(R_z, center=(0,0,0))
            
        # Scale for Voxel Size
        if abs(scale_factor - 1.0) > 1e-6:
            self.logger.info(f"[OpenYOLO3D-Adapter] Scaling geometry by {scale_factor:.4f} (Target Voxel: {target_voxel_size})")
            pcd_proc.scale(scale_factor, center=(0,0,0))
            
        # 3. Save Temp File
        fd, temp_path = tempfile.mkstemp(suffix='.ply')
        os.close(fd)
        
        # We must save as PLY for OpenYOLO3D to load it
        o3d.io.write_point_cloud(temp_path, pcd_proc)
        self.logger.info(f"[OpenYOLO3D-Adapter] Saved preprocessed input to {temp_path}")
        
        try:
            prediction = self.model.predict(
                path_2_scene_data=scene_path,
                depth_scale=depth_scale,
                text=vocabulary,
                datatype="point cloud",
                processed_scene=temp_path 
            )
            
            # Extract results
            scene_name = list(prediction.keys())[0]
            masks, classes, scores = prediction[scene_name]
            
            self.logger.info(f"[OpenYOLO3D] Prediction complete for scene: {scene_name}")
            
            # Convert to numpy
            masks_np = masks.cpu().numpy()
            
            # Verification
            if masks_np.shape[0] != original_points.shape[0]:
                 self.logger.warning(f"[OpenYOLO3D-Adapter] Mismatch! Masks: {masks_np.shape[0]}, Orig: {original_points.shape[0]}")
                 # This should not happen if temp_path preserved order logic
                 # If it happens, we might need to rely on the points OpenYOLO3D used (loaded from temp)
                 # But we want original coordinates.
            else:
                 self.logger.info(f"[OpenYOLO3D-Adapter] Output aligned with original points.")

            classes_np = classes.cpu().numpy()
            scores_np = scores.cpu().numpy()
            
            return SegmentationResult(
                masks=masks_np,
                classes=classes_np,
                scores=scores_np,
                points=original_points, # Return ORIGINAL points for correct downstream visualization/export
                colors=original_colors,
                mesh_path=mesh_path,
                metadata={
                    'model': 'openyolo3d',
                    'vocabulary': vocabulary,
                    'frame_step': frame_step,
                    'conf_threshold': conf_thresh
                }
            )
            
        finally:
            # Cleanup
            if os.path.exists(temp_path):
                os.remove(temp_path)
                self.logger.info(f"[OpenYOLO3D-Adapter] Cleaned up temp file")
