"""
Mask3D adapter for unified pipeline.
Implements specific logic for loading and running Mask3D directly.
"""
import sys
import os
import numpy as np
import torch
import logging
from pathlib import Path
from typing import Dict, Any, List, Union

from models.adapters.base_adapter import BaseSegmentationAdapter, SegmentationResult

# Add Mask3D library to python path
current_dir = os.path.dirname(os.path.abspath(__file__))
# source/models/adapters -> source/models/lib/Mask3D
mask3d_root = os.path.abspath(os.path.join(current_dir, '../lib/Mask3D'))
if mask3d_root not in sys.path:
    sys.path.insert(0, mask3d_root)

# Imports that depend on Mask3D path
try:
    import MinkowskiEngine as ME
    from mask3d import InstanceSegmentation
    import mask3d.conf
    from hydra.experimental import compose, initialize
    from mask3d.utils.utils import (
        load_backbone_checkpoint_with_missing_or_exsessive_keys,
        load_checkpoint_with_missing_or_exsessive_keys,
    )
    import open3d as o3d
    import albumentations as A
    from torch.nn.functional import softmax
except ImportError as e:
    print(f"[Mask3DAdapter] Error importing dependencies: {e}")
    # We allow the class to be defined, but predict will fail


class Mask3DAdapter(BaseSegmentationAdapter):
    """Adapter for Mask3D model."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.logger = logging.getLogger(__name__)
        
    def get_model_name(self) -> str:
        return "mask3d"
        
    def _get_mask_and_scores(
        self,
        mask_cls: torch.Tensor,
        mask_pred: torch.Tensor,
        topk_per_image: int,
        num_queries: int,
        num_classes: int,
        device: Union[str, torch.device],
    ):
        """
        Extract masks and scores from raw model output.
        Verified logic from mask3d.py
        """
        labels = torch.arange(
            num_classes,
            device=device,
        ).unsqueeze(0).repeat(num_queries, 1).flatten(0, 1)

        if topk_per_image != -1:
            scores_per_query, topk_indices = mask_cls.flatten(0, 1).topk(
                topk_per_image,
                sorted=True,
            )
        else:
            scores_per_query, topk_indices = mask_cls.flatten(0, 1).topk(
                num_queries,
                sorted=True,
            )

        labels_per_query = labels[topk_indices]
        topk_indices = topk_indices // num_classes
        mask_pred = mask_pred[:, topk_indices]

        result_pred_mask = (mask_pred > 0).float()
        heatmap = mask_pred.float().sigmoid()

        mask_scores_per_image = (heatmap * result_pred_mask).sum(0) / (
            result_pred_mask.sum(0) + 1e-6)
        score = scores_per_query * mask_scores_per_image
        classes = labels_per_query

        return score, result_pred_mask, classes, heatmap

    def predict(self, scene_path: str, vocabulary: List[str], output_dir: str) -> SegmentationResult:
        """
        Run Mask3D segmentation by wrapping the official mask3d.py script.
        """
        import shutil
        import subprocess
        
        os.makedirs(output_dir, exist_ok=True)
        self.logger.info(f"[Mask3D] Output directory: {output_dir}")

        # 1. Prepare Input for mask3d.py
        # mask3d.py expects 'mesh.ply' or 'textured_output.obj' in the workspace dir
        # We will usage 'output_dir' as the workspace
        
        target_mesh_path = os.path.join(output_dir, 'mesh.ply')
        
        # Determine source
        source_mesh_path = os.path.join(scene_path, 'mesh.ply')
        if not os.path.exists(source_mesh_path):
             for alt_name in ['scene.ply', 'cloud.ply', 'textured_output.obj']:
                 alt_path = os.path.join(scene_path, alt_name)
                 if os.path.exists(alt_path):
                     source_mesh_path = alt_path
                     break
        
        if not os.path.exists(source_mesh_path):
            raise FileNotFoundError(f"Could not find valid point cloud/mesh in {scene_path}")
            
        self.logger.info(f"[Mask3D] Copying {source_mesh_path} -> {target_mesh_path}")
        if source_mesh_path != target_mesh_path:
            shutil.copy2(source_mesh_path, target_mesh_path)
            
        # 2. Run mask3d.py via subprocess
        # Script path: source/models/lib/Mask3D/mask3d.py
        script_path = os.path.join(mask3d_root, 'mask3d.py')
        
        # Checkpoint (handled by script defaults or config)
        # We'll trust the script's internal config loading but ensure pythonpath is right
        
        cmd = [
            sys.executable,
            script_path,
            '--workspace', output_dir,
            '--pcd' # Force using point cloud mode since we provided mesh.ply (likely from scene.ply)
        ]
        
        self.logger.info(f"[Mask3D] Running command: {' '.join(cmd)}")
        
        env = os.environ.copy()
        env['PYTHONPATH'] = f"{mask3d_root}:{env.get('PYTHONPATH', '')}"
        
        try:
            result = subprocess.run(cmd, env=env, capture_output=True, text=True, check=True)
            self.logger.info(result.stdout)
        except subprocess.CalledProcessError as e:
            self.logger.error(f"[Mask3D] Execution failed with code {e.returncode}")
            self.logger.error("[Mask3D] STDOUT:\n" + e.stdout)
            self.logger.error("[Mask3D] STDERR:\n" + e.stderr)
            raise RuntimeError("Mask3D script execution failed")

        # 3. Parse Outputs
        # The script generates:
        # - predictions.txt
        # - pred_mask/*.txt
        # - mesh_labeled.ply (We can use this to get the points/colors exactly as the model saw/saved them)
        
        predictions_file = os.path.join(output_dir, 'predictions.txt')
        if not os.path.exists(predictions_file):
             raise RuntimeError("Mask3D did not generate predictions.txt")
             
        # Load geometry for result (needed for both success and empty cases)
        try:
             # Prefer labeled mesh if available to get colors/points consistent with mask
             labeled_mesh_path = os.path.join(output_dir, 'mesh_labeled.ply')
             geom_path = labeled_mesh_path if os.path.exists(labeled_mesh_path) else target_mesh_path
             
             pcd = o3d.io.read_point_cloud(geom_path)
             points = np.asarray(pcd.points)
             colors = np.asarray(pcd.colors)
        except Exception as e:
             self.logger.warning(f"Failed to load result geometry: {e}")
             points = np.array([])
             colors = np.array([])

        # Read predictions
        # Format: relative_path label confidence
        masks = []
        classes = []
        scores = []
        
        # [Revert] Use input mesh path directly
        mesh_path = target_mesh_path
        
        with open(predictions_file, 'r') as f:
            lines = f.readlines()
            
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 3:
                mask_rel_path = parts[0]
                label_id = int(parts[1])
                score = float(parts[2])
                
                mask_abs_path = os.path.join(output_dir, mask_rel_path)
                if os.path.exists(mask_abs_path):
                    # Load mask txt (0/1 lines)
                    mask_data = np.loadtxt(mask_abs_path, dtype=int)
                    masks.append(mask_data)
                    classes.append(label_id)
                    scores.append(score)
        
        if not masks:
            self.logger.warning("No valid masks loaded")
            return SegmentationResult(
                masks=np.array([]),
                classes=np.array([]),
                scores=np.array([]),
                points=points,
                colors=colors,
                mesh_path=mesh_path
            )

        masks_np = np.column_stack(masks)
        classes_np = np.array(classes)
        scores_np = np.array(scores)
        
        return SegmentationResult(
            masks=masks_np,
            classes=classes_np,
            scores=scores_np,
            points=points,
            colors=colors,
            mesh_path=mesh_path,
            metadata={
                'model': 'mask3d_script',
            }
        )
