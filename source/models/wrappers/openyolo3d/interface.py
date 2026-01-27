"""
OpenYOLO3D Interface Wrapper

Wraps the OpenYolo3D class from the OpenYOLO3D library for use in the unified pipeline.
"""
import sys
import os
# Paths
openyolo3d_lib_path = os.path.join(os.path.dirname(__file__), '../../lib/OpenYOLO3D')

# Add library root to path for its internal dependencies
if os.path.exists(openyolo3d_lib_path) and openyolo3d_lib_path not in sys.path:
    # Insert at 0 to ensure OpenYOLO3D's utils are found first if conflicts remain
    sys.path.insert(0, openyolo3d_lib_path)

# Import from OpenYOLO3D's utils module
from utils import OpenYolo3D


class OpenYolo3DInterface:
    """Interface to OpenYOLO3D model."""
    
    def __init__(self, config_path: str):
        """
        Initialize OpenYOLO3D interface.
        
        :param config_path: Path to OpenYOLO3D config.yaml
        """
        self.model = OpenYolo3D(config_path)
        self.config_path = config_path
    
    def predict(self, scene_path: str, text_prompts: list, depth_scale: float = 1000.0, frame_step: int = 1):
        """
        Run OpenYOLO3D prediction on a scene.
        
        :param scene_path: Path to scene data directory (with color/, depth/, poses/, etc.)
        :param text_prompts: List of text prompts for open-vocabulary detection
        :param depth_scale: Depth scaling factor (default: 1000.0 for typical depth sensors)
        :return: Prediction dictionary with masks, classes, and scores
        """
        import logging
        logger = logging.getLogger(__name__)
        
        # Update frequency/frame_step in the model config dynamically
        if self.model and hasattr(self.model, 'openyolo3d_config'):
            self.model.openyolo3d_config['openyolo3d']['frequency'] = frame_step
            logger.info(f"[OpenYOLO3D Interface] Set frame_step/frequency = {frame_step}")
        
        logger.info(f"[OpenYOLO3D Interface] Running prediction with {len(text_prompts)} text prompts")
        logger.info(f"[OpenYOLO3D Interface] Depth scale: {depth_scale}")

        prediction = self.model.predict(
            path_2_scene_data=scene_path,
            depth_scale=depth_scale,
            text=text_prompts,
            datatype="point cloud"
        )
        
        # prediction is a dict: {scene_name: (masks, classes, scores)}
        # Extract the first (and only) scene result
        scene_name = list(prediction.keys())[0]
        masks, classes, scores = prediction[scene_name]
        
        logger.info(f"[OpenYOLO3D Interface] Prediction complete for scene: {scene_name}")
        logger.info(f"[OpenYOLO3D Interface]   → Masks shape: {masks.shape}")
        logger.info(f"[OpenYOLO3D Interface]   → Classes shape: {classes.shape}, unique: {classes.unique().tolist() if len(classes) > 0 else []}")
        
        # Handle empty results gracefully
        if len(scores) > 0:
            logger.info(f"[OpenYOLO3D Interface]   → Scores shape: {scores.shape}, range: [{scores.min():.4f}, {scores.max():.4f}]")
            logger.info(f"[OpenYOLO3D Interface]   → Total instances: {len(scores)}")
            
            # Log score distribution
            high_conf = (scores > 0.5).sum().item()
            med_conf = ((scores > 0.1) & (scores <= 0.5)).sum().item()
            low_conf = (scores <= 0.1).sum().item()
            logger.info(f"[OpenYOLO3D Interface]   → Score distribution: >0.5: {high_conf}, 0.1-0.5: {med_conf}, <0.1: {low_conf}")
        else:
            logger.warning(f"[OpenYOLO3D Interface]   → NO INSTANCES DETECTED! Scores shape: {scores.shape}")
            logger.warning(f"[OpenYOLO3D Interface]   → This likely means either:")
            logger.warning(f"[OpenYOLO3D Interface]        1. No 3D mask proposals were generated (Mask3D step failed)")
            logger.warning(f"[OpenYOLO3D Interface]        2. No 2D bounding boxes were detected (YOLO-World step failed)")
            logger.warning(f"[OpenYOLO3D Interface]        3. The 3D-2D matching process filtered out all instances")
        
        return {
            'masks': masks,
            'classes': classes,
            'scores': scores,
            'model_instance': self.model  # For advanced usage like save_output_as_ply
        }
