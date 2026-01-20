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
        # Update frequency/frame_step in the model config dynamically
        if self.model and hasattr(self.model, 'openyolo3d_config'):
            self.model.openyolo3d_config['openyolo3d']['frequency'] = frame_step

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
        
        return {
            'masks': masks,
            'classes': classes,
            'scores': scores,
            'model_instance': self.model  # For advanced usage like save_output_as_ply
        }
