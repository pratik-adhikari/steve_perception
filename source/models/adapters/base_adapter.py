"""
Base adapter interface for all segmentation models.
Defines common data structures and abstract interface.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Any
import numpy as np


@dataclass
class SegmentationResult:
    """Standardized segmentation output format."""
    masks: np.ndarray  # Shape: (num_points, num_instances) - binary masks
    classes: np.ndarray  # Shape: (num_instances,) - class indices
    scores: np.ndarray  # Shape: (num_instances,) - confidence scores
    points: np.ndarray  # Shape: (num_points, 3) - XYZ coordinates
    colors: Optional[np.ndarray] = None  # Shape: (num_points, 3) - RGB colors
    mesh_path: Optional[str] = None  # Path to input mesh
    metadata: Dict[str, Any] = None  # Additional info
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class BaseSegmentationAdapter(ABC):
    """Abstract base class for all segmentation model adapters."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the adapter with configuration.
        
        Args:
            config: Dictionary with model-specific configuration
        """
        self.config = config
    
    @abstractmethod
    def predict(self, scene_path: str, vocabulary: list, output_dir: str) -> SegmentationResult:
        """
        Run segmentation on a scene.
        
        Args:
            scene_path: Path to input scene (mesh or point cloud)
            vocabulary: List of class names to detect
            output_dir: Directory to save outputs
            
        Returns:
            SegmentationResult with masks, classes, scores, and geometry
        """
        pass
    
    @abstractmethod
    def get_model_name(self) -> str:
        """Return the name of this model."""
        pass
