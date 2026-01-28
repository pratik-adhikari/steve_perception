"""
Pydantic models for configuration validation.
Ensures type safety and provides helpful error messages.
"""
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Literal


class InferenceConfig(BaseModel):
    """Inference parameters."""
    frame_step: int = Field(default=10, ge=1, description="Frame sampling rate")
    conf_threshold: float = Field(default=0.1, ge=0.0, le=1.0, description="Confidence threshold")
    depth_scale: float = Field(default=1000.0, gt=0.0, description="Depth scale factor")
    batch_size: int = Field(default=200000, ge=1000, description="Batch size for projection")


class VocabularyConfig(BaseModel):
    """Vocabulary configuration."""
    mode: Literal['lvis', 'coco', 'furniture', 'custom'] = Field(description="Vocabulary mode")
    custom_classes: List[str] = Field(default_factory=list, description="Custom class names")
    
    @field_validator('custom_classes')
    @classmethod
    def validate_custom_classes(cls, v, info):
        """Ensure custom_classes is provided when mode is 'custom'."""
        if info.data.get('mode') == 'custom' and not v:
            raise ValueError("custom_classes must be provided when mode='custom'")
        return v


class OutputConfig(BaseModel):
    """Output format configuration."""
    save_detailed_objects: bool = Field(default=True, description="Save individual object PLYs")
    save_scenegraph_format: bool = Field(default=True, description="Save SceneGraph format")
    save_visualization: bool = Field(default=True, description="Save colored mesh")
    generate_scene_graph: bool = Field(default=True, description="Auto-generate scene graph")


class ModelConfig(BaseModel):
    """Model selection."""
    name: Literal['openyolo3d', 'mask3d'] = Field(description="Model to use")


class OpenYolo3DConfig(BaseModel):
    """OpenYOLO3D specific settings."""
    config_path: str = Field(default='source/models/lib/OpenYOLO3D/pretrained/config.yaml')
    use_sam: bool = Field(default=False, description="Use SAM for refinement")


class Mask3DConfig(BaseModel):
    """Mask3D specific settings."""
    checkpoint: str = Field(default='saved/scannet200/scannet200_benchmark.ckpt')
    voxel_size: float = Field(default=0.02, gt=0.0)
    use_dbscan: bool = Field(default=False)


class SegmentationConfig(BaseModel):
    """Complete segmentation pipeline configuration."""
    model: ModelConfig
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    vocabulary: VocabularyConfig
    output: OutputConfig = Field(default_factory=OutputConfig)
    openyolo3d: Optional[OpenYolo3DConfig] = Field(default_factory=OpenYolo3DConfig)
    mask3d: Optional[Mask3DConfig] = Field(default_factory=Mask3DConfig)
    
    def get_model_specific_config(self):
        """Get configuration for the selected model."""
        if self.model.name == 'openyolo3d':
            return self.openyolo3d
        elif self.model.name == 'mask3d':
            return self.mask3d
        else:
            raise ValueError(f"Unknown model: {self.model.name}")


def load_and_validate_config(config_dict: dict) -> SegmentationConfig:
    """
    Load and validate configuration from dictionary.
    
    Args:
        config_dict: Raw configuration dictionary from YAML
        
    Returns:
        Validated SegmentationConfig object
        
    Raises:
        ValidationError: If configuration is invalid
    """
    return SegmentationConfig(**config_dict)
