"""
Model adapters initialization.
"""
from .base_adapter import SegmentationResult, BaseSegmentationAdapter
from .openyolo3d_adapter import OpenYolo3DAdapter
from .mask3d_adapter import Mask3DAdapter

__all__ = [
    'SegmentationResult',
    'BaseSegmentationAdapter',
    'OpenYolo3DAdapter',
    'Mask3DAdapter',
]
