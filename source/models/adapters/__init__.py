"""
Segmentation adapters for unified pipeline.
"""
from .base_adapter import BaseSegmentationAdapter, SegmentationResult

# Adapters are imported on-demand in run_segmentation.py
# Don't import them here to avoid circular import issues

__all__ = [
    'BaseSegmentationAdapter',
    'SegmentationResult',
]
