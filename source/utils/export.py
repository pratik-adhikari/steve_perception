
"""
Export utilities for saving segmentation results.
"""
import os
import numpy as np
import colorsys
from typing import Tuple, List

def get_high_contrast_color(index: int) -> Tuple[int, int, int]:
    """Generate high contrast color for visualization."""
    golden_ratio_conjugate = 0.618033988749895
    h = (index * golden_ratio_conjugate) % 1
    s = 0.8 + (index % 4) * 0.05
    v = 0.95
    
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return (int(r * 255), int(g * 255), int(b * 255))

def get_closest_color_name(rgb: Tuple[int, int, int]) -> str:
    """Get human readable name for a color (placeholder)."""
    return f"rgb_{rgb[0]}_{rgb[1]}_{rgb[2]}"
