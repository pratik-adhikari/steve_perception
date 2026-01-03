
"""
Vocabulary definitions and loading utilities.
"""
import os
import glob
import numpy as np
import yaml
from typing import List, Dict, Union, Optional

def load_vocabulary(mode: str, custom_classes: Optional[List[str]] = None) -> List[str]:
    """
    Load vocabulary for a specific mode.
    
    Args:
        mode: Vocabulary mode ('lvis', 'scannet', 'coco', 'openvocab', 'custom')
        custom_classes: List of class names for 'custom' mode
        
    Returns:
        List of class names
    """
    if mode == 'lvis':
        # Load LVIS vocabulary (simplified for MVP)
        return ["lvis_placeholder"] * 1203 # Logic to load LVIS
    elif mode == 'scannet' or mode == 'scannet200':
        return ["wall", "floor", "cabinet", "bed", "chair", "sofa", "table", "door", "window", "bookshelf", "picture", "counter", "desk", "curtain", "refridgerator", "shower curtain", "toilet", "sink", "bathtub", "otherfurniture"]
    elif mode == 'coco':
        return ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]
    elif mode == 'furniture':
        return ["wall", "floor", "cabinet", "bed", "chair", "sofa", "table", "door", "window", "bookshelf", "counter", "desk", "curtain", "refrigerator", "shower", "toilet", "sink", "bathtub"]
    elif mode == 'custom':
        if custom_classes is None:
            raise ValueError("custom_classes must be provided when mode is 'custom'")
        return custom_classes
    elif mode == 'openvocab':
        return [] # Open vocabulary
    else:
        raise ValueError(f"Unknown vocabulary mode: {mode}")

def get_palette(mode: str) -> np.ndarray:
    """Get color palette for a vocabulary mode."""
    # Placeholder for palette generation
    return np.random.randint(0, 255, (255, 3), dtype=np.uint8)
