"""
Centralized Vocabulary Loading

Provides unified vocabulary loading for all perception models.
Supports LVIS, COCO, furniture-specific, and custom vocabularies.
"""
from typing import List, Optional


# Furniture-specific classes (common indoor objects)
FURNITURE_CLASSES = [
    'chair', 'table', 'desk', 'bed', 'sofa', 'couch', 'armchair',
    'bookshelf', 'shelf', 'cabinet', 'drawer', 'nightstand', 'dresser',
    'wardrobe', 'closet', 'counter', 'countertop', 'kitchen island',
    'dining table', 'coffee table', 'end table', 'side table',
    'tv stand', 'entertainment center', 'bench', 'stool', 'ottoman',
    'lamp', 'floor lamp', 'table lamp', 'light', 'chandelier',
    'picture', 'painting', 'mirror', 'clock', 'vase', 'plant', 'pot',
    'curtain', 'blinds', 'rug', 'carpet', 'pillow', 'cushion',
    'blanket', 'towel', 'basket', 'bin'
]


# COCO 80 classes
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
    'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]


def load_vocabulary(mode: str, custom_classes: Optional[List[str]] = None) -> List[str]:
    """
    Load vocabulary based on the specified mode.
    
    Supported modes:
    - 'furniture': 48 common indoor furniture and object classes
    - 'coco': 80 COCO dataset classes
    - 'lvis': 1203 LVIS classes (loaded from file or default list)
    - 'custom': Use provided custom_classes list
    
    :param mode: Vocabulary mode ('furniture', 'coco', 'lvis', 'custom')
    :param custom_classes: Optional list of custom class names (required if mode='custom')
    :return: List of class names
    """
    mode = mode.lower()
    
    if mode == 'furniture':
        return FURNITURE_CLASSES.copy()
    
    elif mode == 'coco':
        return COCO_CLASSES.copy()
    
    elif mode == 'lvis':
        # For LVIS, we'd normally load from a file, but provide a fallback
        try:
            # Try to load from OpenYOLO3D or similar
            import os
            lvis_file = os.path.join(os.path.dirname(__file__), '../models/lib/OpenYOLO3D/lvis_classes.txt')
            if os.path.exists(lvis_file):
                with open(lvis_file, 'r') as f:
                    return [line.strip() for line in f.readlines()]
        except Exception:
            pass
        
        # Fallback: return COCO + furniture as a reasonable subset
        print("[WARNING] LVIS vocabulary file not found, using COCO + furniture as fallback")
        combined = list(set(COCO_CLASSES + FURNITURE_CLASSES))
        return sorted(combined)
    
    elif mode == 'custom':
        if custom_classes is None:
            raise ValueError("custom_classes must be provided when mode='custom'")
        return custom_classes.copy()
    
    else:
        raise ValueError(f"Unknown vocabulary mode: {mode}. "
                        f"Supported modes: 'furniture', 'coco', 'lvis', 'custom'")


def get_vocabulary_size(mode: str) -> int:
    """
    Get the number of classes in a vocabulary mode.
    
    :param mode: Vocabulary mode
    :return: Number of classes
    """
    if mode == 'furniture':
        return len(FURNITURE_CLASSES)
    elif mode == 'coco':
        return len(COCO_CLASSES)
    elif mode == 'lvis':
        return 1203  # Standard LVIS dataset size
    else:
        return 0
