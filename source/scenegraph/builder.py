"""
Scene graph builder using unified pipeline output.
"""
from models.adapters.base_adapter import SegmentationResult
from typing import List, Dict, Any


class SceneGraphBuilder:
    """Build scene graph from segmentation results."""
    
    def __init__(self):
        pass
    
    def build_from_result(self, result: SegmentationResult, vocabulary: List[str]) -> Dict[str, Any]:
        """
        Build scene graph from unified segmentation result.
        
        Args:
            result: SegmentationResult from pipeline
            vocabulary: List of class names
            
        Returns:
            Scene graph dictionary with nodes and edges
        """
        nodes = []
        
        for i in range(len(result.scores)):
            class_idx = int(result.classes[i])
            class_name = vocabulary[class_idx] if class_idx < len(vocabulary) else f"class_{class_idx}"
            mask = result.masks[:, i].astype(bool)
            
            obj_points = result.points[mask]
            if len(obj_points) < 10:
                continue
            
            centroid = obj_points.mean(axis=0).tolist()
            
            nodes.append({
                'id': i,
                'label': class_name,
                'confidence': float(result.scores[i]),
                'centroid': centroid,
                'num_points': int(mask.sum())
            })
        
        return {
            'nodes': nodes,
            'edges': [],
            'metadata': result.metadata
        }
