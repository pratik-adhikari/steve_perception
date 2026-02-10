"""
SAM3 Server - Flask REST API for Segment Anything Model
Provides endpoints for:
  - Point-prompted segmentation
  - Bounding box segmentation  
  - Automatic mask generation (detect all)
"""

import io
import json
import os
import sys
import base64
from typing import List, Tuple, Optional

import cv2
import numpy as np
import torch
from flask import Flask, request, jsonify
from PIL import Image

# Add SAM2 to path
sys.path.insert(0, "/app/sam2")
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

app = Flask(__name__)

# Global model instances
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PREDICTOR = None
MASK_GENERATOR = None

# Model paths
SAM2_CHECKPOINT = "/app/sam2/checkpoints/sam2.1_hiera_large.pt"
MODEL_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"


def load_models():
    """Load SAM2 models on startup."""
    global PREDICTOR, MASK_GENERATOR
    
    print(f"[SAM3 Server] Loading models on device: {DEVICE}")
    
    # Enable optimizations
    if DEVICE.type == "cuda":
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    
    # Build model
    sam2_model = build_sam2(MODEL_CFG, SAM2_CHECKPOINT, device=DEVICE)
    
    # Create predictor for point/box prompts
    PREDICTOR = SAM2ImagePredictor(sam2_model)
    
    # Create automatic mask generator for "detect all"
    MASK_GENERATOR = SAM2AutomaticMaskGenerator(
        model=sam2_model,
        points_per_side=32,
        points_per_batch=64,
        pred_iou_thresh=0.7,
        stability_score_thresh=0.92,
        crop_n_layers=1,
        min_mask_region_area=100,
    )
    
    print("[SAM3 Server] Models loaded successfully!")


def decode_image(data: str) -> np.ndarray:
    """Decode base64 image to numpy array."""
    img_bytes = base64.b64decode(data)
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    return np.array(img)


def encode_mask(mask: np.ndarray) -> str:
    """Encode binary mask to base64 PNG."""
    mask_uint8 = (mask * 255).astype(np.uint8)
    _, buffer = cv2.imencode('.png', mask_uint8)
    return base64.b64encode(buffer).decode('utf-8')


def masks_to_response(masks: np.ndarray, scores: np.ndarray) -> List[dict]:
    """Convert masks and scores to JSON-serializable format."""
    results = []
    for i, (mask, score) in enumerate(zip(masks, scores)):
        # Get bounding box from mask
        ys, xs = np.where(mask)
        if len(xs) == 0:
            continue
        bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
        
        # Calculate centroid
        centroid = [int(xs.mean()), int(ys.mean())]
        
        # Calculate area
        area = int(mask.sum())
        
        results.append({
            "id": i,
            "score": float(score),
            "bbox": bbox,  # [x1, y1, x2, y2]
            "centroid": centroid,  # [cx, cy]
            "area": area,
            "mask": encode_mask(mask)
        })
    
    return results


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "device": str(DEVICE),
        "cuda_available": torch.cuda.is_available()
    })


@app.route('/segment_point', methods=['POST'])
def segment_point():
    """
    Segment object at given point coordinates.
    
    Request JSON:
    {
        "image": "<base64 encoded image>",
        "points": [[x1, y1], [x2, y2], ...],
        "labels": [1, 0, ...],  # 1=foreground, 0=background
        "multimask": false
    }
    
    Response JSON:
    {
        "masks": [
            {
                "id": 0,
                "score": 0.95,
                "bbox": [x1, y1, x2, y2],
                "centroid": [cx, cy],
                "area": 12345,
                "mask": "<base64 encoded PNG mask>"
            },
            ...
        ]
    }
    """
    try:
        data = request.get_json()
        image = decode_image(data['image'])
        points = np.array(data['points'])
        labels = np.array(data.get('labels', [1] * len(points)))
        multimask = data.get('multimask', False)
        
        # Set image
        PREDICTOR.set_image(image)
        
        # Predict
        masks, scores, logits = PREDICTOR.predict(
            point_coords=points,
            point_labels=labels,
            multimask_output=multimask
        )
        
        # Sort by score
        sorted_idx = np.argsort(scores)[::-1]
        masks = masks[sorted_idx]
        scores = scores[sorted_idx]
        
        results = masks_to_response(masks, scores)
        
        return jsonify({"masks": results})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/segment_box', methods=['POST'])
def segment_box():
    """
    Segment object within bounding box.
    
    Request JSON:
    {
        "image": "<base64 encoded image>",
        "box": [x1, y1, x2, y2],
        "multimask": false
    }
    """
    try:
        data = request.get_json()
        image = decode_image(data['image'])
        box = np.array(data['box'])
        multimask = data.get('multimask', False)
        
        PREDICTOR.set_image(image)
        
        masks, scores, logits = PREDICTOR.predict(
            box=box,
            multimask_output=multimask
        )
        
        sorted_idx = np.argsort(scores)[::-1]
        masks = masks[sorted_idx]
        scores = scores[sorted_idx]
        
        results = masks_to_response(masks, scores)
        
        return jsonify({"masks": results})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/detect_all', methods=['POST'])
def detect_all():
    """
    Automatic mask generation - detect all objects in image.
    
    Request JSON:
    {
        "image": "<base64 encoded image>",
        "min_area": 100,  # optional: minimum mask area
        "max_masks": 50   # optional: maximum number of masks to return
    }
    
    Response JSON:
    {
        "masks": [
            {
                "id": 0,
                "score": 0.95,
                "bbox": [x1, y1, x2, y2],
                "centroid": [cx, cy],
                "area": 12345,
                "mask": "<base64 encoded PNG mask>"
            },
            ...
        ]
    }
    """
    try:
        data = request.get_json()
        image = decode_image(data['image'])
        min_area = data.get('min_area', 100)
        max_masks = data.get('max_masks', 50)
        
        # Generate all masks
        masks_data = MASK_GENERATOR.generate(image)
        
        # Filter by area and sort by score
        masks_data = [m for m in masks_data if m['area'] >= min_area]
        masks_data = sorted(masks_data, key=lambda x: x['predicted_iou'], reverse=True)
        masks_data = masks_data[:max_masks]
        
        results = []
        for i, m in enumerate(masks_data):
            mask = m['segmentation']
            ys, xs = np.where(mask)
            
            results.append({
                "id": i,
                "score": float(m['predicted_iou']),
                "stability_score": float(m['stability_score']),
                "bbox": list(m['bbox']),  # [x, y, w, h] -> convert if needed
                "centroid": [int(xs.mean()), int(ys.mean())] if len(xs) > 0 else [0, 0],
                "area": int(m['area']),
                "mask": encode_mask(mask)
            })
        
        return jsonify({"masks": results, "count": len(results)})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/segment_with_detection', methods=['POST'])
def segment_with_detection():
    """
    Combined endpoint: use detected bounding boxes to get precise SAM masks.
    Useful when you have YOLO detections and want SAM refinement.
    
    Request JSON:
    {
        "image": "<base64 encoded image>",
        "detections": [
            {"box": [x1, y1, x2, y2], "label": "cup", "confidence": 0.85},
            ...
        ]
    }
    """
    try:
        data = request.get_json()
        image = decode_image(data['image'])
        detections = data['detections']
        
        PREDICTOR.set_image(image)
        
        results = []
        for i, det in enumerate(detections):
            box = np.array(det['box'])
            
            masks, scores, _ = PREDICTOR.predict(
                box=box,
                multimask_output=False
            )
            
            if len(masks) > 0:
                mask = masks[0]
                score = scores[0]
                
                ys, xs = np.where(mask)
                if len(xs) > 0:
                    results.append({
                        "id": i,
                        "label": det.get('label', 'unknown'),
                        "detection_confidence": det.get('confidence', 0.0),
                        "sam_score": float(score),
                        "bbox": det['box'],
                        "centroid": [int(xs.mean()), int(ys.mean())],
                        "area": int(mask.sum()),
                        "mask": encode_mask(mask)
                    })
        
        return jsonify({"masks": results, "count": len(results)})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    print("[SAM3 Server] Starting...")
    load_models()
    print("[SAM3 Server] Ready to accept requests on port 5005")
    app.run(host='0.0.0.0', port=5005, threaded=True)
