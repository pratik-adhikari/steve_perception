"""
SAM3 Server - Flask REST API for Segment Anything Model 3
Provides endpoints for:
  - Text-prompted segmentation (NEW!)
  - Point-prompted segmentation
  - Bounding box segmentation  
  - Automatic mask generation
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

# Add SAM3 to path
sys.path.insert(0, "/app/sam3")
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

app = Flask(__name__)

# Global model instances
FORCE_CPU = os.environ.get("FORCE_CPU", "False").lower() == "true"
if FORCE_CPU:
    DEVICE = torch.device("cpu")
else:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL = None
PROCESSOR = None


def load_models():
    """Load SAM3 model on startup."""
    global MODEL, PROCESSOR
    
    print(f"[SAM3 Server] Loading SAM3 model on device: {DEVICE}")
    
    # Build SAM3 image model
    MODEL = build_sam3_image_model()
    PROCESSOR = Sam3Processor(MODEL)
    
    print("[SAM3 Server] SAM3 model loaded successfully!")


def decode_image(data: str) -> Image.Image:
    """Decode base64 image to PIL Image."""
    img_bytes = base64.b64decode(data)
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    return img


def encode_mask(mask: np.ndarray) -> str:
    """Encode binary mask to base64 PNG."""
    mask_uint8 = (mask * 255).astype(np.uint8)
    _, buffer = cv2.imencode('.png', mask_uint8)
    return base64.b64encode(buffer).decode('utf-8')


def results_to_response(masks: torch.Tensor, boxes: torch.Tensor, scores: torch.Tensor) -> List[dict]:
    """Convert SAM3 outputs to JSON-serializable format."""
    results = []
    
    # Convert to numpy
    if isinstance(masks, torch.Tensor):
        masks = masks.cpu().numpy()
    if isinstance(boxes, torch.Tensor):
        boxes = boxes.cpu().numpy()
    if isinstance(scores, torch.Tensor):
        scores = scores.cpu().numpy()
    
    for i, (mask, box, score) in enumerate(zip(masks, boxes, scores)):
        # Handle mask dimensions
        if mask.ndim == 3:
            mask = mask[0]
        
        # Calculate centroid and area
        ys, xs = np.where(mask > 0.5)
        if len(xs) == 0:
            continue
        
        centroid = [int(xs.mean()), int(ys.mean())]
        area = int((mask > 0.5).sum())
        
        results.append({
            "id": i,
            "score": float(score),
            "bbox": box.tolist(),  # [x1, y1, x2, y2]
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
        "model": "sam3",
        "device": str(DEVICE),
        "cuda_available": torch.cuda.is_available()
    })


@app.route('/segment_text', methods=['POST'])
def segment_text():
    """
    Text-prompted segmentation using SAM3.
    
    Request JSON:
    {
        "image": "<base64 encoded image>",
        "prompt": "bottle"  # or "red cup", "all dogs", etc.
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
        ],
        "count": 5
    }
    """
    try:
        data = request.get_json()
        image = decode_image(data['image'])
        prompt = data['prompt']
        
        print(f"[SAM3] Segmenting with text prompt: '{prompt}'")
        
        # Set image in processor
        inference_state = PROCESSOR.set_image(image)
        
        # Prompt with text
        output = PROCESSOR.set_text_prompt(state=inference_state, prompt=prompt)
        
        # Get results
        masks = output["masks"]
        boxes = output["boxes"]
        scores = output["scores"]
        
        print(f"[SAM3] Found {len(masks)} objects for prompt '{prompt}'")
        
        results = results_to_response(masks, boxes, scores)
        
        return jsonify({"masks": results, "count": len(results)})
    
    except Exception as e:
        print(f"[SAM3] Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/segment_point', methods=['POST'])
def segment_point():
    """
    Point-prompted segmentation (SAM 1/2 style).
    
    Request JSON:
    {
        "image": "<base64 encoded image>",
        "points": [[x1, y1], [x2, y2], ...],
        "labels": [1, 0, ...],  # 1=foreground, 0=background
        "multimask": false
    }
    """
    try:
        data = request.get_json()
        image = decode_image(data['image'])
        points = data['points']
        labels = data.get('labels', [1] * len(points))
        
        # Set image
        inference_state = PROCESSOR.set_image(image)
        
        # Add point prompts
        for point, label in zip(points, labels):
            PROCESSOR.add_point_prompt(
                state=inference_state,
                point=point,
                label=label
            )
        
        # Get predictions
        output = PROCESSOR.predict(inference_state)
        
        results = results_to_response(
            output["masks"],
            output["boxes"],
            output["scores"]
        )
        
        return jsonify({"masks": results})
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/detect_all', methods= ['POST'])
def detect_all():
    """
    Automatic mask generation - detect all objects.
    
    Request JSON:
    {
        "image": "<base64 encoded image>",
        "min_area": 100
    }
    """
    try:
        data = request.get_json()
        image = decode_image(data['image'])
        min_area = data.get('min_area', 100)
        
        # Use SAM3's automatic detection with empty text prompt
        # (this triggers all-instance detection)
        inference_state = PROCESSOR.set_image(image)
        output = PROCESSOR.set_text_prompt(state=inference_state, prompt="all objects")
        
        results = results_to_response(
            output["masks"],
            output["boxes"],
            output["scores"]
        )
        
        # Filter by area
        results = [r for r in results if r['area'] >= min_area]
        
        return jsonify({"masks": results, "count": len(results)})
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    print("[SAM3 Server] Starting...")
    load_models()
    print("[SAM3 Server] Ready to accept requests on port 5005")
    app.run(host='0.0.0.0', port=5005, threaded=True)
