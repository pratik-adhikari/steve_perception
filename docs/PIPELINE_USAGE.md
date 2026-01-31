# Steve Perception Pipeline Usage Guide

This document describes how to run the integrated 3D perception pipeline, which combines Mask3D (for furniture) and a fused YOLO-Drawer system (for drawers/doors).

## Overview
The pipeline operates in **4 sequential steps**. All scripts take an explicit `--data` argument pointing to your data directory.

**Data Directory Structure:**
Your data folder (e.g., `data/pipeline_output`) must contain an `export/` subdirectory with the scanned data (images, poses, mesh).

## Execution Steps

### Step 1: Segmentation (Mask3D)
Generates dense segmentation masks for furniture (chairs, tables, etc.).
```bash
python3 source/scripts/run_segmentation.py --data data/pipeline_output
```
**Output**: `mask3d_output/`

### Step 2: YOLO Detection & Fusion
Runs the YOLO model to detect drawers and doors. 
*   **Fusion**: Uses temporal clustering and 3D NMS to merge duplicate detections across frames into robust unique objects.
*   **Debug**: Automatically generates visualization images in `drawer_priors/debug_images/` showing detections > 0.5 (for troubleshooting).
*   **Threshold**: The fused output (`boxes_3d.json`) uses a strict confidence threshold (0.9) to ensure high-quality graph nodes.
```bash
python3 source/scripts/run_yolodrawer.py --data data/pipeline_output
```
**Output**: `drawer_priors/` (contains `boxes_3d.json`)

### Step 3: Combine Inferences
Merges the outputs from Step 1 and Step 2 into a single dataset.
```bash
python3 source/scripts/combine_inferences.py --data data/pipeline_output
```
**Output**: `combined_output/`

### Step 4: Scene Graph Generation
Builds the final semantic scene graph.
```bash
python3 source/scripts/build_scene_graph.py --input data/pipeline_output/combined_output --output data/pipeline_output/generated_graph
```
**Output**: `generated_graph/` (contains `graph.json` and interactive HTML)

---

## Debugging & Advanced Options

### Visualizing YOLO Detections
To verify what the 2D model detected, check the debug folder generated in Step 2:
- **Path**: `<data_dir>/drawer_priors/debug_images/`
- **Content**: Images with bounding boxes and confidence scores for every frame.

### Individual Model Analysis
You can generate scene graphs for *only* drawers or *only* Mask3D objects to isolate performance issues.

#### YOLO-Only Graph
1. Create a drawer-only dataset:
   ```bash
   python3 source/scripts/combine_inferences.py --data data/pipeline_output --output yolo_output --only-drawers
   ```
2. Generate the graph:
   ```bash
   python3 source/scripts/build_scene_graph.py --input data/pipeline_output/yolo_output --output data/pipeline_output/generated_graph_yolo
   ```

#### Mask3D-Only Graph
Run the builder directly on the mask3d output (flag `--no-drawers` disables looking for fused boxes):
```bash
python3 source/scripts/build_scene_graph.py --input data/pipeline_output/mask3d_output --output data/pipeline_output/generated_graph_mask3d --no-drawers
```

### Logging
All scripts save timestamped logs to `<data_dir>/logs/`. Check these files if a script fails silently.
