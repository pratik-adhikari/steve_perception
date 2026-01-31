# STEVE Perception

**A Unified 3D Perception & Scene Graph Pipeline for ROS 2.**

This project uses a dual-environment approach:
1.  **Scanner Env (ROS 2)**: For SLAM, data collection, and export.
2.  **Brain Env (Docker)**: For heavy AI processing (3D segmentation, scene graphs).

---

## 1. Installation

### Clone Repository
```bash
git clone --recursive https://github.com/pratik-adhikari/steve_perception.git
cd steve_perception
```

### Download Checkpoints
Required for the AI models.
```bash
cd source/models/lib/OpenYOLO3D
./scripts/get_checkpoints.sh
cd ../../../..
```

---

## 2. Phase A: Mapping & Export (The "Scanner")
*Runs in your ROS 2 environment (native or standard ROS Docker).*

### Requirements
Ensure you have `rtabmap_ros` and `open3d` installed in your ROS 2 environment:
```bash
sudo apt install ros-humble-rtabmap-ros ros-humble-rtabmap-msgs
pip3 install open3d scipy
```

### Step 1: Run Mapping
Launch the sensor and RTAB-Map stack:
```bash
# Run from package root (src/steve_perception)
ros2 launch steve_perception mapping_pan_tilt.launch.py
```
*Move your robot/camera to map the area. Closing the session saves `data/rtabmap.db` (auto-created in this folder).
For configuration, refer to:
  - `config/rtabmap_*.ini`
    - Internal RTAB-Map algorithm parameters.

  - `source/configs/rtabmap_export.yaml`
    - **Export Settings**: Control voxel size (e.g. 0.02 vs 0.05), enable Multi-Resolution output, and set max points.
    - Used by `source/scripts/export_data.py`.
*

### Step 2: Export Data
Convert the database into the standardized format for the AI pipeline:
```bash
# Run from package root
python3 source/scripts/export_data.py data/rtabmap.db --output_dir data/pipeline_output
```
**Output:** You will see a `data/pipeline_output` folder containing `export/` (images, meshes) and `cloud.ply` / `mesh.ply`.

---

## 3. Phase B: Perception understanding (brain)
*Runs in a specialized GPU-accelerated Docker container.*

### Step 1: Build the Brain by understanding features and whats in scene
Build the Unified Perception image (contains OpenYOLO3D, Mask3D, CUDA deps):
```bash
cd source/models/lib/OpenYOLO3D
docker build -t steve_perception:unified .
cd ../../../..
```
*(Note: This build takes time as it compiles CUDA kernels)*

### Step 2: Run Segmentation
Run the full segmentation pipeline on your exported data:
```bash
# Ensure you are in package root (src/steve_perception)
docker run --rm --gpus all -v $(pwd):/workspace -w /workspace -e PYTHONPATH="/workspace/source" -e PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128" steve_perception:unified python3 source/scripts/run_segmentation.py --data data/pipeline_output --model openyolo3d --vocab furniture
```

**Arguments:**
- `--vocab`: `furniture` (default), `coco`, `lvis`, or `custom`.
  *(Note: 'furniture' vocab is strict. Use 'coco' for general objects)*

### Step 3: Generate Scene Graph
After segmentation, generate the scene graph (nodes and relationships):
```bash
docker run --rm --gpus all \
  -v $(pwd):/workspace \
  -w /workspace \
  -e PYTHONPATH="/workspace/source" \
  steve_perception:unified \
  python3 source/scripts/build_scene_graph.py \
  --input data/pipeline_output/openyolo3d_output \
  --output data/pipeline_output/generated_graph
```

## Usage
**[📖 Click here for the full Pipeline Usage Guide](docs/PIPELINE_USAGE.md)**

The pipeline consists of 4 modular steps:

1.  **Segmentation (Mask3D)**
    ```bash
    python3 source/scripts/run_segmentation.py --data data/pipeline_output --model mask3d --vocab furniture
    ```

2.  **Detection & Fusion (YOLO-Drawer)**
    ```bash
    python3 source/scripts/run_yolodrawer.py --data data/pipeline_output
    ```

3.  **Combination**
    ```bash
    python3 source/scripts/combine_inferences.py --data data/pipeline_output --output combined_output
    ```

4.  **Graph Generation**
    ```bash
    python3 source/scripts/build_scene_graph.py --input data/pipeline_output/combined_output --output data/pipeline_output/generated_graph
    ```

For **debugging options** (visualizations) and **individual model analysis** (YOLO-only graphs), see the [Usage Guide](docs/PIPELINE_USAGE.md).

## 4. Diagnostics & Verification
If you suspect sensor drift or misalignment, use the included diagnostic tools:

### Visualizing 3D Drift (Reprojection)
Generate 3D-to-2D overlays to verify if the 3D position of objects stays consistent across frames:
```bash
python3 source/scripts/diagnose_reprojection.py
```
**Output**: `data/pipeline_output/generated_graph_yolo/debug_reprojections/`
Check the yellow point clouds and "World Centroid" text. If the centroid shifts significantly for the same object, verify your `rtabmap.db` or re-export.

---

## Documentation
- [**Technical Details**](docs/TECHNICAL_DETAILS.md): ROS 2 architecture, RTAB-Map tuning.
- [**Architecture**](docs/ARCHITECTURE_AND_PROPOSALS.md): Internal code overview.