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
ros2 launch steve_perception mapping_pan_tilt.launch.py
```
*Move your robot/camera to map the area. Closing the session saves `~/.ros/rtabmap.db` (default).*

### Step 2: Export Data
Convert the database into the standardized format for the AI pipeline:
```bash
# Locate your rtabmap.db (usually in ~/.ros/ or specified in launch)
python3 source/scripts/export_data.py ~/.ros/rtabmap.db --output_dir data/pipeline_output
```
**Output:** You will see a `data/pipeline_output` folder containing `export/` (images, meshes) and `scene.ply`.

---

## 3. Phase B: AI Processing (The "Brain")
*Runs in a specialized GPU-accelerated Docker container.*

### Step 1: Build the Brain
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
docker run --rm --gpus all \
  -v $(pwd):/workspace \
  -w /workspace \
  -e PYTHONPATH="/workspace/source" \
  -e PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128" \
  steve_perception:unified \
  python3 source/scripts/run_segmentation.py \
  --data data/pipeline_output \
  --model openyolo3d \
  --vocab furniture
```

**Arguments:**
- `--vocab`: `furniture` (default), `coco`, `lvis`, or `custom`.
  *(Note: 'furniture' vocab is strict. Use 'coco' for general objects)*

---

## 4. Visualization

View the results from your host machine:

```bash
# View the labeled 3D mesh
python3 source/scripts/visualize_mesh_labels.py data/pipeline_output/openyolo3d_output

# Browse individual object point clouds
python3 source/scripts/visualize_objects.py data/pipeline_output/openyolo3d_output/objects
```

---

## Documentation
- [**Technical Details**](docs/TECHNICAL_DETAILS.md): ROS 2 architecture, RTAB-Map tuning.
- [**Architecture**](docs/ARCHITECTURE_AND_PROPOSALS.md): Internal code overview.
