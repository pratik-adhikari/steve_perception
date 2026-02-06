# STEVE Perception

Unified 3D Perception & Scene Graph Pipeline.

---

## 1. Installation

### Repository
```bash
git clone --recursive https://github.com/pratik-adhikari/steve_perception.git
cd steve_perception
```

### Docker Setup
Required for full pipeline (Mapping + AI).
```bash
# Pull images and build custom services
docker compose pull
docker compose build
```

### Native Setup (Alternative)
Requirements: ROS 2 Humble, `rtabmap_ros`, `open3d`, `scipy`.
```bash
sudo apt install ros-humble-rtabmap-ros ros-humble-rtabmap-msgs ros-humble-navigation2
pip3 install open3d scipy
```

---

## 2. Mapping & Export

### Step 1: RTAB-Map
**Using Docker:**
```bash
docker compose up -d mapping
docker compose exec mapping ros2 launch steve_perception mapping.launch.py
```

**Using Native ROS 2:**
```bash
ros2 launch steve_perception mapping.launch.py
```
*Saves to `data/rtabmap.db` on exit.*

**Configuration:**
- `config/mapping.yaml`: Topics & Logging.
- `config/rtabmap_*.ini`: RTAB-Map Parameters.
- `source/configs/rtabmap_export.yaml`: export voxel size.

### Step 2: Export Data
```bash
# Export .db to .ply and images
python3 source/scripts/export_data.py data/rtabmap.db --output_dir data/pipeline_output
```

---

## 3. Perception Pipeline (AI)

Run these steps using the Docker services.

### Step 1: Segmentation (Mask3D)
```bash
# Start Service
docker compose up -d mask3d

# Execute Segmentation
docker compose exec mask3d python3 source/scripts/run_segmentation.py --data data/pipeline_output --model mask3d
```

### Step 2: Detection & Fusion
**Option A: YOLO-Drawer**
```bash
docker compose up -d yolodrawer
python3 source/scripts/run_yolodrawer.py --data data/pipeline_output
```

**Option B: OpenMask3D**
```bash
docker compose up -d openmask
python3 source/scripts/run_openmask_client.py --data data/pipeline_output
```

### Step 3: Scene Graph Generation
```bash
python3 source/scripts/combine_inferences.py --data data/pipeline_output --output combined_output
python3 source/scripts/build_scene_graph.py --input data/pipeline_output/combined_output --output data/pipeline_output/generated_graph
```

---

## Documentation
- [**Pipeline Details**](docs/PIPELINE_USAGE.md)
- [**Technical Architecture**](docs/TECHNICAL_DETAILS.md)