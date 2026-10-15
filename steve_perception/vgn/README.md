# VGN grasp detection

Volumetric Grasping Network ([ethz-asl/vgn](https://github.com/ethz-asl/vgn)) for Steve.
Everything VGN needs lives here; no external VGN checkout is required.

```
vgn/
├── bridge.py          VGNBridge: collects depth views (MuJoCo or real camera), runs the worker
├── worker.py          inference subprocess, runs under the vgn_env interpreter
├── models/
│   └── vgn_conv.pth   pretrained weights (upstream data/models/vgn_conv.pth)
└── third_party/vgn/   vendored subset of ethz-asl/vgn (BSD-3, see LICENSE)
```

## What is vendored

Only the modules the worker imports, from `ItsShriks/vgn` @ `28162b0` (fork of ethz-asl/vgn):

| Module | Used for |
|---|---|
| `perception.py` | `TSDFVolume`, `CameraIntrinsic` (Open3D TSDF integration) |
| `networks.py` | `load_network` (ConvNet) |
| `detection.py` | `predict`, `process`, `select` |
| `grasp.py` | `Grasp`, `from_voxel_coordinates` |
| `utils/transform.py` | `Transform`, `Rotation` |

The only change to them: the ROS 1 RViz wrapper class `VGN` and its `from vgn import vis` import
were removed from `detection.py`, so the worker never imports `rospy` / `tf2_ros`.
Simulation (pybullet), training, datasets, URDFs and the ROS 1 nodes are not vendored.

## Python environment (`vgn_env`)

VGN runs in its own Python 3.10 interpreter, because the torch / open3d wheels must not mix with
the ROS 2 Python. The bridge starts the worker with the ROS `PYTHONPATH` removed and user
site-packages disabled.

```bash
conda create -n vgn_env python=3.10
conda activate vgn_env
pip install numpy scipy torch open3d
```

The worker needs only `numpy`, `scipy`, `torch` and `open3d`. `VGNBridge.start_health_check()`
verifies that torch and NumPy work together (torch built for NumPy 1.x fails with NumPy 2).

The interpreter is found at `~/{anaconda3,miniconda3}/envs/vgn_env/bin/python3`; override it with
`VGN_PYTHON=/path/to/python`, and the weights with `VGN_MODEL=/path/to/model.pth`.

## Usage

Each inference starts `worker.py` in `vgn_env` (~4–7 s: interpreter, torch, network). The robot
WBC calls `start_health_check()` instead, which keeps one worker running (`worker.py --serve`,
loaded and warmed up in ~3 s): ~0.1 s per inference afterwards. `close()` stops it.

```python
from steve_perception.vgn import VGNBridge

vgn = VGNBridge()
vgn.add_real_view(depth_m, cam_pos, R_world_optical, K, object_xyz)   # or collect_view(...) in MuJoCo
if vgn.run_inference():
    pos, rot, width, score = vgn.best_grasp_world(near=object_xyz, xy_radius=0.05, z_window=0.10)
```
