# Perception model servers (Grounded SAM, Contact-GraspNet)

GPU models for clutter, each in its **own conda env**, kept loaded in a server process that the
ROS nodes (system Python) call over ZMQ:

| Server | Model | Env | Endpoint | Used by |
|---|---|---|---|---|
| `gsam` | GroundingDINO (SwinT) + SAM (ViT-B): text prompt → object masks | `gsam_env` | `tcp://127.0.0.1:5556` | `gsam_target_detector`, `CGNBridge` |
| `cgn` | Contact-GraspNet (PyTorch port): depth + target mask → 6-DOF grasps | `cgn_env` | `tcp://127.0.0.1:5555` | `CGNBridge`, `grasp_test/cgn_client.py` (DiNBV-Grasp protocol) |

```
model_server/
├── server.py      one model per process (env interpreter): ZMQ REP, pickled dicts
├── gsam.py        GroundedSAM.segment(rgb, prompt) -> [{box, score, phrase, mask}]
├── cgn.py         ContactGraspNet.grasps(depth, K, segmap) -> {pred_grasps_cam, scores, ...}
├── launcher.py    `ros2 run steve_perception model_server gsam|cgn`: config -> env python -> server.py
├── client.py      ModelClient (ROS side): segment(), grasps(), ping()
└── envs/          gsam_env.yml, cgn_env.yml
config/model_server.yaml   env, endpoint and model paths per server
```

## Setup (once)

Model code and weights are not in git: they live in the workspace's `perception_tests/`
(with a `COLCON_IGNORE`, listed in the workspace `.gitignore`):

```
~/steve_ros2_ws/perception_tests/
├── contact_graspnet_pytorch/   git clone https://github.com/elchun/contact_graspnet_pytorch (checkpoint included)
├── GroundingDINO/              GroundingDINO source (IDEA-Research)
├── segment-anything/           SAM source (facebookresearch)
└── weights/                    groundingdino_swint_ogc.pth, sam_vit_b_01ec64.pth
```

On this PC `GroundingDINO`, `segment-anything` and the two weights are symlinks into
`~/grounded_sam_ros2_perception`. GroundingDINO's text encoder (`bert-base-uncased`) is taken
from the Hugging Face cache (`~/.cache/huggingface`); the server runs offline unless
`--allow_download`.

```bash
cd ~/steve_ros2_ws/src/steve_perception/steve_perception/model_server/envs
PYTHONNOUSERSITE=1 conda env create -f gsam_env.yml
PYTHONNOUSERSITE=1 conda env create -f cgn_env.yml
sudo apt install python3-zmq          # the ROS-side client
```

`PYTHONNOUSERSITE=1`: packages in `~/.local` (e.g. a user-site torch) otherwise count as
installed and pip leaves them out of the env. The servers run with the user site disabled too.

## Run

The launch file starts what it needs (`active_perception_wbc.launch.py detector:=gsam` /
`grasp_backend:=cgn`). To keep the models loaded across runs, start them once in their own
terminals and launch with `model_servers:=false`:

```bash
ros2 run steve_perception model_server gsam
ros2 run steve_perception model_server cgn
```

Options after the model name override the config, e.g. `--gdino_image_size 800`, `--fp32`,
`--bind tcp://0.0.0.0:5556` (pickle runs code on load: open a port only between trusted machines).

Detector on its own (publishes `/steve_perception/target_point`, overlay on
`/gsam_target_detector/debug_image`):

```bash
ros2 run steve_perception gsam_target_detector --ros-args -p prompt:="red ball" \
    -p "cameras:=[gripper_camera]" -p object_size:=0.065
```

## Offline self-test (no robot, no camera)

```bash
ros2 run steve_perception perception_selftest --vgn     # newest grasp_test snapshot, servers running
```

`~/ros_logs/perception_<time>/selftest/`: `gsam.jpg` (masks), `scan01_view1.jpg` (Contact-GraspNet),
`scan02_view1.jpg` (VGN), each with `.npz` + `_grasps.csv`.

## Saved images

Every run started with `ros2 launch` keeps them in its log folder (`~/ros_logs/<launch>_<time>/`,
`~/ros_logs/latest`); nodes started with `ros2 run` in `~/ros_logs/perception_<time>/`:

| Folder | Written by | Content |
|---|---|---|
| `detector/` | `gsam_target_detector`, `red_ball_detector` (`save_dir`, `save_period` 0.5 s) | annotated frame every 0.5 s (masks, scores, rejection reason, chosen object, position); `detections.csv` with every frame |
| `grasp_scan/` | `active_perception_wbc` (`--save_dir`), VGN and Contact-GraspNet | per scan and view `scanNN_viewI.jpg` (target mask, all grasps, chosen one green), `.npz` (rgb, depth, K, camera pose, mask, grasps), `scanNN_grasps.csv` |

Off: `save_images:=false` (launch), `-p save_dir:=''` (detectors), `--save_dir ''` (WBC).

## Performance (RTX 3070 laptop, 8 GB; 640x480)

| | time | GPU memory |
|---|---|---|
| `gsam` segment (fp16 autocast): GroundingDINO ~85 ms + SAM encoder ~120 ms | ~220 ms/frame → detector ~4.5 Hz | ~2.7 GB |
| `cgn` one view, target segment, 1 forward pass | ~1.0–1.2 s | ~2.5 GB during a request, freed after |

GroundingDINO's deformable attention runs as its PyTorch fallback (no `nvcc` to build the CUDA op;
`gsam.py`). Contact-GraspNet with 3 forward passes does not fit next to `gsam` in 8 GB.

Tested on the grasp-test snapshot (`~/ros_logs/grasp_test/20261014_145910`: ball next to a mug and
headphones): "red ball" 0.92, mask on the ball only; the detector's sphere-fit centre within 0.6 cm
of the mask's; Contact-GraspNet 5–10 grasps on the ball, none on the clutter, fingertip centre
~2 cm above the ball centre, approach from above.

## Testing next to the robot

`~/.ros/fastdds_steve.xml` makes every participant on this PC contact the robot by unicast,
**whatever `ROS_DOMAIN_ID` is set** — a test replaying recorded topics would reach the robot's
nodes. For offline tests drop the profile:

```bash
unset FASTRTPS_DEFAULT_PROFILES_FILE; export ROS_DOMAIN_ID=99 ROS_LOCALHOST_ONLY=1
ros2 node list --no-daemon      # must show none of the robot's nodes
```
