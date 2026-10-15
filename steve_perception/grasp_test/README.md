# Hand-held grasp test (no robot)

`handheld_grasp_test.py`: a hand-held RealSense (D415 / D435i over USB) stands in for the wrist camera.
One RGB-D snapshot → grasp detector → grasp in the robot's `odom` frame → **9-DOF WBC trajectory**
(base x, y, yaw + 6 UR5e joints; steve_wbc's strict-hierarchy controller, kinematic rollout with
table collision constraints). With `--live`, RViz shows the camera, its point cloud, the grasp and the
Steve model replaying the trajectory.

```
grasp_test/
├── handheld_grasp_test.py   snapshot → anchor → VGN / Contact-GraspNet → WBC rollout → files
├── rviz_live.py             --live: live camera pose + cloud, Enter = snapshot, animated robot
└── cgn_client.py            Contact-GraspNet client (DiNBV-Grasp's ZMQ protocol)
launch/grasp_test_rviz.launch.py   robot_state_publisher (Steve URDF) + RViz (rviz/grasp_test.rviz)
```

## Run

```bash
# terminal 1 — camera (namespace is found automatically, --camera to pick one)
ros2 launch realsense2_camera rs_launch.py align_depth.enable:=true
# terminal 2 — RViz + Steve model
ros2 launch steve_perception grasp_test_rviz.launch.py
# terminal 3 — live test: Enter = snapshot -> grasp + trajectory, q + Enter = quit
ros2 run steve_perception handheld_grasp_test --live
```

Other modes:

```bash
ros2 run steve_perception handheld_grasp_test                       # one snapshot, terminal only
ros2 run steve_perception handheld_grasp_test --backend cgn         # Contact-GraspNet (see below)
ros2 run steve_perception handheld_grasp_test --from_npz ~/ros_logs/grasp_test/<run>/snapshot.npz \
    --orientation full --viewer                                     # offline re-run, MuJoCo replay
MUJOCO_GL=egl ros2 run steve_perception handheld_grasp_test --synthetic          # no camera
MUJOCO_GL=egl ros2 run steve_perception handheld_grasp_test --synthetic --live   # RViz, no camera
```

The test publishes `/joint_states` and TF `odom → base_link` itself: do not run it on the same ROS
domain as the real robot's bringup (`export ROS_DOMAIN_ID=42` in all terminals, camera included).

## Holding the camera

* **0.4–0.6 m above the object**, looking down (tilt < ~30°), object near the image centre,
  **image-top pointing away from you** (= robot forward).
* D435i: no depth closer than ~0.28 m (1280×720). D415: ~0.45 m at 1280×720, ~0.31 m at 848×480
  (`depth_module.depth_profile:=848x480x30 rgb_camera.color_profile:=848x480x30`), so hold it at
  0.5–0.6 m. A camera on USB 2 runs at 640×480 @ 15 Hz, which is enough for this test.
* Object: 2–8 cm wide (VGN's gripper opens 8 cm), 1–20 cm tall, nothing else within ~15 cm.
  Black, shiny or transparent objects give no RealSense depth.

## How the hand-held camera maps onto the robot TF

There is no TF chain from a camera in your hand, and its position cannot be measured from one
image. The script therefore **anchors the robot frame on the scene**:

| Quantity | From |
|---|---|
| up, camera tilt, camera height | the table plane in view (RANSAC) + `top_z` of the `--table` model |
| heading | **image-top = robot forward** (true for the wrist camera at the scan pose) |
| x / y | the object seen is placed at `--object_xy` (default: table near half, 25 cm in) |

The table model is a **virtual** table in front of a virtual robot at `--base0` (default `0 0 0`,
arm at the scan pose); any flat desk works. What this tests: the grasp **relative to the object**
(position, approach, finger axis, width) and whether the WBC reaches it. What it does not test:
the absolute object position (given here, from the robot's TF on the real robot).

## RViz (`--live`, fixed frame `odom`)

| Display | Topic | Shows |
|---|---|---|
| Steve | `/robot_description` + `/joint_states` | the robot; replays the 9-DOF trajectory after a snapshot |
| Live cloud | `/grasp_test/live_cloud` | the camera's cloud, re-anchored at `--live_hz` (5 Hz) |
| Live camera | `/grasp_test/live_markers` | frustum, detected object, "camera X m above table" / "searching: …" |
| Grasp + trajectory | `/grasp_test/markers` | table, object, gripper at grasp (green) / pre-grasp + lift (blue), approach arrow, EE path (yellow), base path (blue), result text |
| Snapshot cloud (off) | `/grasp_test/snapshot_cloud` | exactly what the grasp detector saw |
| Grasp pose | `/grasp_test/grasp_pose` | pinch point + WBC goal orientation |

The robot model maps the WBC's MuJoCo joint angles to the URDF with
`shoulder_pan_urdf = q1 − (UR mount yaw)`, the yaw read from `/robot_description` (joint
`ur5ebase_joint`; `--viz_pan` overrides). With the robot's URDF (`steve_simulation` `origin/test`,
mount +90°, as on the real robot and in the SRDF) that is `q1 − π/2`, the WBC's `--pan_offset`; an
older `steve_simulation` checkout (mount −90°) gives `q1 + π/2`. Checked with
robot_state_publisher: the animated URDF gripper passes within 1 mm of the computed grasp.

## Outputs (`~/ros_logs/grasp_test/<time>/`)

* `grasp.json`: same keys as `steve_wbc vgn_grasp_real`, so on the robot
  `steve_figure8_wbc --mode ros --goto_file grasp.json` executes it.
* `trajectory.csv`: 50 Hz; `base_x, base_y, base_yaw, q1..q6` (MuJoCo convention) plus `q1_ros`
  (shoulder_pan − π/2, the WBC's real-robot convention), the 9 velocity commands, and the EE
  reference and actual pose. Phases are `pregrasp → grasp → lift`.
* `trajectory.png`, `snapshot.npz`.

## Backends

**VGN** (`--backend vgn`, `steve_perception.vgn`, `vgn_env`): the 30 cm TSDF is placed **as in
VGN's training scenes, with the table plane 5 cm above the volume floor**, centred on the
object in x/y. VGN's grasp frame equals the WBC `pinch` site convention (z = approach,
y = finger axis), so `--orientation full` passes its 6-DOF rotation through unchanged.
The ~4 s per call is mostly subprocess start and model load. The network itself is ~10 ms on the
GPU, so a persistent worker would make it real-time.

**Contact-GraspNet / DiNBV** (`--backend cgn`): speaks DiNBV-Grasp's ZMQ protocol
(`grasp_executor.py`: pickled `{depth, K, segmap, rgb, z_range, forward_passes}` →
`{pred_grasps_cam, scores, contact_pts}`). The object mask from the plane segmentation is sent
as `segmap`. The server is steve_perception's `cgn` model server (Contact-GraspNet PyTorch port in
`cgn_env`, port 5555): `ros2 run steve_perception model_server cgn`, see
[`model_server/README.md`](../model_server/README.md).

The rest of DiNBV (Stage I view selection: VGGT + distilled student; Stage II distance scorer)
cannot run here. Its `student_nbv/best.pt` and `local_distance_scorer/best.pt` checkpoints are
not released ("train or export"), and the full stack (VGGT, SAM-H, GroundingDINO, Qwen-VL 8B)
targets ≥ 24 GB of VRAM. DiNBV is a view planner on top of a grasp detector, not a grasp
detector itself. On the robot, exploring other views when no grasp is found is done by
`steve_wbc active_perception_wbc --vgn_explore` (around the table, see its README).
