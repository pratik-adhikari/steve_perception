# Steve Perception System Documentation

This document explains the design, configuration, and debugging workflow of the `steve_perception` package.

**Goal:** provide a single, synchronized `rtabmap_msgs/msg/RGBDImage` per camera for RTAB-Map and mapping.

**Non-negotiable design rule:** `CameraAgent` is the **only** place where RGB + Depth + CameraInfo are synchronized.

---

## Naming

- `<cam_ns>`: the camera driver namespace/prefix (e.g., `/wrist_camera/wrist_camera`)
- `<camera_name>`: the logical camera name used by `steve_perception` (e.g., `wrist`, `front`, `pan_tilt`)
- Unified output topic: `/<camera_name>/rgbd_image`

---

## 1) Overview

`CameraAgent` (`steve_perception/core/camera_agent.py`) subscribes to raw camera streams:

- RGB: `sensor_msgs/msg/Image`
- Depth: `sensor_msgs/msg/Image`
- CameraInfo: `sensor_msgs/msg/CameraInfo`

It time-synchronizes them and publishes:

- `/<camera_name>/rgbd_image` (`rtabmap_msgs/msg/RGBDImage`)

Downstream nodes (RTAB-Map, mapping, nav) **must not** subscribe to raw topics.

---

## 2) Architecture and Data Flow

```
Gazebo / Hardware Drivers
├── /<cam_ns>/image_raw                 (sensor_msgs/msg/Image)
├── /<cam_ns>/depth/image_raw           (sensor_msgs/msg/Image)
└── /<cam_ns>/camera_info               (sensor_msgs/msg/CameraInfo)
|
v
CameraAgent
(ApproximateTimeSynchronizer)
|
v
/<camera_name>/rgbd_image         (rtabmap_msgs/msg/RGBDImage)
|
v
RTAB-Map / Mapping / Navigation
```

---

## 3) Synchronization Policy

All temporal synchronization is performed **exclusively** inside `CameraAgent`.

- Mechanism: `message_filters::ApproximateTimeSynchronizer`
- Why approximate:
  - simulation jitter
  - real sensors rarely stamp perfectly aligned topics

**Benefit of central sync:** one place to control and debug timing behavior; no duplicated sync logic in launch files.

---

## 4) RGBDImage Contract

`CameraAgent` outputs `rtabmap_msgs/msg/RGBDImage` with these expectations:

### 4.1 Header consistency
- `rgb.header.stamp == depth.header.stamp`
- `rgb.header.frame_id == depth.header.frame_id`

### 4.2 Intrinsics source
- Camera intrinsics are taken from the synchronized `CameraInfo`.

### 4.3 Depth units (important!)
- The pipeline **expects depth in meters**.
- **Confirmed (Sim):** Simulation publishes `32FC1` (float meters). `CameraAgent` passes this through.
- RTAB-Map default `depth_scale` (1.0) is used.
- If the source publishes scaled integer depth (e.g., `16UC1`), conversion must happen **before or inside** `CameraAgent`.

### 4.4 RGB encoding
- RGB encoding should be stable per camera (`rgb8` or `bgr8`).

---

## 5) TF Requirements

`CameraAgent` does TF lookup at the **synchronized message timestamp**.

Each camera optical frame must be connected to `base_link`:

```
base_link  ->  <camera_optical_frame>
```

If this transform is missing/disconnected, RGBD output will be invalid (or dropped).

---

## 6) Launch Workflow (Simulation)

Run the stack in three terminals:

### Terminal 1: Simulation Base
```bash
ros2 launch neo_nav2_bringup slam_simulation.launch.py
```

### Terminal 2: Perception Layer (CameraAgents)

```bash
ros2 launch steve_perception perception.launch.py
```

Expected outputs:

* `/wrist/rgbd_image`
* `/pan_tilt/rgbd_image`
* `/front/rgbd_image`

### Terminal 3: SLAM / Mapping (RTAB-Map)

```bash
ros2 launch steve_perception rgbd2_rtab.launch.py
```

---

## 7) Camera Configuration (`config/steve.yaml`)

Camera profiles map raw topics into unified outputs.

### 7.1 Wrist Camera (`wrist`)

* Output: `/wrist/rgbd_image`

| Stream | Topic                                        | Expected Frame ID                  | Type                         |
| ------ | -------------------------------------------- | ---------------------------------- | ---------------------------- |
| RGB    | `/wrist_camera/wrist_camera/image_raw`       | `wrist_camera_color_optical_frame` | `sensor_msgs/msg/Image`      |
| Depth  | `/wrist_camera/wrist_camera/depth/image_raw` | `wrist_camera_color_optical_frame` | `sensor_msgs/msg/Image`      |
| Info   | `/wrist_camera/wrist_camera/camera_info`     | `wrist_camera_color_optical_frame` | `sensor_msgs/msg/CameraInfo` |

### 7.2 Pan-Tilt Camera (`pan_tilt`)

* Output: `/pan_tilt/rgbd_image`

| Stream | Topic                                              | Expected Frame ID               | Type                         |
| ------ | -------------------------------------------------- | ------------------------------- | ---------------------------- |
| RGB    | `/pan_tilt_camera/pan_tilt_camera/image_raw`       | `pan_tilt_camera_optical_frame` | `sensor_msgs/msg/Image`      |
| Depth  | `/pan_tilt_camera/pan_tilt_camera/depth/image_raw` | `pan_tilt_camera_optical_frame` | `sensor_msgs/msg/Image`      |
| Info   | `/pan_tilt_camera/pan_tilt_camera/camera_info`     | `pan_tilt_camera_optical_frame` | `sensor_msgs/msg/CameraInfo` |

### 7.3 Front Camera (`front`)

* Output: `/front/rgbd_image`

| Stream | Topic                                        | Expected Frame ID            | Type                         |
| ------ | -------------------------------------------- | ---------------------------- | ---------------------------- |
| RGB    | `/front_camera/front_camera/image_raw`       | `front_camera_optical_frame` | `sensor_msgs/msg/Image`      |
| Depth  | `/front_camera/front_camera/depth/image_raw` | `front_camera_optical_frame` | `sensor_msgs/msg/Image`      |
| Info   | `/front_camera/front_camera/camera_info`     | `front_camera_optical_frame` | `sensor_msgs/msg/CameraInfo` |

---

## 8) Debug Playbook (Run This First)

### 8.1 Are raw topics alive?

```bash
ros2 topic hz /wrist_camera/wrist_camera/image_raw
ros2 topic hz /wrist_camera/wrist_camera/depth/image_raw
ros2 topic hz /wrist_camera/wrist_camera/camera_info
```

### 8.2 Is the unified RGBD output alive?

```bash
ros2 topic hz /wrist/rgbd_image
```

Expected: close to the slowest of {RGB, Depth, CameraInfo}.

### 8.3 Check synchronization (RGB vs Depth stamps)

```bash
ros2 topic echo --once /wrist/rgbd_image | egrep "rgb:|depth:|stamp:" -n
```

Expected: the `rgb.header.stamp` and `depth.header.stamp` lines match exactly.

### 8.4 Is TF connected?

```bash
ros2 run tf2_ros tf2_echo base_link wrist_camera_color_optical_frame
```

Expected: continuous transform output (no “frame does not exist”).

### 8.5 Verify message encodings (RGB + depth)

```bash
ros2 topic echo --once /wrist_camera/wrist_camera/image_raw | grep encoding
ros2 topic echo --once /wrist_camera/wrist_camera/depth/image_raw | grep encoding
```

### 8.6 Is sim time consistent? (Simulation)

```bash
ros2 param get /perception_node use_sim_time
ros2 param get /rtabmap use_sim_time
```

Expected: `true` for all timestamp-sensitive nodes in Gazebo.

---

## 9) Failure → Likely cause → First fix

| Symptom | Likely cause | First command | First fix |
| :--- | :--- | :--- | :--- |
| `/wrist/rgbd_image` not publishing | raw topics missing | `ros2 topic list \| grep wrist_camera` | start sim/driver |
| `/wrist/rgbd_image` very low Hz | `camera_info` slow | `ros2 topic hz /wrist_camera/wrist_camera/camera_info` | increase info rate / remove from sync (only if intentional) |
| `base_link` missing in TF | robot_state_publisher not running / namespace mismatch | `ros2 topic echo /tf_static --once` | fix robot_state_publisher / namespace |
| TF exists but mapping bad | wrong frame_id / wrong optical frame | `tf2_echo base_link <optical>` | fix frame_id and TF tree |

---

## 10) Known-good reference (Simulation)

Recorded on 2025-12-16 (Sim):

* **rgbd rate**: `~14 Hz` (varies 5-18 Hz depending on load, e.g., drops when RViz is active)
* **depth encoding**: `32FC1` (Float32 Meters)
* **camera_info frame_id**: `wrist_camera_color_optical_frame` (matches config)

---

## 11) CameraAgent Parameters

These parameters can be added to **`config/steve.yaml`** under each camera profile to override defaults.

- `sync_slop` (float, default **0.1s**): Max time difference between messages to sync.
- `queue_size` (int, default **20**): Number of messages to buffer for syncing.
- `use_camera_info` (bool): Implicitly `True` by design (syncs RGB+Depth+Info).
- `publish_tf` (bool): Defined in launch/node, usually `False` for `CameraAgent` (robot_state_publisher handles TF).
