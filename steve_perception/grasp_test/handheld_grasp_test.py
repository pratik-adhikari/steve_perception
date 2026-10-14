#!/usr/bin/env python3
"""
handheld_grasp_test.py
======================
Grasp-perception test WITHOUT the robot: a hand-held RealSense (D415 / D435i over USB) plays
the wrist camera. One RGB-D snapshot -> grasp detector -> grasp pose in the robot's odom frame ->
9-DOF whole-body trajectory (steve_wbc, kinematic rollout: base x, y, yaw + 6 UR5e joints).

Grasp backends (--backend)
  vgn   VGN (steve_perception.vgn, vgn_env): TSDF of the snapshot -> 6-DOF grasps. The 30 cm TSDF
        is placed like VGN's training scenes: the table plane 5 cm above the volume's floor.
  cgn   Contact-GraspNet over the DiNBV-Grasp server protocol (cgn_client.py) — the grasp
        generator DiNBV runs from its selected view. DiNBV's view planning itself (VGGT + student
        + distance scorer) needs checkpoints that are not released; see grasp_test/README.md.

How the hand-held camera is tied to the robot TF (no TF tree, no marker)
  The camera's position is not observable, so the robot frame is ANCHORED on the scene:
    * up + height : the dominant plane in view is the table top (--table model, top_z)
    * heading     : image-top points along the robot's forward axis (the wrist camera at the
                    scan pose looks the same way: image-top = away from the base)
    * x / y       : the object seen is placed at --object_xy (default: on the table's near half)
  i.e. "the robot stands at --base0 and the object lies at --object_xy; you hold the camera where
  the wrist camera would be". The grasp relative to the object is measured; the object position is
  given. Hold the camera 0.4-0.6 m above the object (D415/D435i see no depth closer than ~0.3 m),
  looking down, image-top away from you.

Outputs (--out_dir, default ~/ros_logs/grasp_test/<time>/)
  snapshot.npz        depth, rgb, K (re-run offline: --from_npz)
  grasp.json          pinch, pregrasp, quaternion, score, width, object, camera pose (odom)
                      — same keys as steve_wbc vgn_grasp_real, so `steve_figure8_wbc --goto_file`
                      takes it on the real robot
  trajectory.csv      9-DOF trajectory at --rate Hz: t, phase, base x/y/yaw, q1..q6 (MuJoCo model
                      convention) + q1_ros (shoulder_pan in the URDF / UR driver convention),
                      9 velocity commands, EE reference and actual pose
  trajectory.png      base path, EE path, joint plots

Run
  ros2 launch realsense2_camera rs_launch.py align_depth.enable:=true             # terminal 1
  ros2 run steve_perception handheld_grasp_test --backend vgn --table lab_table   # terminal 2
  ros2 run steve_perception handheld_grasp_test --from_npz <dir>/snapshot.npz --backend cgn
  ros2 run steve_perception handheld_grasp_test --synthetic                       # no camera
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parents[3]      # steve_ros2_ws/src (source-tree fallback)
try:
    from steve_wbc import steve_figure8_wbc as wbc
    from steve_wbc.table_model import TableModel, lidar_clusters
    from steve_wbc.vgn_grasp_real import R_to_quat_wxyz
except ImportError:  # steve_wbc not built / not sourced: the source tree
    sys.path.insert(0, str(_SRC / "steve_wbc" / "steve_wbc"))
    import steve_figure8_wbc as wbc
    from table_model import TableModel, lidar_clusters
    from vgn_grasp_real import R_to_quat_wxyz
try:
    from steve_perception.grasp_test import cgn_client
    from steve_perception.vgn import VGNBridge
except ImportError:
    sys.path.insert(0, str(_SRC / "steve_perception"))
    from steve_perception.grasp_test import cgn_client
    from steve_perception.vgn import VGNBridge

SCAN_Q = [0.0, -1.8, 1.4, -1.2, -1.57, 0.0]   # active_perception_wbc SCAN_SEED_Q: wrist camera looks down
VGN_TABLE_IN_TSDF = 0.05                    # m, table plane above the TSDF floor in VGN's training scenes
VGN_FINGER_DEPTH = 0.05                     # m, VGN grasp origin -> fingertip centre along the approach
LIFT = 0.15                                 # m


class GraspTestError(RuntimeError):
    """A snapshot that yields no grasp (live mode keeps running, single-shot mode exits)."""


# ══════════════════════════════════════════════════════════════════════════════
# Camera input
# ══════════════════════════════════════════════════════════════════════════════
def find_camera_ns(node, camera, timeout):
    """`camera`, or with 'auto' the namespace of the published */aligned_depth_to_color/camera_info
    (it differs between launch files: /camera/camera, /camera, ...)."""
    if camera != "auto":
        return camera
    import rclpy
    info_sfx, found, t0 = "/aligned_depth_to_color/camera_info", [], time.monotonic()
    while not found and time.monotonic() - t0 < min(timeout, 5.0):
        rclpy.spin_once(node, timeout_sec=0.2)
        found = sorted(t for t, _ in node.get_topic_names_and_types()
                       if t.endswith(info_sfx) and node.count_publishers(t) > 0)
    if not found:
        sys.exit("[ERROR] no published */aligned_depth_to_color/camera_info — is the RealSense node running "
                 "with align_depth.enable:=true? (ros2 topic list | grep aligned)")
    camera = found[0][:-len(info_sfx)]
    print(f"camera: {camera}" + (f"  (also found {[f[:-len(info_sfx)] for f in found[1:]]}; "
                                 "pick one with --camera)" if len(found) > 1 else ""))
    return camera


def depth_from_msg(m):
    """sensor_msgs/Image depth -> float32 metres."""
    if m.encoding in ("16UC1", "mono16"):
        return np.frombuffer(m.data, np.uint16).reshape(m.height, m.width).astype(np.float32) / 1000.0
    return np.frombuffer(m.data, np.float32).reshape(m.height, m.width).copy()


def rgb_from_msg(m):
    """sensor_msgs/Image rgb8 / bgr8 -> HxWx3 RGB uint8 (None for other encodings)."""
    if m.encoding not in ("rgb8", "bgr8"):
        return None
    im = np.frombuffer(m.data, np.uint8).reshape(m.height, m.width, 3)
    return im[..., ::-1].copy() if m.encoding == "bgr8" else im.copy()


def capture_ros(camera, n_frames, timeout):
    """n aligned depth frames [m] + one colour frame + K from realsense2_camera. Press Enter to
    take the snapshot (hold the camera still)."""
    import rclpy
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import CameraInfo, Image

    rclpy.init()
    n = rclpy.create_node("handheld_grasp_test")
    camera = find_camera_ns(n, camera, timeout)
    st = {"K": None, "depth": [], "rgb": None, "armed": False}
    n.create_subscription(CameraInfo, f"{camera}/aligned_depth_to_color/camera_info",
                          lambda m: st.__setitem__("K", np.array(m.k).reshape(3, 3)), 10)

    def depth_cb(m):
        if st["armed"] and len(st["depth"]) < n_frames:
            st["depth"].append(depth_from_msg(m))

    def rgb_cb(m):
        if st["armed"] and st["rgb"] is None:
            st["rgb"] = rgb_from_msg(m)

    n.create_subscription(Image, f"{camera}/aligned_depth_to_color/image_raw", depth_cb, qos_profile_sensor_data)
    n.create_subscription(Image, f"{camera}/color/image_raw", rgb_cb, qos_profile_sensor_data)
    t0 = time.monotonic()
    while st["K"] is None and time.monotonic() - t0 < timeout:
        rclpy.spin_once(n, timeout_sec=0.1)
    if st["K"] is None:
        sys.exit(f"[ERROR] no {camera}/aligned_depth_to_color/camera_info within {timeout:.0f} s — "
                 "rs_launch.py align_depth.enable:=true running? (--camera sets the namespace)")
    input("Camera ready. Hold it 0.4-0.6 m above the object, looking down, image-top away from you, "
          "then press Enter and keep still ... ")
    st["armed"] = True
    t0 = time.monotonic()
    while (len(st["depth"]) < n_frames or st["rgb"] is None) and time.monotonic() - t0 < timeout:
        rclpy.spin_once(n, timeout_sec=0.05)
    n.destroy_node()
    rclpy.try_shutdown()
    if not st["depth"]:
        sys.exit("[ERROR] no depth frames")
    return st["depth"], st["rgb"], st["K"]


def capture_synthetic(table, obj_xy, height=0.50, tilt_deg=12.0, size=(480, 640), noise=0.0015,
                      n_frames=3, seed=0):
    """MuJoCo stand-in for the hand-held camera: the sim ball placed at obj_xy on the table, a free
    camera `height` above the table top looking down (tilted `tilt_deg` back toward the robot, image-
    top along +x). Returns depth frames, rgb, K and the TRUE world<-optical pose (for checking)."""
    import mujoco
    model = wbc.load_model(wbc.resolve_model(None), spec_edit=table.apply_to_spec)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    j = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "box_joint")]
    r = float(model.geom_size[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "box_geom")][0])
    data.qpos[j:j + 7] = [*obj_xy, table.top_z + r, 1, 0, 0, 0]
    mujoco.mj_forward(model, data)

    H, W = size
    model.vis.global_.offwidth, model.vis.global_.offheight = max(W, 640), max(H, 480)
    ren = mujoco.Renderer(model, H, W)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [*obj_xy, table.top_z]
    cam.distance = height / np.cos(np.radians(tilt_deg))
    cam.azimuth, cam.elevation = 0.0, -(90.0 - tilt_deg)     # camera behind (-x) the object, image-top +x
    ren.update_scene(data, camera=cam)
    gl = ren.scene.camera[0]
    fwd, up = np.array(gl.forward), np.array(gl.up)
    R_wo = np.column_stack([np.cross(fwd, up), -up, fwd])    # optical: x right, y down, z forward
    t_wo = 0.5 * (np.array(gl.pos) + np.array(ren.scene.camera[1].pos))   # mono image: between the stereo eyes
    ren.enable_depth_rendering()
    d0 = ren.render().astype(np.float32)
    ren.disable_depth_rendering()
    rgb = ren.render().copy()
    ren.close()
    f = 0.5 * H / np.tan(np.radians(model.vis.global_.fovy) / 2)
    K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1.0]])
    rng = np.random.default_rng(seed)
    frames = [np.where(d0 < 3.0, d0 + rng.normal(0, noise, d0.shape) * d0 ** 2 / 0.25, 0).astype(np.float32)
              for _ in range(n_frames)]   # D4xx-like depth noise ~ z^2
    ball = np.array([*obj_xy, table.top_z + r])
    return frames, rgb, K, (R_wo, t_wo, ball)


# ══════════════════════════════════════════════════════════════════════════════
# Anchoring the hand-held camera in the robot frame
# ══════════════════════════════════════════════════════════════════════════════
def backproject(D, K, step=2, zmin=0.15, zmax=2.0):
    """Optical-frame points (N x 3) and their pixel (v, u)."""
    H, W = D.shape
    v, u = np.mgrid[0:H:step, 0:W:step]
    z = D[v, u]
    ok = (z > zmin) & (z < zmax) & np.isfinite(z)
    P = np.stack([(u[ok] - K[0, 2]) * z[ok] / K[0, 0], (v[ok] - K[1, 2]) * z[ok] / K[1, 1], z[ok]], 1)
    return P, v[ok], u[ok]


def fit_plane(P, central, thr=0.008, iters=400, seed=0):
    """RANSAC plane through the central image region (the table under the object), refined by least
    squares. Returns unit normal n (toward the camera) and offset d with n . p = d on the plane."""
    rng = np.random.default_rng(seed)
    Q = P[central]
    if len(Q) < 500:
        raise ValueError(f"only {len(Q)} valid depth points in the image centre — too close / too far?")
    best, best_n = None, -1
    for _ in range(iters):
        a, b, c = Q[rng.choice(len(Q), 3, replace=False)]
        n = np.cross(b - a, c - a)
        if np.linalg.norm(n) < 1e-9:
            continue
        n /= np.linalg.norm(n)
        inl = np.abs((Q - a) @ n) < thr
        if inl.sum() > best_n:
            best, best_n = inl, inl.sum()
    S = Q[best]
    for _ in range(3):  # refine: least-squares plane (smallest covariance eigenvector), re-select inliers
        mu = S.mean(0)
        n = np.linalg.eigh(np.cov((S - mu).T))[1][:, 0]
        S = Q[np.abs((Q - mu) @ n) < thr]
    if n @ mu > 0:      # camera (origin) on the + side: n points up, toward the camera
        n = -n
    return n, float(n @ mu), len(S) / len(Q)


def anchor_camera(D, K, table, obj_xy, yaw0, h_obj=(0.01, 0.25), max_ext=0.20, step=2, iters=400):
    """World(odom) <- optical pose of the hand-held camera from the scene (see the module doc).
    Returns R_wc, t_wc, object centre (world), object top z, info dict (incl. the world cloud).
    step / iters: pixel subsampling and RANSAC iterations (coarser for the live preview)."""
    P, v, u = backproject(D, K, step=step)
    H, W = D.shape
    central = (np.abs(v - H / 2) < 0.35 * H) & (np.abs(u - W / 2) < 0.35 * W)
    n_c, d_c, frac = fit_plane(P, central, iters=iters)
    h_cam = -d_c                              # camera height above the plane
    # leveled frame L (camera at origin): z = plane normal, x = image-top projected onto the plane
    fwd = np.array([0.0, -1.0, 0.0]) - (-n_c[1]) * n_c      # -y_c minus its normal component
    if np.linalg.norm(fwd) < 0.2:
        raise ValueError("camera looks along the table (image-top ~ normal) — look down at the object")
    x_L = fwd / np.linalg.norm(fwd)
    R_Lc = np.vstack([x_L, np.cross(n_c, x_L), n_c])        # rows: L axes in camera coordinates
    R_wc = wbc.rot_z(yaw0) @ R_Lc
    Pw = P @ R_wc.T                                          # rotated, camera at the origin
    hz = Pw[:, 2] + h_cam                                    # height above the table plane
    m = (hz > h_obj[0]) & (hz < h_obj[1])
    if m.sum() < max(30, 120 // step ** 2):
        raise ValueError(f"only {m.sum()} depth points above the table plane — object in view?")
    pts = Pw[m]
    # the object = compact cluster closest to where the optical axis meets the table
    ray = R_wc[:, 2]
    aim = ray * (h_cam / max(-ray[2], 1e-3))
    best = None
    for c in lidar_clusters(pts[:, :2], link=0.0075 * max(step, 2)):
        if len(c) < max(5, 80 // step ** 2) or np.ptp(c, axis=0).max() > max_ext:
            continue
        dist = np.linalg.norm(c.mean(0) - aim[:2])
        if best is None or dist < best[0]:
            best = (dist, c)
    if best is None:
        raise ValueError(f"no compact object (< {max_ext * 100:.0f} cm, >= 20 points) on the plane")
    xy = best[1].mean(0)
    r = 0.5 * np.ptp(best[1], axis=0).max() + 0.01
    sel = np.hypot(*(pts[:, :2] - xy).T) < r
    top = float(np.percentile(hz[m][sel], 95))
    t_wc = np.array([obj_xy[0] - xy[0], obj_xy[1] - xy[1], table.top_z + h_cam])
    obj = np.array([obj_xy[0], obj_xy[1], table.top_z + 0.5 * top])
    tilt = np.degrees(np.arccos(np.clip(R_wc[:, 2] @ [0, 0, -1], -1, 1)))
    # pixel mask of the object (for Contact-GraspNet's segmap)
    mask = np.zeros(D.shape, np.int32)
    vv, uu = v[m][sel], u[m][sel]
    for dv in (0, 1):
        for du in (0, 1):
            mask[np.minimum(vv + dv, H - 1), np.minimum(uu + du, W - 1)] = 1
    return R_wc, t_wc, obj, table.top_z + top, dict(h_cam=h_cam, plane_frac=frac, tilt=tilt,
                                                    n_obj=int(sel.sum()), mask=mask, cloud=Pw + t_wc,
                                                    pix=(v, u), obj_xy_extent=float(2 * r - 0.02))


def default_object_xy(table, inset=0.25):
    """On the table's near (-x, robot-facing) half: `inset` in from the near edge, at its centre line."""
    return np.asarray(table.center) + table.R() @ np.array([-0.5 * table.size[0] + inset, 0.0])


# ══════════════════════════════════════════════════════════════════════════════
# Grasp backends -> (pinch, R_pinch(world), width, score, raw info)
# ══════════════════════════════════════════════════════════════════════════════
def grasp_vgn(frames, K, R_wc, t_wc, obj, table, near):
    vgn = VGNBridge()
    if not vgn.available:
        raise GraspTestError("VGN not available (vgn_env / weights) — see steve_perception/vgn/README.md")
    # 30 cm cube centred on the object in x/y; table plane 5 cm above its floor (as VGN was trained)
    centre = np.array([obj[0], obj[1], table.top_z - VGN_TABLE_IN_TSDF + 0.5 * vgn.tsdf_size])
    for d in frames:
        vgn.add_real_view(d, t_wc, R_wc, K, centre)
    t0 = time.monotonic()
    if not vgn.run_inference():
        raise GraspTestError("VGN found no grasp")
    dt = time.monotonic() - t0
    g = vgn.best_grasp_world(near=obj, xy_radius=near, z_window=0.10)
    if g is None:
        raise GraspTestError(f"no VGN grasp within {near * 100:.0f} cm of the object")
    pos, rot, width, score = g
    # VGN grasp frame = WBC pinch convention (z approach, y finger axis); origin -> fingertip centre
    return pos + VGN_FINGER_DEPTH * rot[:, 2], rot, width, score, dict(backend="vgn", vgn_pos=pos.tolist(),
                                                                      infer_s=dt, n_candidates=vgn._n_candidates)


def grasp_cgn(frames, rgb, K, R_wc, t_wc, obj, mask, near, host, port, tcp_depth):
    D = np.median(np.stack(frames), axis=0)
    t0 = time.monotonic()
    try:
        cands = cgn_client.get_grasps(D, K, segmap=mask, rgb=rgb, host=host, port=port)
    except RuntimeError as e:
        raise GraspTestError(str(e)) from e
    dt = time.monotonic() - t0
    T_wc = np.eye(4)
    T_wc[:3, :3], T_wc[:3, 3] = R_wc, t_wc
    for T, score, width in cands:
        pinch, R = cgn_client.to_pinch_frame(T_wc @ T, tcp_depth)
        if np.hypot(*(pinch[:2] - obj[:2])) <= near:
            return pinch, R, width if width is not None else 0.08, score, dict(
                backend="cgn", infer_s=dt, n_candidates=len(cands))
    raise GraspTestError(f"none of {len(cands)} Contact-GraspNet grasps within {near * 100:.0f} cm of the object")


def goal_orientation(R_grasp, mode, yaw0):
    """Gripper orientation the WBC tracks. topdown: gripper straight down, turned about z to the
    grasp's finger axis (as vgn_grasp_real); full: the detector's 6-DOF rotation. The parallel
    gripper is symmetric about its approach axis: of the two equivalent rotations the one closer to
    the base heading is used (no unnecessary half-turn of wrist 3)."""
    if mode == "topdown":
        ax = R_grasp[:, 1]
        cands = [wbc.rot_z(np.arctan2(ax[1], ax[0]) - np.pi / 2 + k * np.pi) @ wbc.R_TOPDOWN for k in (0, 1)]
    else:
        cands = [R_grasp, R_grasp @ np.diag([-1.0, -1.0, 1.0])]
    ref = wbc.rot_z(yaw0) @ wbc.R_TOPDOWN
    return min(cands, key=lambda R: np.linalg.norm(wbc.log_so3(R @ ref.T)))


# ══════════════════════════════════════════════════════════════════════════════
# 9-DOF whole-body trajectory (steve_wbc controller, kinematic rollout)
# ══════════════════════════════════════════════════════════════════════════════
def wbc_rollout(table, base0, q0, waypoints, names, a):
    """Roll the strict-hierarchy WBC (EE > posture, collision constraints incl. the table) out on the
    kinematic backend through the EE waypoints. Returns rows (dicts) and a status string."""
    import mujoco
    wargs = wbc.parse_args(["--mode", "kin", "--table", a.table, "--rate", str(a.rate), "--max_base_v", str(a.max_base_v),
                            "--max_arm_vel", str(a.max_arm_vel)])
    wargs.table_model, wargs.collision_env = table, "model"
    model = wbc.load_model(wbc.resolve_model(None), spec_edit=table.apply_to_spec)
    kin = wbc.WholeBodyKinematics(model)
    j = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "box_joint")]
    kin.data.qpos[j:j + 3] = [50.0, 50.0, 0.0]   # park the MJCF's sim ball: not an obstacle here
    st0 = wbc.RobotState(np.asarray(base0, float), np.asarray(q0, float))
    kin.set_state(st0)
    p0, R0 = kin.ee_pose()
    ref_gen = wbc.GotoReference(p0, R0, waypoints, v_max=a.ee_speed, w_max=0.4, hold=1.0, names=names)
    ctrl = wbc.WholeBodyController(kin, wargs, q_nom=st0.q, yaw_nom=st0.base[2])
    dt = 1.0 / a.rate
    be = wbc.KinematicBackend(st0, dt)
    u_prev, rows, status = np.zeros(wbc.NU), [], "ok"
    while True:
        st = be.read_state()
        ref = ref_gen(be.now())
        if ref.done:
            break
        u, info = ctrl.step(st, ref, u_prev, dt)
        e = float(np.linalg.norm(info["e_p"]))
        rows.append(dict(t=be.now(), phase=ref.phase, base=st.base.copy(), q=st.q.copy(), u=u.copy(),
                         p_ref=ref.p.copy(), p=info["p"].copy(), quat=R_to_quat_wxyz(info["R"]),
                         e_p=e, e_o=float(np.linalg.norm(info["e_o"])), clr=float(info["d_min"])))
        if e > wargs.max_track_err:
            status = f"tracking error {e * 100:.1f} cm at t={be.now():.1f}s ({ref.phase}) — grasp out of reach?"
            break
        if be.now() > ref_gen.T_total + 5.0:
            status = "timeout"
            break
        be.send(u)
        u_prev = u
        be.tick()
    return rows, status, ctrl


def write_trajectory(rows, path, pan_offset):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "phase", "base_x", "base_y", "base_yaw", "q1", "q2", "q3", "q4", "q5", "q6", "q1_ros",
                    "u_vx", "u_vy", "u_wz", "u_dq1", "u_dq2", "u_dq3", "u_dq4", "u_dq5", "u_dq6",
                    "ee_ref_x", "ee_ref_y", "ee_ref_z", "ee_x", "ee_y", "ee_z", "ee_qw", "ee_qx", "ee_qy", "ee_qz",
                    "err_pos", "err_ori", "min_clearance"])
        for r in rows:
            w.writerow([f"{r['t']:.3f}", r["phase"], *np.round(r["base"], 5), *np.round(r["q"], 5),
                        round(r["q"][0] - pan_offset, 5), *np.round(r["u"], 5), *np.round(r["p_ref"], 5),
                        *np.round(r["p"], 5), *np.round(r["quat"], 5), round(r["e_p"], 5), round(r["e_o"], 5),
                        round(r["clr"], 4) if np.isfinite(r["clr"]) else ""])


def plot_trajectory(rows, table, obj, pinch, cam, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    t = np.array([r["t"] for r in rows])
    B = np.array([r["base"] for r in rows])
    Q = np.array([r["q"] for r in rows])
    P = np.array([r["p"] for r in rows])
    U = np.array([r["u"] for r in rows])
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    c, R, (sx, sy) = np.asarray(table.center), table.R(), table.size
    corners = np.array([c + R @ [s1 * sx / 2, s2 * sy / 2] for s1, s2 in ((-1, -1), (1, -1), (1, 1), (-1, 1), (-1, -1))])
    a = ax[0, 0]
    a.plot(*corners.T, "k-", lw=1, label="table")
    a.plot(B[:, 0], B[:, 1], "b-", label="base")
    a.plot(P[:, 0], P[:, 1], "g-", label="gripper (pinch)")
    a.quiver(B[::max(1, len(B) // 12), 0], B[::max(1, len(B) // 12), 1],
             np.cos(B[::max(1, len(B) // 12), 2]), np.sin(B[::max(1, len(B) // 12), 2]), color="b", scale=20, width=0.004)
    a.plot(*obj[:2], "ro", label="object")
    a.plot(*pinch[:2], "g*", ms=14, label="grasp")
    a.plot(*cam[:2], "m^", ms=10, label="hand-held camera")
    a.set_aspect("equal")
    a.set_title("top view (odom)")
    a.legend(fontsize=8)
    a = ax[0, 1]
    for i, n in enumerate("xyz"):
        a.plot(t, P[:, i], label=f"EE {n}")
    a.axhline(table.top_z, color="k", ls=":", lw=1)
    a.set_title("EE position [m]")
    a.legend(fontsize=8)
    a = ax[1, 0]
    a.plot(t, B[:, 0], label="base x")
    a.plot(t, B[:, 1], label="base y")
    a.plot(t, B[:, 2], label="base yaw [rad]")
    a.set_title("base DOF")
    a.legend(fontsize=8)
    a = ax[1, 1]
    for i in range(6):
        a.plot(t, Q[:, i], label=f"q{i + 1}")
    a.set_title("arm joints [rad] (MuJoCo convention)")
    a.legend(fontsize=8, ncol=3)
    for a in ax.flat[1:]:
        a.set_xlabel("t [s]")
        a.grid(alpha=0.3)
    ph = [r["phase"] for r in rows]
    for i in range(1, len(ph)):
        if ph[i] != ph[i - 1]:
            for a in ax.flat[1:]:
                a.axvline(t[i], color="grey", lw=0.6)
    fig.suptitle(f"9-DOF WBC trajectory — max |u| base {np.abs(U[:, :2]).max():.2f} m/s, "
                 f"arm {np.abs(U[:, 3:]).max():.2f} rad/s")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    return True


# ══════════════════════════════════════════════════════════════════════════════
def process_snapshot(frames, rgb, K, a, table, obj_xy, out, truth=None):
    """Snapshot -> anchored camera -> grasp -> 9-DOF WBC rollout; writes grasp.json, trajectory.csv/png
    and snapshot.npz to `out`. Raises GraspTestError when no grasp is found."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "snapshot.npz", depth=np.stack(frames), rgb=rgb if rgb is not None else np.zeros(0), K=K)
    D = np.median(np.stack(frames), axis=0)
    try:
        R_wc, t_wc, obj, obj_top, info = anchor_camera(D, K, table, obj_xy, a.base0[2])
    except ValueError as e:
        raise GraspTestError(str(e)) from e
    print(f"camera: {info['h_cam']:.3f} m above the table, tilted {info['tilt']:.1f} deg from straight down "
          f"(plane = {info['plane_frac'] * 100:.0f} % of the centre pixels) -> at {np.round(t_wc, 3)} odom")
    print(f"object: {info['n_obj']} points, top {obj_top - table.top_z:.3f} m above the table, centre {np.round(obj, 3)}")
    if info["h_cam"] < 0.30:
        print("  [WARN] closer than ~0.3 m: D415/D435i depth is missing or noisy there")
    if truth is not None:
        R_true, t_true, ball = truth
        print(f"  [synthetic] camera pose error {np.linalg.norm(t_wc - t_true) * 100:.2f} cm, "
              f"{np.degrees(np.linalg.norm(wbc.log_so3(R_wc @ R_true.T))):.2f} deg; object centre error "
              f"{np.linalg.norm(obj - ball) * 100:.2f} cm")

    if a.backend == "vgn":
        pinch, R_g, width, score, gi = grasp_vgn(frames, K, R_wc, t_wc, obj, table, a.near)
    else:
        pinch, R_g, width, score, gi = grasp_cgn(frames, rgb, K, R_wc, t_wc, obj, info["mask"], a.near,
                                                 a.cgn_host, a.cgn_port, a.cgn_tcp_depth)
    approach = R_g[:, 2]
    R_goal = goal_orientation(R_g, a.orientation, a.base0[2])
    app_goal = R_goal[:, 2]
    pregrasp = pinch - a.pregrasp * app_goal
    lift = pinch + np.array([0.0, 0.0, a.lift])
    print(f"grasp [{gi['backend']}]: pinch {np.round(pinch, 3)} score {score:.2f} width {width * 100:.1f} cm "
          f"approach {np.round(approach, 2)} ({np.degrees(np.arccos(np.clip(-approach[2], -1, 1))):.0f} deg from "
          f"vertical) | {gi['n_candidates']} candidates, inference {gi['infer_s']:.2f} s")

    res = {"frame": "odom", "table": table.name, "object": obj.tolist(), "pinch": pinch.tolist(),
           "pregrasp": pregrasp.tolist(), "lift": lift.tolist(), "quat_wxyz": R_to_quat_wxyz(R_goal),
           "score": float(score), "width": float(width), "approach": approach.tolist(),
           "orientation": a.orientation, "camera": t_wc.tolist(), "camera_quat_wxyz": R_to_quat_wxyz(R_wc),
           "base0": list(a.base0), "q0": list(a.q0), "source": "handheld_grasp_test", **gi,
           "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    (out / "grasp.json").write_text(json.dumps(res, indent=2))

    rows, status, ctrl = wbc_rollout(table, a.base0, a.q0, [(pregrasp, R_goal), (pinch, R_goal), (lift, R_goal)],
                                     ["pregrasp", "grasp", "lift"], a)
    write_trajectory(rows, out / "trajectory.csv", a.pan_offset)
    plotted = plot_trajectory(rows, table, obj, pinch, t_wc, out / "trajectory.png")
    e = np.array([r["e_p"] for r in rows])
    clr = np.array([r["clr"] for r in rows])
    clr = clr[np.isfinite(clr)]
    last = rows[-1]
    print(f"\n9-DOF WBC rollout: {status} | {len(rows)} steps, {last['t']:.1f} s | EE error RMS "
          f"{np.sqrt(np.mean(e ** 2)) * 100:.2f} cm, max {e.max() * 100:.2f} cm"
          + (f" | min clearance {clr.min() * 100:.1f} cm" if len(clr) else "")
          + (f" | QP failures {ctrl.qp_failures}" if ctrl.qp_failures else ""))
    for ph in ("pregrasp", "grasp", "lift"):
        r = next((r for r in reversed(rows) if r["phase"] == ph), None)
        if r is not None:
            print(f"  end of {ph:8s}: base ({r['base'][0]:+.3f}, {r['base'][1]:+.3f}, {np.degrees(r['base'][2]):+6.1f} deg)"
                  f"  q {np.round(r['q'], 3)}")
    print(f"saved {out}/: grasp.json, trajectory.csv{', trajectory.png' if plotted else ''}, snapshot.npz")
    print(f"real robot: ros2 run steve_wbc steve_figure8_wbc --mode ros --goto_file {out}/grasp.json --table {table.name}")
    return dict(res=res, rows=rows, status=status, R_wc=R_wc, t_wc=t_wc, obj=obj, obj_top=obj_top, info=info,
                pinch=pinch, R_goal=R_goal, R_grasp=R_g, pregrasp=pregrasp, lift=lift, width=width, score=score)



def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_argument_group("input")
    g.add_argument("--camera", default="auto", help="realsense2_camera namespace, e.g. /camera (auto: find it)")
    g.add_argument("--frames", type=int, default=5, help="depth frames fused (median / TSDF), hold still")
    g.add_argument("--from_npz", default=None, help="re-run on a saved snapshot.npz instead of the camera")
    g.add_argument("--synthetic", action="store_true", help="MuJoCo depth stand-in, no camera needed")
    g.add_argument("--timeout", type=float, default=20.0)
    g = p.add_argument_group("robot frame")
    g.add_argument("--table", default="lab_table", help="steve_wbc table model (config/tables/<name>.yaml)")
    g.add_argument("--object_xy", type=float, nargs=2, default=None,
                   help="where the object lies in odom [m] (default: table near half, 25 cm in)")
    g.add_argument("--base0", type=float, nargs=3, default=[0.0, 0.0, 0.0], help="robot start x y yaw (odom)")
    g.add_argument("--q0", type=float, nargs=6, default=SCAN_Q, help="arm start (MuJoCo convention)")
    g = p.add_argument_group("grasp")
    g.add_argument("--backend", choices=["vgn", "cgn"], default="vgn")
    g.add_argument("--orientation", choices=["topdown", "full"], default="topdown",
                   help="topdown: gripper down, turned to the grasp's finger axis | full: 6-DOF grasp")
    g.add_argument("--near", type=float, default=0.05, help="grasps within this xy radius of the object [m]")
    g.add_argument("--cgn_host", default="localhost")
    g.add_argument("--cgn_port", type=int, default=5555)
    g.add_argument("--cgn_tcp_depth", type=float, default=cgn_client.CGN_TCP_DEPTH,
                   help="Contact-GraspNet grasp origin -> fingertip centre [m]")
    g = p.add_argument_group("trajectory")
    g.add_argument("--pregrasp", type=float, default=0.10, help="pre-grasp distance back along the approach [m]")
    g.add_argument("--lift", type=float, default=LIFT)
    g.add_argument("--ee_speed", type=float, default=0.10, help="max EE speed [m/s]")
    g.add_argument("--rate", type=float, default=50.0)
    g.add_argument("--max_base_v", type=float, default=0.20)
    g.add_argument("--max_arm_vel", type=float, default=0.50)
    g.add_argument("--pan_offset", type=float, default=np.pi / 2,
                   help="q1_ros = q1 - offset: steve_figure8_wbc's --pan_offset (q_model = q_ros + offset)")
    g.add_argument("--out_dir", default=None)
    g.add_argument("--viewer", action="store_true", help="replay the trajectory in the MuJoCo viewer")
    g = p.add_argument_group("live RViz (--live; start `ros2 launch steve_perception grasp_test_rviz.launch.py`)")
    g.add_argument("--live", action="store_true",
                   help="live camera pose + cloud in odom; Enter = snapshot -> grasp + animated 9-DOF trajectory")
    g.add_argument("--live_hz", type=float, default=5.0, help="live re-anchoring rate [Hz]")
    g.add_argument("--viz_pan", type=float, default=None,
                   help="RViz robot: shoulder_pan_urdf = q1 + viz_pan. Default: -(UR mount yaw) read from "
                        "/robot_description (joint ur5ebase_joint): -pi/2 for the robot's URDF (+90 deg mount, "
                        "= the WBC's --pan_offset), +pi/2 for an older steve_simulation (-90 deg)")
    a = p.parse_args(*([argv] if argv is not None else []))

    table = TableModel.load(a.table)
    obj_xy = np.asarray(a.object_xy if a.object_xy is not None else default_object_xy(table), float)
    print(f"table {table.name}: top {table.top_z:.2f} m | object placed at {np.round(obj_xy, 3)} | "
          f"robot at {np.round(a.base0, 3)} | backend {a.backend}")
    if a.live:
        try:
            from steve_perception.grasp_test import rviz_live
        except ImportError:
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            import rviz_live
        return rviz_live.run(a, table, obj_xy, sys.modules[__name__])
    out = Path(a.out_dir or Path.home() / "ros_logs" / "grasp_test" / time.strftime("%Y%m%d_%H%M%S"))

    truth = None
    if a.from_npz:
        z = np.load(a.from_npz)
        frames = [z["depth"][i] for i in range(z["depth"].shape[0])]
        rgb, K = (z["rgb"] if z["rgb"].size else None), z["K"]
    elif a.synthetic:
        frames, rgb, K, truth = capture_synthetic(table, obj_xy, n_frames=a.frames)
    else:
        frames, rgb, K = capture_ros(a.camera, a.frames, a.timeout)
    try:
        r = process_snapshot(frames, rgb, K, a, table, obj_xy, out, truth)
    except GraspTestError as e:
        sys.exit(f"[ERROR] {e}")
    if a.viewer:
        replay(r["rows"], table)
    return 0 if r["status"] == "ok" else 1


def replay(rows, table):
    import mujoco
    import mujoco.viewer
    model = wbc.load_model(wbc.resolve_model(None), spec_edit=table.apply_to_spec)
    kin = wbc.WholeBodyKinematics(model)
    with mujoco.viewer.launch_passive(model, kin.data) as v:
        while v.is_running():
            for r in rows:
                if not v.is_running():
                    break
                kin.set_state(wbc.RobotState(r["base"], r["q"]))
                v.sync()
                time.sleep(0.02)


if __name__ == "__main__":
    sys.exit(main())
