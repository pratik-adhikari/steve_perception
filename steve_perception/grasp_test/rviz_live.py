#!/usr/bin/env python3
"""
rviz_live.py
============
Live RViz view for handheld_grasp_test (`--live`):

  continuously   the hand-held camera is re-anchored (table plane + object, as in the snapshot) at
                 --live_hz: its pose (TF odom -> handheld_camera, frustum) and its point cloud are
                 shown in the robot's odom frame while you position it
  Enter          snapshot (--frames) -> grasp -> 9-DOF WBC rollout (background thread); the grasp
                 (gripper at pre-grasp / grasp / lift), EE + base paths and the snapshot cloud are
                 shown, and the Steve model replays the trajectory in a loop
  q + Enter      quit

Topics (fixed frame odom; `ros2 launch steve_perception grasp_test_rviz.launch.py` starts
robot_state_publisher with the Steve URDF and RViz with this layout):
  /joint_states, TF odom -> base_link      robot model (UR5e + pan-tilt joints)
  TF odom -> handheld_camera                live camera pose (ROS optical frame)
  /grasp_test/live_cloud                    live cloud (PointCloud2, odom)
  /grasp_test/live_markers                  camera frustum, live object, status text
  /grasp_test/snapshot_cloud    (latched)   cloud of the last snapshot
  /grasp_test/markers           (latched)   table, object, grippers, paths, result text
  /grasp_test/grasp_pose        (latched)   grasp (pinch point, WBC goal orientation)
"""

import queue
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

URDF_ARM = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
            "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
URDF_OTHER = ["pan_tilt_tilt_motor_joint", "pan_tilt_pan_motor_joint"]   # held at 0
URDF_PREFIX = "ur5e"
CAMERA_FRAME = "handheld_camera"
ROBOT_HZ = 30.0


def _quat_xyzw(R):
    """Rotation matrix -> quaternion (x, y, z, w), Shepperd's method (stable for any rotation)."""
    R = np.asarray(R, float)
    tr = np.trace(R)
    if tr > 0:
        s = 2.0 * np.sqrt(tr + 1.0)
        q = (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s, 0.25 * s
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2.0 * np.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k])
        v = np.zeros(3)
        v[i], v[j], v[k] = 0.25 * s, (R[j, i] + R[i, j]) / s, (R[k, i] + R[i, k]) / s
        q = (*v, (R[k, j] - R[j, k]) / s)
    q = np.asarray(q)
    return tuple(q / np.linalg.norm(q))


def _yaw_quat(yaw):
    return 0.0, 0.0, np.sin(yaw / 2), np.cos(yaw / 2)


# ══════════════════════════════════════════════════════════════════════════════
# Frame sources
# ══════════════════════════════════════════════════════════════════════════════
class RosCamera:
    """Latest aligned depth + colour + K of the RealSense node (converted on demand)."""

    def __init__(self, node, camera, h, timeout):
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import CameraInfo, Image
        self.h, self.K, self._d, self._c, self.seq, self._used = h, None, None, None, 0, 0
        ns = h.find_camera_ns(node, camera, timeout)
        node.create_subscription(CameraInfo, f"{ns}/aligned_depth_to_color/camera_info",
                                 lambda m: setattr(self, "K", np.array(m.k).reshape(3, 3)), 10)
        node.create_subscription(Image, f"{ns}/aligned_depth_to_color/image_raw", self._depth_cb,
                                 qos_profile_sensor_data)
        node.create_subscription(Image, f"{ns}/color/image_raw", lambda m: setattr(self, "_c", m),
                                 qos_profile_sensor_data)

    def _depth_cb(self, m):
        self._d = m
        self.seq += 1

    def latest(self):
        if self._d is None or self.K is None:
            return None
        return self.h.depth_from_msg(self._d), (self.h.rgb_from_msg(self._c) if self._c is not None else None)

    def new_frame(self):
        """The latest frame if it arrived after the previous call, else None."""
        if self.seq == self._used:
            return None
        self._used = self.seq
        return self.latest()


class SyntheticCamera:
    """MuJoCo stand-in (handheld_grasp_test.capture_synthetic): a fixed view with fresh noise."""

    def __init__(self, h, table, obj_xy):
        frames, self.rgb, self.K, _ = h.capture_synthetic(table, obj_xy, n_frames=8)
        self.frames, self.i = frames, 0

    def latest(self):
        self.i = (self.i + 1) % len(self.frames)
        return self.frames[self.i], self.rgb

    new_frame = latest


# ══════════════════════════════════════════════════════════════════════════════
# Publishing
# ══════════════════════════════════════════════════════════════════════════════
class Viz:
    def __init__(self, node, a, table):
        from geometry_msgs.msg import PoseStamped
        from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
        from sensor_msgs.msg import JointState, PointCloud2
        from tf2_ros import TransformBroadcaster
        from visualization_msgs.msg import MarkerArray
        self.node, self.a, self.table = node, a, table
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             reliability=ReliabilityPolicy.RELIABLE)
        self.p_markers = node.create_publisher(MarkerArray, "/grasp_test/markers", latched)
        self.p_snap = node.create_publisher(PointCloud2, "/grasp_test/snapshot_cloud", latched)
        self.p_pose = node.create_publisher(PoseStamped, "/grasp_test/grasp_pose", latched)
        self.p_live = node.create_publisher(MarkerArray, "/grasp_test/live_markers", 10)
        self.p_live_cloud = node.create_publisher(PointCloud2, "/grasp_test/live_cloud", 10)
        self.p_js = node.create_publisher(JointState, "/joint_states", 10)
        self.tf = TransformBroadcaster(node)
        self.anim, self.next_robot = None, 0.0
        # shoulder_pan offset MuJoCo -> URDF: -(UR mount yaw in the URDF), from /robot_description
        self.viz_pan = a.viz_pan
        if self.viz_pan is None:
            from std_msgs.msg import String
            node.create_subscription(String, "/robot_description", self._description_cb, latched)
        self.publish_result(None)

    def _description_cb(self, msg):
        import xml.etree.ElementTree as ET
        try:
            for j in ET.fromstring(msg.data).findall("joint"):
                if j.find("child").get("link") == URDF_PREFIX + "base_link":
                    o = j.find("origin")
                    yaw = float(((o.get("rpy") if o is not None else None) or "0 0 0").split()[2])
                    if self.a.viz_pan is None:
                        self.viz_pan = -yaw
                        print(f"[live] URDF: UR5e mounted at {np.degrees(yaw):+.0f} deg -> "
                              f"shoulder_pan_urdf = q1 {-yaw:+.4f}")
                    return
        except Exception as e:
            print(f"[live] /robot_description not parsed ({e}); robot shown with shoulder_pan offset -pi/2")

    # ── helpers ──────────────────────────────────────────────────────────────
    def _stamp(self):
        return self.node.get_clock().now().to_msg()

    def _marker(self, ns, mid, mtype, rgba, pos=(0, 0, 0), quat=(0, 0, 0, 1), scale=(1, 1, 1), frame="odom"):
        from visualization_msgs.msg import Marker
        m = Marker()
        m.header.frame_id, m.header.stamp = frame, self._stamp()
        m.ns, m.id, m.type, m.action = ns, mid, mtype, Marker.ADD
        m.pose.position.x, m.pose.position.y, m.pose.position.z = (float(v) for v in pos)
        (m.pose.orientation.x, m.pose.orientation.y, m.pose.orientation.z,
         m.pose.orientation.w) = (float(v) for v in quat)
        m.scale.x, m.scale.y, m.scale.z = (float(v) for v in scale)
        m.color.r, m.color.g, m.color.b, m.color.a = (float(v) for v in rgba)
        return m

    def _line(self, ns, mid, pts, rgba, width=0.006, mtype=None):
        from geometry_msgs.msg import Point
        from visualization_msgs.msg import Marker
        m = self._marker(ns, mid, Marker.LINE_STRIP if mtype is None else mtype, rgba, scale=(width, 0, 0))
        m.points = [Point(x=float(p[0]), y=float(p[1]), z=float(p[2])) for p in pts]
        return m

    def _text(self, ns, mid, pos, text, rgba=(1, 1, 1, 1), size=0.04, frame="odom"):
        from visualization_msgs.msg import Marker
        m = self._marker(ns, mid, Marker.TEXT_VIEW_FACING, rgba, pos, scale=(0, 0, size), frame=frame)
        m.text = text
        return m

    def _gripper(self, ns, mid0, p, R, width, rgba):
        """Two fingers + palm + stem in the pinch convention (z approach, y finger axis; origin at
        the fingertip centre)."""
        from visualization_msgs.msg import Marker
        q = _quat_xyzw(R)
        w = max(float(width), 0.01)
        parts = [((0, +(w / 2 + 0.006), -0.02), (0.02, 0.012, 0.05)),   # finger
                 ((0, -(w / 2 + 0.006), -0.02), (0.02, 0.012, 0.05)),   # finger
                 ((0, 0, -0.05), (0.03, w + 0.024, 0.014)),            # palm
                 ((0, 0, -0.09), (0.03, 0.03, 0.07))]                  # stem toward the flange
        return [self._marker(ns, mid0 + i, Marker.CUBE, rgba, p + R @ np.array(off), q, size)
                for i, (off, size) in enumerate(parts)]

    def _cloud(self, P, colors):
        from sensor_msgs.msg import PointCloud2, PointField
        n = len(P)
        arr = np.zeros(n, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("rgb", "<u4")])
        arr["x"], arr["y"], arr["z"] = P[:, 0], P[:, 1], P[:, 2]
        c = colors.astype(np.uint32)
        arr["rgb"] = (c[:, 0] << 16) | (c[:, 1] << 8) | c[:, 2]
        msg = PointCloud2()
        msg.header.frame_id, msg.header.stamp = "odom", self._stamp()
        msg.height, msg.width, msg.is_bigendian, msg.is_dense = 1, n, False, True
        msg.fields = [PointField(name=k, offset=4 * i, datatype=PointField.FLOAT32, count=1)
                      for i, k in enumerate(("x", "y", "z", "rgb"))]
        msg.point_step, msg.row_step, msg.data = 16, 16 * n, arr.tobytes()
        return msg

    def _cloud_colors(self, info, rgb, D_shape):
        v, u = info["pix"]
        if rgb is not None and rgb.shape[:2] == D_shape:
            return rgb[v, u]
        hz = np.clip((info["cloud"][:, 2] - self.table.top_z) / 0.2, 0, 1)  # height above the table
        return np.stack([255 * hz, 160 * np.ones_like(hz), 255 * (1 - hz)], 1).astype(np.uint8)

    def _tf(self, child, p, q):
        from geometry_msgs.msg import TransformStamped
        t = TransformStamped()
        t.header.frame_id, t.header.stamp, t.child_frame_id = "odom", self._stamp(), child
        t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = (float(v) for v in p)
        (t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z,
         t.transform.rotation.w) = (float(v) for v in q)
        self.tf.sendTransform(t)

    # ── live camera ──────────────────────────────────────────────────────────
    def live(self, R_wc, t_wc, obj, obj_top, info, rgb, K, D_shape):
        from visualization_msgs.msg import Marker, MarkerArray
        self._tf(CAMERA_FRAME, t_wc, _quat_xyzw(R_wc))
        self.p_live_cloud.publish(self._cloud(info["cloud"], self._cloud_colors(info, rgb, D_shape)))
        # frustum 0.15 m deep, in the camera frame
        H, W = D_shape
        d = 0.15
        c = [np.array([(u - K[0, 2]) * d / K[0, 0], (v - K[1, 2]) * d / K[1, 1], d]) for u, v in
             ((0, 0), (W, 0), (W, H), (0, H))]
        seg = [np.zeros(3), c[0], np.zeros(3), c[1], np.zeros(3), c[2], np.zeros(3), c[3],
               c[0], c[1], c[1], c[2], c[2], c[3], c[3], c[0]]
        fr = self._line("live", 0, seg, (1, 0.2, 1, 1), 0.004, Marker.LINE_LIST)
        fr.header.frame_id = CAMERA_FRAME
        h_obj = max(obj_top - self.table.top_z, 0.01)
        ext = max(info.get("obj_xy_extent", 0.05), 0.02)
        ob = self._marker("live", 1, Marker.CYLINDER, (1, 0.3, 0.1, 0.35),
                          (obj[0], obj[1], self.table.top_z + h_obj / 2), scale=(ext, ext, h_obj))
        txt = self._text("live", 2, t_wc + [0, 0, 0.06],
                         f"camera {info['h_cam']:.2f} m above table, tilt {info['tilt']:.0f} deg"
                         + ("  (too close!)" if info["h_cam"] < 0.3 else ""), (1, 0.6, 1, 1), 0.035)
        self.p_live.publish(MarkerArray(markers=[fr, ob, txt]))

    def live_lost(self, why, t_last):
        from visualization_msgs.msg import MarkerArray
        pos = t_last if t_last is not None else np.array([*self.table.center, self.table.top_z + 0.5])
        self.p_live.publish(MarkerArray(markers=[self._text("live", 2, pos + [0, 0, 0.06], f"searching: {why}",
                                                            (1, 0.4, 0.4, 1), 0.035)]))

    # ── snapshot result ──────────────────────────────────────────────────────
    def status(self, text, rgba=(1, 1, 0.3, 1)):
        from visualization_msgs.msg import MarkerArray
        t = self.table
        self.p_markers.publish(MarkerArray(markers=[self._text("status", 0, [*t.center, t.top_z + 0.45], text, rgba)]))
        print(f"[live] {text}")

    def publish_result(self, r, rgb=None, D_shape=None):
        from geometry_msgs.msg import PoseStamped
        from visualization_msgs.msg import Marker, MarkerArray
        t = self.table
        ms = [Marker(action=Marker.DELETEALL)]
        ms.append(self._marker("table", 0, Marker.CUBE, (0.6, 0.4, 0.2, 0.45),
                               (*t.center, t.top_z - 0.5 * t.thickness), _yaw_quat(t.yaw),
                               (*t.size, t.thickness)))
        if r is None:
            ms.append(self._text("status", 0, [*t.center, t.top_z + 0.45],
                                 "press Enter in the terminal for a snapshot", (1, 1, 0.3, 1)))
            self.p_markers.publish(MarkerArray(markers=ms))
            return
        info, obj, rows = r["info"], r["obj"], r["rows"]
        h_obj = max(r["obj_top"] - t.top_z, 0.01)
        ext = max(info.get("obj_xy_extent", 0.05), 0.02)
        ms.append(self._marker("object", 0, Marker.CYLINDER, (0.9, 0.15, 0.1, 0.8),
                               (obj[0], obj[1], t.top_z + h_obj / 2), scale=(ext, ext, h_obj)))
        ms += self._gripper("grasp", 0, r["pinch"], r["R_goal"], r["width"], (0.1, 0.9, 0.2, 0.9))
        ms += self._gripper("pregrasp", 0, r["pregrasp"], r["R_goal"], r["width"], (0.2, 0.6, 1.0, 0.3))
        ms += self._gripper("lift", 0, r["lift"], r["R_goal"], r["width"], (0.2, 0.6, 1.0, 0.3))
        a = self._marker("approach", 0, Marker.ARROW, (0.1, 0.9, 0.2, 0.9), scale=(0.008, 0.016, 0.02))
        from geometry_msgs.msg import Point
        a.points = [Point(x=float(p[0]), y=float(p[1]), z=float(p[2])) for p in (r["pregrasp"], r["pinch"])]
        ms.append(a)
        ms.append(self._line("ee_ref", 0, [x["p_ref"] for x in rows[::5]], (0.1, 0.9, 0.2, 0.8), 0.004))
        ms.append(self._line("ee", 0, [x["p"] for x in rows[::5]], (1.0, 0.85, 0.1, 0.9), 0.006))
        ms.append(self._line("base", 0, [[*x["base"][:2], 0.01] for x in rows[::5]], (0.2, 0.4, 1.0, 1), 0.02))
        res = r["res"]
        ok = r["status"] == "ok"
        e = max(x["e_p"] for x in rows)
        ms.append(self._text("status", 0, [*obj[:2], t.top_z + 0.45],
                             f"{res['backend'].upper()} score {res['score']:.2f} | width {r['width'] * 100:.1f} cm | "
                             f"WBC {'ok' if ok else r['status']} {rows[-1]['t']:.1f} s, max err {e * 100:.1f} cm",
                             (0.4, 1, 0.4, 1) if ok else (1, 0.4, 0.4, 1)))
        self.p_markers.publish(MarkerArray(markers=ms))
        ps = PoseStamped()
        ps.header.frame_id, ps.header.stamp = "odom", self._stamp()
        ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = (float(v) for v in r["pinch"])
        (ps.pose.orientation.x, ps.pose.orientation.y, ps.pose.orientation.z,
         ps.pose.orientation.w) = _quat_xyzw(r["R_goal"])
        self.p_pose.publish(ps)
        if D_shape is not None:
            self.p_snap.publish(self._cloud(info["cloud"], self._cloud_colors(info, rgb, D_shape)))
        self.anim = (rows, time.monotonic())

    # ── robot ────────────────────────────────────────────────────────────────
    def robot(self, now):
        """Steve model: the trajectory in a loop (1.5 s pause at the end), else the start pose."""
        if now < self.next_robot:
            return
        self.next_robot = now + 1.0 / ROBOT_HZ
        from sensor_msgs.msg import JointState
        if self.anim is None:
            base, q = np.asarray(self.a.base0, float), np.asarray(self.a.q0, float)
        else:
            rows, t0 = self.anim
            period = rows[-1]["t"] + 1.5
            k = min(int(((now - t0) % period) * self.a.rate), len(rows) - 1)
            base, q = rows[k]["base"], rows[k]["q"]
        self._tf("base_link", (base[0], base[1], 0.0), _yaw_quat(base[2]))
        js = JointState()
        js.header.stamp = self._stamp()
        js.name = [URDF_PREFIX + n for n in URDF_ARM] + URDF_OTHER
        qu = np.asarray(q, float).copy()
        qu[0] += -np.pi / 2 if self.viz_pan is None else self.viz_pan   # until the description is read
        js.position = [float(v) for v in qu] + [0.0] * len(URDF_OTHER)
        self.p_js.publish(js)


# ══════════════════════════════════════════════════════════════════════════════
def _stdin_reader(q):
    try:
        for line in sys.stdin:
            q.put(line.strip().lower())
    finally:
        q.put("q")   # EOF / closed terminal


def run(a, table, obj_xy, h):
    """Live loop. `h` = the handheld_grasp_test module (anchoring, pipeline)."""
    import rclpy
    rclpy.init()
    node = rclpy.create_node("handheld_grasp_live")
    viz = Viz(node, a, table)
    src = SyntheticCamera(h, table, obj_xy) if a.synthetic else RosCamera(node, a.camera, h, a.timeout)
    cmds = queue.Queue()
    threading.Thread(target=_stdin_reader, args=(cmds,), daemon=True).start()
    pool = ThreadPoolExecutor(max_workers=1)
    job, collecting, next_anchor, t_last = None, None, 0.0, None
    print("\nLIVE — RViz: `ros2 launch steve_perception grasp_test_rviz.launch.py` (fixed frame odom)\n"
          "  hold the camera 0.4-0.6 m above the object, looking down, image-top away from you\n"
          "  Enter = snapshot -> grasp + 9-DOF trajectory      q + Enter = quit\n")
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.005)
            now = time.monotonic()
            try:
                c = cmds.get_nowait()
            except queue.Empty:
                c = None
            if c == "q":
                break
            if c is not None:
                if job is not None or collecting is not None:
                    print("[live] still busy with the last snapshot")
                elif src.K is None:
                    print("[live] no camera_info yet")
                else:
                    collecting = []
                    viz.status(f"snapshot: hold still ({a.frames} frames) ...")
            if collecting is not None:
                f = src.new_frame()
                if f is not None:
                    collecting.append(f)
                if len(collecting) >= a.frames:
                    frames, rgb = [d for d, _ in collecting], collecting[-1][1]
                    out = Path(a.out_dir or Path.home() / "ros_logs" / "grasp_test") / time.strftime("%Y%m%d_%H%M%S")
                    job = (pool.submit(h.process_snapshot, frames, rgb, src.K, a, table, obj_xy, out), rgb,
                           frames[0].shape)
                    collecting = None
                    viz.status(f"computing grasp ({a.backend}) + 9-DOF trajectory ...")
            if job is not None and job[0].done():
                fut, rgb, shape = job
                job = None
                try:
                    viz.publish_result(fut.result(), rgb, shape)
                except h.GraspTestError as e:
                    viz.status(f"no grasp: {e}", (1, 0.4, 0.4, 1))
                except Exception as e:  # keep the live view running
                    traceback.print_exc()
                    viz.status(f"error: {e}", (1, 0.4, 0.4, 1))
            if now >= next_anchor and collecting is None:
                next_anchor = now + 1.0 / a.live_hz
                f = src.latest()
                if f is not None:
                    D, rgb = f
                    try:
                        R_wc, t_wc, obj, top, info = h.anchor_camera(D, src.K, table, obj_xy, a.base0[2],
                                                                     step=6, iters=120)
                        viz.live(R_wc, t_wc, obj, top, info, rgb, src.K, D.shape)
                        t_last = t_wc
                    except ValueError as e:
                        viz.live_lost(str(e), t_last)
            viz.robot(now)
    except KeyboardInterrupt:
        pass
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
        node.destroy_node()
        rclpy.try_shutdown()
    return 0
