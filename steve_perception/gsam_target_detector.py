#!/usr/bin/env python3
"""
gsam_target_detector.py
=======================
Target detector with Grounded SAM: the object is named by a text prompt ("red ball"), found by
GroundingDINO, segmented by SAM (model server `ros2 run steve_perception model_server gsam`), and
its CENTRE is published as geometry_msgs/PointStamped in `target_frame` on `target_topic` — the
same output as steve_wbc's colour-threshold red_ball_detector, so active_perception_wbc works
with either (launch argument detector:=hsv|gsam).

Per camera: synchronised colour + depth, the newest frame pair is sent to the server (frames
arriving meanwhile are dropped; ~5 Hz on an RTX 3070 laptop). Every returned mask is checked
against the known object:
  * sphere: least-squares sphere fit of the mask's depth (known radius) must succeed, and the
    mask's apparent size must match the object size at its depth (size_tolerance);
  * other shapes: the size check only; centre = visible surface pushed half a size along the ray.
Several plausible masks (clutter with look-alikes): the one nearest the last published position
(within `gate`, for `gate_hold` s), otherwise the one GroundingDINO scores highest.

Records (save_dir, default auto = the launch's log folder, '' = off): <run>/detector/ gets an
annotated frame every save_period s (masks, boxes, scores, why a mask was rejected, the chosen one
green) and detections.csv with every processed frame.

    ros2 run steve_perception gsam_target_detector --ros-args -p prompt:="red ball" \\
        -p cameras:="[gripper_camera]" -p object_size:=0.065
"""

import threading
import time

import numpy as np
import rclpy
from geometry_msgs.msg import PointStamped
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import Buffer, TransformException, TransformListener

from steve_perception.model_server import GSAM_ENDPOINT, ModelClient, ModelServerError
from steve_perception.utils.object_geometry import (R_OPT_TO_XFWD, image_to_array, mask_points,
                                                    quat_to_mat, sphere_center)
from steve_perception.utils.run_record import ImageRecorder, save_folder

# topic names under a camera namespace: (colour, aligned depth, camera_info)
TOPIC_STYLES = {
    "gazebo": ("image_raw", "depth/image_raw", "camera_info"),
    "realsense": ("color/image_raw", "aligned_depth_to_color/image_raw", "color/camera_info"),
}
# Gazebo's head camera publishes in a non-optical (x-forward) frame
DEFAULT_CAMERAS = {"pan_tilt_camera": ("/pan_tilt_camera/pan_tilt_camera", "x_forward")}


class GsamTargetDetector(Node):
    def __init__(self):
        super().__init__("gsam_target_detector")
        p = lambda n, v: self.declare_parameter(n, v).value
        self.prompt = p("prompt", "red ball")
        self.target_frame = p("target_frame", "odom")
        self.size = p("object_size", 0.065)          # m: sphere diameter / largest box side
        self.shape = p("shape", "sphere")            # sphere | other
        self.box_threshold = p("box_threshold", 0.35)
        self.text_threshold = p("text_threshold", 0.25)
        self.size_tol = p("size_tolerance", 0.4)     # apparent size within (1 ± tol) x expected
        self.gate, self.gate_hold = p("gate", 0.10), p("gate_hold", 2.0)
        self.min_range, self.max_range = p("min_range", 0.12), p("max_range", 2.0)
        self.max_rate = p("max_rate", 10.0)
        self.debug = p("debug_image", True)
        cameras = p("cameras", ["gripper_camera"])
        style = p("topic_style", "realsense")
        server = p("server", GSAM_ENDPOINT)

        self.client = ModelClient(server, timeout_s=p("server_timeout", 5.0))
        self.tf = Buffer(cache_time=Duration(seconds=10.0))
        self.tf_listener = TransformListener(self.tf, self)
        self.pub = self.create_publisher(PointStamped, p("target_topic", "/steve_perception/target_point"), 10)
        self.dbg_pub = self.create_publisher(Image, "~/debug_image", 1) if self.debug else None
        folder = save_folder(p("save_dir", "auto"), "detector")
        self.rec = None
        if folder is not None:
            self.rec = ImageRecorder(folder, p("save_period", 0.5),
                                     ["wall_time", "stamp", "camera", "masks", "chosen", "score", "x", "y", "z",
                                      "rejected"])
            self.get_logger().info(f"saving detections to {folder}")

        self.lock = threading.Lock()
        self.cams = {}
        img_t, depth_t, info_t = TOPIC_STYLES[style]
        for name in cameras:
            ns_default, frame_default = DEFAULT_CAMERAS.get(name, (f"/{name}", "optical"))
            ns = p(f"{name}.namespace", ns_default)
            conv = p(f"{name}.frame", frame_default if style == "gazebo" else "optical")
            c = self.cams[name] = dict(name=name, K=None, conv=conv, frame=None, done=None, n=0, n_req=0)
            self.create_subscription(CameraInfo, f"{ns}/{info_t}",
                                     lambda m, c=c: c.__setitem__("K", np.array(m.k).reshape(3, 3)), 10)
            c["sync"] = ApproximateTimeSynchronizer(
                [Subscriber(self, Image, f"{ns}/{img_t}"), Subscriber(self, Image, f"{ns}/{depth_t}")],
                queue_size=5, slop=0.05)
            c["sync"].registerCallback(lambda rgb, depth, c=c: self._frame_cb(c, rgb, depth))
            self.get_logger().info(f"camera '{name}': {ns}/{{{img_t}, {depth_t}}}  frame={conv}")
        self.get_logger().info(f"target: '{self.prompt}' ({self.size * 100:.1f} cm {self.shape}) via "
                               f"Grounded SAM at {server}")
        self.last = None            # (position, monotonic time) of the last published detection
        self.t_sum, self.err = 0.0, None
        self.create_timer(5.0, self._report)
        self._stop = False
        self.worker = threading.Thread(target=self._loop, daemon=True)
        self.worker.start()

    # ── ROS callbacks (executor thread): keep only the newest frame pair per camera ──
    def _frame_cb(self, c, rgb, depth):
        with self.lock:
            c["frame"] = (rgb, depth)

    def _report(self):
        n = sum(c["n_req"] for c in self.cams.values())
        msg = "detections: " + ", ".join(f"{k}={c['n']}" for k, c in self.cams.items())
        if n:
            msg += f"  ({n} frames, {self.t_sum / n * 1000:.0f} ms each)"
        if self.err:
            self.get_logger().warn(f"{msg}  model server: {self.err}")
        else:
            self.get_logger().info(msg)
        for c in self.cams.values():
            c["n_req"] = 0
        self.t_sum, self.err = 0.0, None

    # ── worker thread: server round trips ─────────────────────────────────────
    def _loop(self):
        period = 1.0 / max(self.max_rate, 0.1)
        while rclpy.ok() and not self._stop:
            t0 = time.monotonic()
            job = None
            with self.lock:  # the camera whose newest frame waited longest
                ready = [c for c in self.cams.values() if c["frame"] is not None and c["K"] is not None]
                if ready:
                    c = min(ready, key=lambda c: c["done"] or 0.0)
                    job, c["frame"] = (c, *c["frame"]), None
            if job is None:
                time.sleep(0.01)
                continue
            try:
                self._process(*job)
            except ModelServerError as e:
                self.err = str(e)
                time.sleep(1.0)  # server down / loading: do not spin
            except Exception as e:  # never kill the worker thread
                self.get_logger().error(f"{type(e).__name__}: {e}", throttle_duration_sec=2.0)
            job[0]["done"] = time.monotonic()
            time.sleep(max(0.0, period - (time.monotonic() - t0)))

    def _process(self, c, rgb_msg, depth_msg):
        rgb, depth = image_to_array(rgb_msg), image_to_array(depth_msg)
        if depth.shape != rgb.shape[:2]:
            self.get_logger().warn(f"colour {rgb.shape[:2]} and depth {depth.shape} differ: depth must be "
                                   "aligned to colour", throttle_duration_sec=5.0)
            return
        t0 = time.monotonic()
        dets = self.client.segment(rgb, self.prompt, self.box_threshold, self.text_threshold)
        self.t_sum += time.monotonic() - t0
        c["n_req"] += 1

        K, cands = c["K"], []
        for d in dets:
            p_opt, why = self._locate(d["mask"], depth, K)
            d["why"] = why
            if p_opt is not None:
                cands.append((d, p_opt))
        best = None
        if cands:
            tf = self._lookup(rgb_msg)
            if tf is None:
                return
            R, tr = tf
            world = [(d, R @ (R_OPT_TO_XFWD @ p if c["conv"] == "x_forward" else p) + tr) for d, p in cands]
            best = max(world, key=lambda dw: dw[0]["score"])
            if self.last is not None and time.monotonic() - self.last[1] < self.gate_hold:
                near = min(world, key=lambda dw: np.linalg.norm(dw[1] - self.last[0]))
                if np.linalg.norm(near[1] - self.last[0]) < self.gate:
                    best = near  # stay on the tracked object among look-alikes
            out = PointStamped()
            out.header.stamp, out.header.frame_id = rgb_msg.header.stamp, self.target_frame
            out.point.x, out.point.y, out.point.z = (float(v) for v in best[1])
            self.pub.publish(out)
            self.last = (best[1], time.monotonic())
            c["n"] += 1
        chosen = best[0] if best else None
        if self.rec is not None:
            st = rgb_msg.header.stamp
            self.rec.log([f"{time.time():.3f}", f"{st.sec}.{st.nanosec:09d}", c["name"], len(dets),
                          chosen["phrase"] if chosen else "", f"{chosen['score']:.3f}" if chosen else "",
                          *(f"{v:.4f}" for v in (best[1] if best else ())), *([""] * (3 if not best else 0)),
                          "; ".join(f"{d['phrase']} {d['score']:.2f}: {d['why']}" for d in dets if d.get("why"))])
        publish = self.dbg_pub is not None and self.dbg_pub.get_subscription_count() > 0
        save = self.rec is not None and self.rec.due()
        if publish or save:
            vis = self._annotate(rgb, dets, chosen, best[1] if best else None)
            if publish:
                self._publish_image(vis, rgb_msg.header)
            if save:
                self.rec.save(vis, c["name"] if chosen else f"{c['name']}_none")

    def _locate(self, mask, depth, K):
        """(centre in the camera optical frame, None) or (None, reason)."""
        P = mask_points(depth, mask, K, self.min_range, self.max_range)
        if len(P) < 30:
            return None, "no depth"
        z = float(np.median(P[:, 2]))
        # apparent size: diameter of a disc of the mask's area vs. the object's at this depth
        d_px = 2.0 * np.sqrt(mask.sum() / np.pi)
        ratio = d_px / (K[0, 0] * self.size / z)
        if abs(ratio - 1.0) > self.size_tol:
            return None, f"size x{ratio:.2f}"
        if self.shape == "sphere":
            c = sphere_center(depth, mask, K, self.size / 2, self.min_range, self.max_range)
            return (c, "") if c is not None else (None, "not a sphere")
        s = np.median(P, axis=0)
        return s + 0.5 * self.size * s / np.linalg.norm(s), ""

    def _lookup(self, msg):
        try:
            t = self.tf.lookup_transform(self.target_frame, msg.header.frame_id, Time.from_msg(msg.header.stamp),
                                         Duration(seconds=0.1))
        except TransformException:
            try:  # stamps racing the TF buffer: the latest transform is close enough for a still camera
                t = self.tf.lookup_transform(self.target_frame, msg.header.frame_id, Time())
            except TransformException as e:
                self.get_logger().warn(f"TF {msg.header.frame_id} -> {self.target_frame}: {e}",
                                       throttle_duration_sec=2.0)
                return None
        tr = t.transform.translation
        return quat_to_mat(t.transform.rotation), np.array([tr.x, tr.y, tr.z])

    def _annotate(self, rgb, dets, chosen, pos):
        import cv2
        vis = rgb.copy()
        for d in dets:
            col = (0, 255, 0) if d is chosen else (255, 160, 0)
            vis[d["mask"]] = (0.55 * vis[d["mask"]] + 0.45 * np.array(col)).astype(np.uint8)
            x0, y0, x1, y1 = (int(v) for v in d["box"])
            cv2.rectangle(vis, (x0, y0), (x1, y1), col, 2)
            label = f"{d['phrase']} {d['score']:.2f}" + (f" [{d['why']}]" if d.get("why") else "")
            cv2.putText(vis, label, (x0, max(14, y0 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)
        head = (f"'{self.prompt}': " + (f"{self.target_frame} ({pos[0]:+.3f}, {pos[1]:+.3f}, {pos[2]:.3f})"
                                        if pos is not None else f"none ({len(dets)} mask(s))"))
        cv2.putText(vis, head, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 3)
        cv2.putText(vis, head, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
        return vis

    def _publish_image(self, vis, header):
        msg = Image()
        msg.header, msg.height, msg.width, msg.encoding = header, vis.shape[0], vis.shape[1], "rgb8"
        msg.step, msg.data = vis.shape[1] * 3, vis.tobytes()
        self.dbg_pub.publish(msg)

    def destroy_node(self):
        self._stop = True
        self.worker.join(timeout=2.0)
        self.client.close()
        if self.rec is not None:
            self.rec.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = GsamTargetDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
