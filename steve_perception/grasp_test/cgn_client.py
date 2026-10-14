#!/usr/bin/env python3
"""
cgn_client.py
=============
Client for the Contact-GraspNet inference server used by DiNBV-Grasp
(XieZilongAI/DiNBV-Grasp, DiNBV_Grasp_real/src/vlm_nbv/grasp_executor.py): same ZMQ REQ
protocol, so the same `graspnet_server.py` serves DiNBV and this test.

Request  (pickle): {depth HxW float32 [m], K 3x3, segmap HxW int32 | None, rgb HxWx3 | None,
                    z_range [min, max], forward_passes int}
Reply    (pickle): {pred_grasps_cam {seg_id: Nx4x4}, scores {seg_id: N}, contact_pts {seg_id: Nx3},
                    [gripper_openings {seg_id: N}], error None | str}

Grasp frame (Contact-GraspNet, Panda hand), in the camera optical frame:
  origin = gripper base, +z = approach, +x = finger closing axis; the fingertip centre lies
  CGN_TCP_DEPTH (0.1034 m) along +z.
"""

import pickle
import socket
import time

import numpy as np

CGN_TCP_DEPTH = 0.1034   # m, Panda hand base -> fingertip centre (Contact-GraspNet's gripper)


def get_grasps(depth, K, segmap=None, rgb=None, host="localhost", port=5555, z_range=(0.2, 1.5),
               forward_passes=1, timeout_ms=30000):
    """Grasps from the server: list of (T_cam_grasp 4x4, score, width or None), best first.
    Raises RuntimeError when the server is unreachable or reports an error."""
    try:
        import zmq
    except ImportError as e:
        raise RuntimeError("pyzmq not installed: pip install pyzmq") from e
    try:
        socket.create_connection((host, port), timeout=2).close()
    except OSError as e:
        raise RuntimeError(f"Contact-GraspNet server not reachable at {host}:{port} — start "
                           "graspnet_server.py (see grasp_test/README.md)") from e

    ctx = zmq.Context()
    s = ctx.socket(zmq.REQ)
    s.setsockopt(zmq.RCVTIMEO, timeout_ms)
    s.setsockopt(zmq.SNDTIMEO, timeout_ms)
    s.setsockopt(zmq.LINGER, 0)
    s.connect(f"tcp://{host}:{port}")
    try:
        t0 = time.monotonic()
        s.send(pickle.dumps({"depth": np.asarray(depth, np.float32), "K": np.asarray(K, float),
                             "segmap": None if segmap is None else np.asarray(segmap, np.int32),
                             "rgb": rgb, "z_range": list(z_range), "forward_passes": forward_passes}))
        res = pickle.loads(s.recv())
        print(f"  [CGN] server answered in {time.monotonic() - t0:.2f} s")
    except zmq.error.Again as e:
        raise RuntimeError(f"Contact-GraspNet server timed out after {timeout_ms / 1000:.0f} s") from e
    finally:
        s.close()
        ctx.term()
    if res.get("error"):
        raise RuntimeError(f"Contact-GraspNet server: {res['error']}")

    out = []
    widths = res.get("gripper_openings") or {}
    for sid, G in (res.get("pred_grasps_cam") or {}).items():
        sc = res.get("scores", {}).get(sid, np.zeros(len(G)))
        w = widths.get(sid)
        for i in range(len(G)):
            out.append((np.asarray(G[i], float), float(sc[i]), None if w is None else float(w[i])))
    out.sort(key=lambda g: -g[1])
    return out


def to_pinch_frame(T_wg, tcp_depth=CGN_TCP_DEPTH):
    """Contact-GraspNet grasp (world) -> (fingertip centre, rotation in the WBC 'pinch' convention:
    z = approach, y = finger axis)."""
    R = T_wg[:3, :3]
    z, y = R[:, 2], R[:, 0]          # approach, finger closing axis
    x = np.cross(y, z)
    return T_wg[:3, 3] + tcp_depth * z, np.column_stack([x, y, z])
