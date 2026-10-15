"""
client.py — ROS-side (system Python) client of the model servers (server.py).

    gsam = ModelClient(GSAM_ENDPOINT)
    dets = gsam.segment(rgb, "red ball")            # [{box, score, phrase, mask}], best first
    cgn = ModelClient(CGN_ENDPOINT)
    res = cgn.grasps(depth, K, segmap)              # {pred_grasps_cam, scores, contact_pts, gripper_openings}

Needs pyzmq in the system Python (sudo apt install python3-zmq). A timed-out request leaves a ZMQ
REQ socket unusable, so the socket is recreated after every failure.
"""

import pickle

import numpy as np

GSAM_ENDPOINT = "tcp://127.0.0.1:5556"
CGN_ENDPOINT = "tcp://127.0.0.1:5555"


class ModelServerError(RuntimeError):
    pass


class ModelClient:
    def __init__(self, endpoint, timeout_s=10.0):
        try:
            import zmq
        except ImportError as e:
            raise ModelServerError("pyzmq missing in the system Python: sudo apt install python3-zmq") from e
        self.zmq, self.endpoint, self.timeout_s = zmq, endpoint, timeout_s
        self.ctx = zmq.Context.instance()
        self.sock = None

    def _socket(self):
        if self.sock is None:
            self.sock = self.ctx.socket(self.zmq.REQ)
            self.sock.setsockopt(self.zmq.LINGER, 0)
            self.sock.connect(self.endpoint)
        return self.sock

    def _reset(self):
        if self.sock is not None:
            self.sock.close(0)
            self.sock = None

    def call(self, req, timeout_s=None):
        s = self._socket()
        ms = int(1000 * (self.timeout_s if timeout_s is None else timeout_s))
        try:
            s.send(pickle.dumps(req, protocol=pickle.HIGHEST_PROTOCOL))
            if not s.poll(ms):
                raise ModelServerError(f"no reply from {self.endpoint} within {ms / 1000:.1f} s "
                                       "(model server not running? ros2 run steve_perception model_server ...)")
            rep = pickle.loads(s.recv())
        except ModelServerError:
            self._reset()
            raise
        except Exception as e:
            self._reset()
            raise ModelServerError(f"{self.endpoint}: {e}") from e
        if rep.get("error"):
            raise ModelServerError(f"{self.endpoint}: {rep['error']}")
        return rep

    def ping(self, timeout_s=1.0):
        """The served model's name, or None if no server answers."""
        try:
            return self.call({"op": "ping"}, timeout_s)["model"]
        except ModelServerError:
            return None

    def segment(self, rgb, prompt, box_threshold=0.35, text_threshold=0.25, max_dets=5):
        return self.call({"op": "segment", "rgb": np.ascontiguousarray(rgb, np.uint8), "prompt": prompt,
                          "box_threshold": box_threshold, "text_threshold": text_threshold,
                          "max_dets": max_dets})["detections"]

    def grasps(self, depth, K, segmap=None, z_range=(0.2, 1.5), forward_passes=1, local_regions=True,
               filter_grasps=True):
        return self.call({"op": "grasp", "depth": np.asarray(depth, np.float32), "K": np.asarray(K, float),
                          "segmap": None if segmap is None else np.asarray(segmap, np.int32),
                          "z_range": list(z_range), "forward_passes": forward_passes,
                          "local_regions": local_regions, "filter_grasps": filter_grasps})

    def close(self):
        self._reset()
