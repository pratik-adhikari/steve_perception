#!/usr/bin/env python3
"""
server.py — perception model server: one GPU model, kept loaded between requests and WBC runs.
Started by `ros2 run steve_perception model_server <model>` (launcher.py), each in its own env:

  gsam : GroundingDINO + SAM   text prompt -> instance masks       gsam_env   tcp://127.0.0.1:5556
  cgn  : Contact-GraspNet      depth (+ target segment) -> grasps  cgn_env    tcp://127.0.0.1:5555

ZMQ REP, pickled dicts (cgn: the protocol of DiNBV-Grasp's graspnet_server, extended by "op"):
  {"op": "ping"}                                    -> {"model": "gsam" | "cgn"}
  gsam {"op": "segment", "rgb", "prompt", ["box_threshold", "text_threshold", "max_dets"]}
                                                    -> {"detections": [{box, score, phrase, mask}]}
  cgn  {"op": "grasp" (or no op), "depth", "K", ["segmap", "z_range", "forward_passes",
        "local_regions", "filter_grasps"]}          -> {pred_grasps_cam, scores, contact_pts, gripper_openings}
Every reply has "error" (None or a message) and "t" (seconds spent).

Pickle executes code on load: the server binds to 127.0.0.1 by default — expose it on a network
only between trusted machines.
"""

import argparse
import os
import pickle
import signal
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

DEFAULT_BIND = {"gsam": "tcp://127.0.0.1:5556", "cgn": "tcp://127.0.0.1:5555"}


def main():
    p = argparse.ArgumentParser(description="steve_perception model server (one model per process)")
    p.add_argument("model", choices=sorted(DEFAULT_BIND))
    p.add_argument("--bind", default=None, help="default: gsam tcp://127.0.0.1:5556, cgn tcp://127.0.0.1:5555")
    p.add_argument("--device", default="cuda")
    g = p.add_argument_group("gsam")
    g.add_argument("--gdino_repo")
    g.add_argument("--gdino_config", help="default: <gdino_repo>/groundingdino/config/GroundingDINO_SwinT_OGC.py")
    g.add_argument("--gdino_weights")
    g.add_argument("--gdino_image_size", type=int, default=800, help="shorter image side for GroundingDINO")
    g.add_argument("--sam_repo")
    g.add_argument("--sam_weights")
    g.add_argument("--sam_type", default="vit_b")
    g.add_argument("--fp32", action="store_true", help="no fp16 autocast (slower, twice the memory)")
    g.add_argument("--allow_download", action="store_true",
                   help="let transformers contact the Hugging Face hub (default: offline, cached BERT only)")
    g = p.add_argument_group("cgn")
    g.add_argument("--cgn_repo")
    g.add_argument("--cgn_ckpt", help="default: <cgn_repo>/checkpoints/contact_graspnet")
    a = p.parse_args()
    bind = a.bind or DEFAULT_BIND[a.model]

    import zmq

    # bind first: a second server for the same port (one already running) stops before loading
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REP)
    try:
        sock.bind(bind)
    except zmq.ZMQError as e:
        print(f"[model_server] {a.model}: cannot bind {bind} ({e}) — a server is probably running already "
              "(launch with model_servers:=false to use it)", flush=True)
        sock.close(0)
        ctx.term()
        return

    if a.model == "gsam":
        missing = [n for n in ("gdino_repo", "gdino_weights", "sam_repo", "sam_weights") if not getattr(a, n)]
        if missing:
            sys.exit(f"[model_server] gsam needs --{', --'.join(missing)}")
        if not a.allow_download:  # BERT (GroundingDINO's text encoder) from the local cache
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        from gsam import GroundedSAM
        cfg = a.gdino_config or str(Path(a.gdino_repo) / "groundingdino/config/GroundingDINO_SwinT_OGC.py")
        model = GroundedSAM(a.gdino_repo, cfg, a.gdino_weights, a.sam_repo, a.sam_weights, a.sam_type,
                            a.device, a.gdino_image_size, fp16=not a.fp32)
    else:
        if not a.cgn_repo:
            sys.exit("[model_server] cgn needs --cgn_repo")
        from cgn import ContactGraspNet
        model = ContactGraspNet(a.cgn_repo, a.cgn_ckpt, a.device)

    def handle(req):
        op = req.get("op", "grasp")
        if op == "ping":
            return {"model": a.model}
        if a.model == "gsam" and op == "segment":
            return {"detections": model.segment(req["rgb"], req["prompt"], req.get("box_threshold", 0.35),
                                                req.get("text_threshold", 0.25), req.get("max_dets", 5))}
        if a.model == "cgn" and op == "grasp":
            return model.grasps(req["depth"], req["K"], req.get("segmap"), req.get("rgb"),
                                req.get("z_range", (0.2, 1.5)), req.get("forward_passes", 1),
                                req.get("local_regions", True), req.get("filter_grasps", True))
        raise ValueError(f"op '{op}' not served by the {a.model} server")

    print(f"[model_server] {a.model} listening on {bind}", flush=True)

    stop = []
    signal.signal(signal.SIGTERM, lambda *_: stop.append(1))
    n, t_sum, t_report = 0, 0.0, time.monotonic()
    try:
        while not stop:
            if not sock.poll(200):
                continue
            raw = sock.recv()
            t0, req = time.monotonic(), {}
            try:
                req = pickle.loads(raw)
                rep = handle(req)
                rep["error"] = None
            except Exception as e:  # report to the client, keep serving
                traceback.print_exc()
                rep = {"error": f"{type(e).__name__}: {e}"}
            rep["t"] = time.monotonic() - t0
            sock.send(pickle.dumps(rep, protocol=pickle.HIGHEST_PROTOCOL))
            op = req.get("op", "grasp") if isinstance(req, dict) else "?"
            if rep["error"]:
                print(f"[model_server] {op}: ERROR {rep['error']}", flush=True)
            elif op == "grasp":
                print(f"[model_server] grasp: {rep['t'] * 1000:.0f} ms, "
                      f"{sum(len(v) for v in rep['scores'].values())} grasp(s)", flush=True)
            elif op == "segment":  # continuous (detector): a summary every 10 s
                n, t_sum = n + 1, t_sum + rep["t"]
                if time.monotonic() - t_report > 10.0:
                    print(f"[model_server] segment: {n} requests, mean {t_sum / n * 1000:.0f} ms", flush=True)
                    n, t_sum, t_report = 0, 0.0, time.monotonic()
    except KeyboardInterrupt:
        pass
    finally:
        sock.close(0)
        ctx.term()
        print(f"[model_server] {a.model} stopped", flush=True)


if __name__ == "__main__":
    main()
