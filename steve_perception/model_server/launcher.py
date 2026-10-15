#!/usr/bin/env python3
"""
launcher.py — `ros2 run steve_perception model_server gsam|cgn [--config FILE] [server options]`

Runs server.py with the interpreter of the model's conda env (config/model_server.yaml), replacing
this process (exec), so ros2 launch's SIGINT/SIGTERM reach the server directly. The ROS 2
PYTHONPATH is dropped and the user site (~/.local) disabled: their packages would shadow the env's.
"""

import os
import shutil
import sys
from pathlib import Path

import yaml

SERVER = Path(__file__).resolve().parent / "server.py"


def config_path(arg=None):
    if arg:
        return Path(arg).expanduser()
    try:
        from ament_index_python.packages import get_package_share_directory
        return Path(get_package_share_directory("steve_perception")) / "config" / "model_server.yaml"
    except Exception:  # not installed: the source tree
        return Path(__file__).resolve().parents[2] / "config" / "model_server.yaml"


def env_python(env):
    """A conda env name (searched in the usual conda roots) or an interpreter path."""
    p = Path(env).expanduser()
    if p.is_file():
        return p
    roots = [Path(os.environ["CONDA_EXE"]).parents[1]] if os.environ.get("CONDA_EXE") else []
    roots += [Path.home() / d for d in ("anaconda3", "miniconda3", "miniforge3", "mambaforge")]
    for r in roots:
        for exe in ("python3", "python"):
            c = r / "envs" / env / "bin" / exe
            if c.is_file() and os.access(c, os.X_OK):
                return c
    sys.exit(f"[model_server] conda env '{env}' not found in {', '.join(map(str, roots))} — create it "
             f"from {SERVER.parent / 'envs'} or set 'env' in the config to an interpreter path")


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    argv = argv[:argv.index("--ros-args")] if "--ros-args" in argv else argv  # added by ros2 launch
    if not argv or argv[0] not in ("gsam", "cgn"):
        sys.exit("usage: ros2 run steve_perception model_server gsam|cgn [--config FILE] [server options]")
    model, rest = argv[0], argv[1:]
    cfg_arg = None
    if "--config" in rest:
        i = rest.index("--config")
        cfg_arg, rest = rest[i + 1], rest[:i] + rest[i + 2:]
    path = config_path(cfg_arg)
    with open(path) as f:
        cfg = dict((yaml.safe_load(f) or {}).get(model) or {})
    py = env_python(cfg.pop("env", f"{model}_env"))

    args = [str(py), "-s", str(SERVER), model]
    for k, v in cfg.items():  # config keys = server options; command-line options come last and win
        v = os.path.expanduser(str(v))
        if k.endswith(("_repo", "_weights", "_ckpt", "_config")) and not Path(v).exists():
            sys.exit(f"[model_server] {model}.{k} in {path}: {v} does not exist")
        args += [f"--{k}", v]
    args += rest
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    # less memory held by the CUDA caching allocator between requests (two servers share 8 GB)
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    print(f"[model_server] {model}: {py} ({path.name})", flush=True)
    os.execve(str(py), args, env)


if __name__ == "__main__":
    main()
