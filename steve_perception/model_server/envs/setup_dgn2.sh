#!/usr/bin/env bash
# setup_dgn2.sh — DexGraspNet 2.0 for steve_perception's model server `dgn2` (see ../README.md). Re-runnable: finished
# steps are skipped. ~30-60 min, mostly building MinkowskiEngine; ~12 GB (env ~6 GB, checkpoints 6.2 GB download).
#
#   bash setup_dgn2.sh                       # workspace ~/Master_Thesis/steve_ros2_ws
#   WS=/path/to/steve_ros2_ws bash setup_dgn2.sh
#
# Everything runs with the env's own python and the user site disabled (a broken torch in ~/.local otherwise
# shadows the env's). Steps:
#   1. DexGraspNet2 repo -> $WS/perception_tests/DexGraspNet2 (moved from ~/DexGraspNet2 if cloned there)
#   2. conda env dgn2_env (dgn2_env.yml), PyTorch 2.0.1 + CUDA 11.7 runtime, CUDA 11.7 compiler (nvcc) in the env
#   3. pytorch3d 0.7.5 (prebuilt py38_cu117_pyt201, as the repo's README)
#   4. torchprimitivesdf (compiled), MinkowskiEngine (compiled, for the RTX 3070: sm_86)
#   5. the dexterous checkpoint (the tar's OURS) -> experiments/dex_ours/{config.yaml, ckpt/ckpt_50000.pth}
#   6. selftest: the model on a synthetic view
set -euo pipefail
WS=${WS:-$HOME/Master_Thesis/steve_ros2_ws}
PT=$WS/perception_tests
REPO=$PT/DexGraspNet2
ENV=dgn2_env
HERE=$(cd "$(dirname "$0")" && pwd)
export PYTHONNOUSERSITE=1
step() { echo -e "\n=== $* ==="; }
fail() { echo -e "\n[FAILED] $*" >&2; exit 1; }

command -v conda >/dev/null || fail "conda not found (source ~/anaconda3/etc/profile.d/conda.sh)"
CONDA_BASE=$(conda info --base)
PREFIX=$CONDA_BASE/envs/$ENV
PY=$PREFIX/bin/python

step "1/6 repo -> $REPO"
mkdir -p "$PT" && touch "$PT/COLCON_IGNORE"
if [ ! -d "$REPO/src" ]; then
  if [ -d "$HOME/DexGraspNet2/src" ]; then mv "$HOME/DexGraspNet2" "$REPO"
  else git clone https://github.com/PKU-EPIC/DexGraspNet2 "$REPO"; fi
fi

step "2/6 conda env $ENV"
# the right torch: 2.0.1, CUDA 11.7, old C++ ABI (the pytorch channel's — what the prebuilt pytorch3d is built for),
# and no second torch from pip (nvidia-*-cu12 / triton wheels)
TORCH_OK="import torch, sys, importlib.metadata as m
names = {d.metadata['Name'].lower() for d in m.distributions()}
# (triton 2.0.0 = the pytorch channel's torchtriton, a dependency of the right build; pip's torch brings 3.x)
bad = sorted(n for n in names if n.startswith('nvidia-') or (n == 'triton' and not m.version(n).startswith('2.0.')))
ok = torch.__version__ == '2.0.1' and torch.version.cuda == '11.7' and not torch.compiled_with_cxx11_abi() and not bad
print('torch', torch.__version__, 'cuda', torch.version.cuda, 'cxx11_abi', torch.compiled_with_cxx11_abi(), 'pip extras', bad)
sys.exit(0 if ok else 1)"
if [ -x "$PY" ] && ! "$PY" -c "$TORCH_OK"; then
  echo "$ENV has the wrong torch build (see above) -> removing and recreating it"
  conda env remove -n $ENV -y
fi
if [ ! -x "$PY" ]; then conda env create -f "$HERE/dgn2_env.yml"; fi
"$PY" -c "$TORCH_OK" || fail "$ENV: torch is not the pytorch channel's 2.0.1 + CUDA 11.7 build (see the line above)"
if ! "$PY" -c "import nflows.nn.nets.resnet" 2>/dev/null; then
  "$PY" -m pip install --no-deps nflows==0.14   # pure torch; its dependencies would pull a second torch
fi
if [ ! -x "$PREFIX/bin/nvcc" ]; then
  conda install -n $ENV -y --freeze-installed -c "nvidia/label/cuda-11.7.1" cuda-nvcc cuda-cudart-dev cuda-libraries-dev
  "$PY" -c "$TORCH_OK" || fail "installing the CUDA compiler changed torch"
fi
"$PY" -c "import torch; assert torch.cuda.is_available(), 'no CUDA in torch'; print('torch', torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))" \
  || fail "PyTorch with CUDA is not working in $ENV"
"$PREFIX/bin/nvcc" --version | tail -2

export CUDA_HOME=$PREFIX TORCH_CUDA_ARCH_LIST="8.6" MAX_JOBS=${MAX_JOBS:-6}
export LDFLAGS="-Wl,-rpath,$PREFIX/lib ${LDFLAGS:-}"   # the extensions find libmkl_rt etc. in the env at runtime
export CPATH=$PREFIX/targets/x86_64-linux/include:$PREFIX/include:${CPATH:-}
export LIBRARY_PATH=$PREFIX/targets/x86_64-linux/lib:$PREFIX/lib:${LIBRARY_PATH:-}

step "3/6 pytorch3d 0.7.5"
if ! "$PY" -c "import pytorch3d.ops" 2>/dev/null; then
  PKG=pytorch3d-0.7.5-py38_cu117_pyt201.tar.bz2
  [ -f "$PT/$PKG" ] || curl -L --fail -o "$PT/$PKG" "https://conda.anaconda.org/pytorch3d/linux-64/$PKG" \
    || curl -L --fail -o "$PT/$PKG" "https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch3d/linux-64/$PKG"
  conda install -n $ENV -y --offline "$PT/$PKG"
fi
"$PY" -c "import pytorch3d.ops, pytorch3d.transforms; print('pytorch3d ok')" || fail "pytorch3d"

step "4/6 torchprimitivesdf + MinkowskiEngine (compiling)"
mkdir -p "$PT/build"
if ! "$PY" -c "import torchprimitivesdf" 2>/dev/null; then
  [ -d "$PT/build/torchprimitivesdf" ] || git clone https://github.com/mzhmxzh/torchprimitivesdf "$PT/build/torchprimitivesdf"
  (cd "$PT/build/torchprimitivesdf" && "$PY" -m pip install --no-build-isolation -e .) || fail "torchprimitivesdf build"
fi
if ! "$PY" -c "import MinkowskiEngine" 2>/dev/null; then
  [ -d "$PT/build/MinkowskiEngine" ] || git clone https://github.com/NVIDIA/MinkowskiEngine "$PT/build/MinkowskiEngine"
  # MKL (torch's BLAS); --blas_library_dirs is not used: ME's setup.py puts the list's repr into the rpath —
  # LIBRARY_PATH / LDFLAGS above give the env's lib dir instead
  (cd "$PT/build/MinkowskiEngine" && "$PY" setup.py install --blas=mkl --blas_include_dirs="$PREFIX/include" \
      2>&1 | tee "$PT/build/minkowski_build.log" | tail -5) \
    && "$PY" -c "import MinkowskiEngine" || fail "MinkowskiEngine build — send $PT/build/minkowski_build.log"
fi
"$PY" -c "import MinkowskiEngine as ME, torchprimitivesdf; print('MinkowskiEngine', ME.__version__)"

step "5/6 checkpoint (dex_ours)"
CK=$REPO/experiments/dex_ours
if [ ! -f "$CK/ckpt/ckpt_50000.pth" ]; then
  TAR=$PT/DexGraspNet2.0-ckpts.tar
  # (a complete tar is not downloaded again: resuming a finished download makes curl fail)
  if [ ! -f "$TAR" ] || ! tar -tf "$TAR" > "$PT/ckpts_contents.txt" 2>/dev/null; then
    curl -L --fail -C - -o "$TAR" "https://huggingface.co/datasets/lhrlhr/DexGraspNet2.0/resolve/main/DexGraspNet2.0-ckpts.tar"
    tar -tf "$TAR" > "$PT/ckpts_contents.txt"
  fi
  # the paper's dexterous model is DexGraspNet2.0-ckpts/OURS (LEAP hand, graspness_diffusion); OURS_gripper is the
  # parallel-jaw one, the rest ablations / baselines / scaling studies
  CFG=$(grep -E '(^|/)(OURS|dex_ours)/config\.yaml$' "$PT/ckpts_contents.txt" | head -1 || true)
  PTH=$(grep -E '(^|/)(OURS|dex_ours)/ckpt/ckpt_50000\.pth$' "$PT/ckpts_contents.txt" | head -1 || true)
  [ -n "$CFG" ] && [ -n "$PTH" ] || fail "no OURS/config.yaml + ckpt/ckpt_50000.pth in the tar (contents: $PT/ckpts_contents.txt)"
  mkdir -p "$PT/ckpt_x" && tar -xf "$TAR" -C "$PT/ckpt_x" "$CFG" "$PTH"
  mkdir -p "$CK/ckpt" && mv "$PT/ckpt_x/$CFG" "$CK/config.yaml" && mv "$PT/ckpt_x/$PTH" "$CK/ckpt/ckpt_50000.pth"
  rm -rf "$PT/ckpt_x"
  echo "checkpoint in $CK (the tar stays in $TAR: other checkpoints; delete it to free 6.2 GB)"
fi

"$PY" -c "import nflows.nn.nets.resnet, diffusers, einops, urdf_parser_py, transforms3d, trimesh, ikpy, plotly, cv2, zmq" \
  || fail "a python dependency is missing in $ENV (see the error above)"

step "6/6 selftest (synthetic view)"
"$PY" "$HERE/../dgn2.py" --repo "$REPO" --ckpt "$CK/ckpt/ckpt_50000.pth" || fail "selftest"
echo -e "\nDexGraspNet 2.0 ready: ros2 run steve_perception model_server dgn2"
