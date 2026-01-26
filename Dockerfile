# OpenYOLO3D with CUDA 11.3 support, conda-based
FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
ENV PYTHONUNBUFFERED=1
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 6.2 7.0 7.2 7.5 8.0 8.6"
ENV FORCE_CUDA=1
ENV CUDA_HOME=/usr/local/cuda
ENV MAX_JOBS=1

# ----------------------------------------------------------------------
# 1. System deps
# ----------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    git \
    build-essential \
    cmake \
    ninja-build \
    libopenblas-dev \
    libblas-dev \
    libgl1 \
    libglib2.0-0 \
    libxrender1 \
    libsm6 \
    ca-certificates \
    bash \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# ----------------------------------------------------------------------
# 2. Install Miniconda
# ----------------------------------------------------------------------
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-py310_23.9.0-0-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh

ENV PATH=/opt/conda/bin:$PATH

# Ensure we use bash for conda commands
SHELL ["bash", "-lc"]

# ----------------------------------------------------------------------
# 3. Copy environment.yml and create conda env
# ----------------------------------------------------------------------
# Copy from submodule (assuming build context is steve_perception root)
COPY source/models/lib/OpenYOLO3D/environment.yml /tmp/environment.yml

# Now create the environment
RUN conda env create -f /tmp/environment.yml && \
    conda clean -afy

# Set default env
ENV CONDA_DEFAULT_ENV=openyolo3d
ENV PATH=/opt/conda/envs/openyolo3d/bin:$PATH


# ----------------------------------------------------------------------
# 4. Copy the OpenYOLO3D Project files
# ----------------------------------------------------------------------
# We copy the submodule content to /workspace/OpenYOLO3D
WORKDIR /workspace/OpenYOLO3D
COPY source/models/lib/OpenYOLO3D .

# ----------------------------------------------------------------------
# 5. Follow README: install torch, scatter, detectron2, Mask3D deps
# ----------------------------------------------------------------------
WORKDIR /workspace/OpenYOLO3D/models/Mask3D

# PyTorch + scatter (inside conda env)
RUN pip install --no-cache-dir \
    torch==1.11.0+cu113 \
    torchvision==0.12.0+cu113 \
    --extra-index-url https://download.pytorch.org/whl/cu113 && \
    pip install --no-cache-dir \
    torch-scatter -f https://data.pyg.org/whl/torch-1.11.0+cu113.html

# Detectron2 (use prebuilt wheel for cu113/torch1.11)
RUN pip install --no-cache-dir detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.11/index.html

# ----------------------------------------------------------------------
# 6. Third-party for Mask3D: MinkowskiEngine, ScanNet, pointnet2
# ----------------------------------------------------------------------
WORKDIR /workspace/OpenYOLO3D/models/Mask3D/third_party

RUN git clone --recursive "https://github.com/NVIDIA/MinkowskiEngine" && \
    cd MinkowskiEngine && \
    git checkout 02fc608bea4c0549b0a7b00ca1bf15dee4a0b228 && \
    python setup.py install --force_cuda --blas=openblas

RUN git clone https://github.com/ScanNet/ScanNet.git && \
    cd ScanNet/Segmentator && \
    git checkout 3e5726500896748521a6ceb81271b0f5b2c0e7d2 && \
    make

RUN cd pointnet2 && python setup.py install

# ----------------------------------------------------------------------
# 7. Remaining Python deps exactly as README
# ----------------------------------------------------------------------
WORKDIR /workspace/OpenYOLO3D/models/Mask3D

RUN pip install --no-cache-dir pytorch-lightning==1.7.2 && \
    pip install --no-cache-dir \
    black==21.4b2 \
    cloudpickle==3.0.0 \
    future \
    hydra-core==1.0.5 \
    "pycocotools>=2.0.2" \
    pydot \
    iopath==0.1.7 \
    loguru \
    albumentations && \
    pip install --no-cache-dir .

# ----------------------------------------------------------------------
# 8. YOLO-World, mmdet, mmyolo, mmcv, open3d, etc.
# ----------------------------------------------------------------------
WORKDIR /workspace/OpenYOLO3D

# 1. Install OpenMIM first
RUN pip install --no-cache-dir openmim

# 2. Force install the correct MMCV binary for CUDA 11.3
# Fix numpy/pandas/sklearn/scipy binary incompatibility (pin versions to match env, no upgrades)
RUN pip install --no-deps --force-reinstall "numpy==1.24.2" "pandas==2.0.0" "scikit-learn" "scipy"

# Re-assert torch version to prevent openmim/pip from having upgraded it to an incompatible version
RUN pip install --no-cache-dir torch==1.11.0+cu113 torchvision==0.12.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113

RUN mim install "mmcv==2.0.0"

# 3. Install other heavy deps that might trigger builds
RUN pip install --no-cache-dir mmdet==3.0.0 mmyolo==0.6.0

# 4. NOW install YOLO-World
WORKDIR /workspace/OpenYOLO3D/models/YOLO-World
RUN pip install --no-cache-dir --no-build-isolation -e .

# 5. Install remaining pure-python deps
WORKDIR /workspace/OpenYOLO3D
RUN pip install --no-cache-dir \
    plyfile \
    open3d \
    pillow \
    pyviz3d \
    supervision==0.19.0 \
    shapely \
    transformers==4.30.0 \
    gdown

# ----------------------------------------------------------------------
# 9. Download checkpoints (Optional but good for readiness)
# ----------------------------------------------------------------------
RUN chmod +x scripts/get_class_agn_masks.sh scripts/get_checkpoints.sh && \
    ./scripts/get_class_agn_masks.sh && \
    ./scripts/get_checkpoints.sh

# ----------------------------------------------------------------------
# 10. Runtime Setup
# ----------------------------------------------------------------------
ENV PYTHONPATH="/workspace/OpenYOLO3D:${PYTHONPATH}"

# Add any additional Steve Perception dependencies here if needed
# (e.g., if you need bosdyn or other utils, pip install them or copy source)

WORKDIR /workspace/OpenYOLO3D
CMD ["bash"]
