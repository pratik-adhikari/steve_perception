# SAM3 Docker - Real SAM3 with Text-Prompted Segmentation
# Upgraded from SAM 2.1 to SAM3 for native text query support

FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# System dependencies (Python 3.10 is already in base image)
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    git \
    wget \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Ensure pip is up to date
RUN python3 -m pip install --upgrade pip

# Install PyTorch 2.5.1 with CUDA 12.4 (compatible with Python 3.10)
RUN pip3 install --no-cache-dir \
    torch==2.5.1 \
    torchvision==0.20.1 \
    torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu124

# Install additional dependencies needed for SAM3
RUN pip3 install --no-cache-dir \
    flask==3.0.0 \
    gunicorn==21.2.0 \
    pillow==11.0.0 \
    numpy==1.26.4 \
    opencv-python==4.10.0.84 \
    scipy==1.11.4 \
    matplotlib==3.8.0 \
    hydra-core==1.3.2 \
    iopath==0.1.10 \
    "huggingface_hub[cli]" \
    tqdm \
    ftfy \
    regex \
    einops \
    decord \
    timm \
    pycocotools \
    psutil

# Copy SAM3 source from parent directory
# Note: Build context must be set to parent directory
WORKDIR /app
COPY ../../../sam3 /app/sam3

# Install SAM3
WORKDIR /app/sam3
RUN pip3 install -e .

# Authenticate with HuggingFace and download SAM3 checkpoint
# Token will be provided as build arg
ARG HF_TOKEN
ENV HF_TOKEN=${HF_TOKEN}

RUN if [ -n "$HF_TOKEN" ]; then \
    mkdir -p /app/sam3/checkpoints && \
    python3 -c "from huggingface_hub import login, snapshot_download; login(token='${HF_TOKEN}'); snapshot_download('facebook/sam3', local_dir='/app/sam3/checkpoints', local_dir_use_symlinks=False)"; \
    else \
    echo "WARNING: HF_TOKEN not provided. Model download skipped."; \
    fi

# Copy Flask server
WORKDIR /app
COPY docker/sam3/app.py /app/app.py

# Expose port
EXPOSE 5005

# Run server
CMD ["python3", "app.py"]
