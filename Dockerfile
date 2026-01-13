FROM osrf/ros:humble-desktop

# Set environment
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# System dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    git \
    wget \
    curl \
    vim \
    unzip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies for Steve Perception & OpenMask3D Client
RUN pip3 install --no-cache-dir \
    torch \
    torchvision \
    "numpy<2.0.0" \
    scipy \
    open3d \
    matplotlib \
    pillow \
    pyyaml \
    ftfy \
    regex \
    tqdm

# Install CLIP (OpenAI)
RUN pip3 install git+https://github.com/openai/CLIP.git

# Install Steve Perception as a package (optional, or just mount it)
# WORKDIR /ws/src/steve_perception
# COPY . .
# RUN pip3 install -e .

WORKDIR /ws
