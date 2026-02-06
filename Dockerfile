# STEVE Perception: Scanner Environment
# Base: ROS 2 Humble
# Includes: rtabmap-ros, navigation2, open3d, scipy

FROM osrf/ros:humble-desktop

# Set environment
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-humble-rtabmap-ros \
    ros-humble-navigation2 \
    ros-humble-rmw-cyclonedds-cpp \
    python3-pip \
    python3-dev \
    git \
    wget \
    curl \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Minimal Python dependencies for export scripts
RUN pip3 install --no-cache-dir \
    open3d \
    scipy \
    tqdm

# Setup ROS environment
RUN echo "source /opt/ros/humble/setup.bash" >> /root/.bashrc

WORKDIR /root/ws
CMD ["bash"]