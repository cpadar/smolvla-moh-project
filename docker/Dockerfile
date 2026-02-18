# Base image: PyTorch 2.7.1 with CUDA 12.6, Ubuntu 22.04
# Python 3.10 — verified working with LeRobot smolVLA and ManiSkill3
FROM pytorch/pytorch:2.7.1-cuda12.6-cudnn9-runtime

LABEL maintainer="your-team"
LABEL description="SmolVLA + Mixture of Horizons research environment"

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    git-lfs \
    wget \
    curl \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libegl1-mesa-dev \
    libgles2-mesa-dev \
    libglfw3-dev \
    libvulkan1 \
    vulkan-tools \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Install Python dependencies in order of stability
# 1. ManiSkill3
RUN pip install --no-cache-dir mani_skill

# 2. LeRobot with SmolVLA extras
RUN git clone https://github.com/huggingface/lerobot.git /opt/lerobot && \
    cd /opt/lerobot && \
    pip install --no-cache-dir -e ".[smolvla]"

# 3. Additional dependencies for training and logging
RUN pip install --no-cache-dir \
    wandb \
    omegaconf \
    hydra-core \
    einops \
    huggingface_hub \
    accelerate \
    tensorboard

# 4. Copy project source
COPY . /workspace

# Environment variables
ENV PYTHONPATH="/workspace:/opt/lerobot:${PYTHONPATH}"
ENV MPLBACKEND=Agg

# Default command — opens a shell
CMD ["/bin/bash"]
