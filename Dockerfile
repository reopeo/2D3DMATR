# 2D3D-MATR: Docker image (no conda)
# Base: CUDA 11.7 devel (contains nvcc, needed for vision3d and pykeops CUDA extensions)
FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8

# ── System dependencies ─────────────────────────────────────────────────────
# Python 3.8 is installed from the deadsnakes PPA
# (Ubuntu 22.04 ships Python 3.10 by default)
# libeigen3-dev is required to compile vision3d CUDA extensions
# cmake is required by pykeops
RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y \
        python3.8 \
        python3.8-dev \
        python3.8-venv \
        git \
        wget \
        curl \
        cmake \
        ninja-build \
        build-essential \
        libeigen3-dev \
        libgl1-mesa-glx \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ── Python virtual environment (Python 3.8) ─────────────────────────────────
RUN python3.8 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --upgrade pip setuptools wheel

# ── PyTorch 1.13.1 + CUDA 11.7 ──────────────────────────────────────────────
RUN pip install \
    torch==1.13.1+cu117 \
    torchvision==0.14.1+cu117 \
    torchaudio==0.13.1+cu117 \
    --extra-index-url https://download.pytorch.org/whl/cu117

# ── torch-scatter (must match torch + CUDA version) ─────────────────────────
RUN pip install torch-scatter \
    -f https://data.pyg.org/whl/torch-1.13.1+cu117.html

# ── Other Python dependencies (from vision3d requirements.txt) ───────────────
# NOTE: open3d==0.11.2 depends on the deprecated 'sklearn' PyPI package,
#       which fails with modern pip. Using open3d>=0.15.0 instead.
RUN pip install \
    numpy \
    scipy \
    tqdm \
    loguru \
    easydict \
    h5py \
    scikit-learn \
    scikit-image \
    einops \
    tensorboard \
    ipython \
    ipdb \
    opencv-python \
    pykeops \
    open3d

# ── vision3d (builds CUDA extensions via setup.py) ──────────────────────────
# TORCH_CUDA_ARCH_LIST must be set explicitly because Docker build runs without
# a GPU, so PyTorch cannot auto-detect the architecture (→ IndexError).
# Covers: Pascal(6.x), Volta(7.0), Turing(7.5), Ampere(8.0/8.6).
# Add "8.9" for Ada Lovelace (RTX 4000) or "9.0" for Hopper (H100) if needed.
ENV TORCH_CUDA_ARCH_LIST="8.6"

RUN git clone https://github.com/qinzheng93/vision3d.git /opt/vision3d

# vision3d/requirements.txt pins open3d==0.11.2, which depends on the
# deprecated 'sklearn' package. setup.py develop uses easy_install to
# re-resolve dependencies, causing a failure even after we installed a
# newer open3d via pip. Remove the version pin so easy_install accepts
# the already-installed open3d.
RUN sed -i 's/^open3d==0\.11\.2$/open3d/' /opt/vision3d/requirements.txt

WORKDIR /opt/vision3d
RUN python setup.py develop

# ── Project ──────────────────────────────────────────────────────────────────
WORKDIR /workspace
COPY . /workspace
