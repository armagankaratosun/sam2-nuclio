# Base image - CUDA PyTorch
FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel

# Set environment (important for GPU detection inside the container)
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV DEBIAN_FRONTEND=noninteractive
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6 8.9+PTX"

# Set working directory 
WORKDIR /opt/nuclio

# Install system dependencies and clean up
RUN apt-get update && \
    apt-get install -y git wget && \
    rm -rf /var/lib/apt/lists/*

# Clone and install SAM2 (specific commit) - will fix this in the future.
RUN git clone https://github.com/facebookresearch/segment-anything-2.git /opt/nuclio/segment-anything-2 && \
    cd /opt/nuclio/segment-anything-2 && \
    git checkout 0f6515ae853c40420ea8e3dd250f8031bbf03023

# Download checkpoints
RUN cd /opt/nuclio/segment-anything-2/checkpoints && ./download_ckpts.sh && \
    mkdir -p /opt/nuclio/segment-anything-2/checkpoints && \
    wget -O /opt/nuclio/segment-anything-2/checkpoints/sam2_hiera_large.pt \
    https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt

# Copy Requirements
COPY requirements.txt /opt/nuclio/

# Install Python dependencies and SAM2 package in editable mode
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r /opt/nuclio/requirements.txt && \
    pip install -e /opt/nuclio/segment-anything-2

# Copy required objects from Nuclio's onbuild image
COPY --from=quay.io/nuclio/handler-builder-python-onbuild:1.13.0-amd64 /home/nuclio/bin/processor /usr/local/bin/processor
COPY --from=quay.io/nuclio/handler-builder-python-onbuild:1.13.0-amd64 /home/nuclio/bin/py /opt/nuclio/
COPY --from=quay.io/nuclio/handler-builder-python-onbuild:1.13.0-amd64 /home/nuclio/bin/py*-whl/* /opt/nuclio/whl/

# Copy handler.py and model.py
COPY handler.py /opt/nuclio/
COPY model.py /opt/nuclio/

# Copy sam2 configs
RUN cp -r /opt/nuclio/segment-anything-2/sam2_configs /opt/nuclio/configs

# Ensure SAM2 is in the PYTHONPATH
ENV PYTHONPATH="/opt/nuclio/segment-anything-2"

# Set the entry point for Nuclio
CMD ["processor"]
