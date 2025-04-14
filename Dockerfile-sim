# Use CUDA-enabled base image
FROM nvidia/cuda:12.6.3-devel-ubuntu22.04

# Set environment variables for CUDA.
ENV PATH=/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

# Install dependencies required for extracting the tarball.
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    ca-certificates \
    tar \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory.
WORKDIR /tmp

# Download cuDNN from NVIDIA's website
RUN wget -q https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/linux-x86_64/cudnn-linux-x86_64-8.9.5.30_cuda12-archive.tar.xz -O /tmp/cudnn-linux-x86_64-8.9.5.30_cuda12-archive.tar.xz

# Extract and install cuDNN.
RUN tar --xz -xvf cudnn-linux-x86_64-8.9.5.30_cuda12-archive.tar.xz && \
    cp -P cudnn-linux-x86_64-8.9.5.30_cuda12-archive/include/cudnn*.h /usr/local/cuda/include/ && \
    cp -P cudnn-linux-x86_64-8.9.5.30_cuda12-archive/lib/libcudnn* /usr/local/cuda/lib64/ && \
    chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn* && \
    rm -rf cudnn-linux-x86_64-8.9.5.30_cuda12-archive cudnn-linux-x86_64-8.9.5.30_cuda12-archive.tar.xz


# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    python3.10-venv \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create and activate virtual environment
RUN python3.10 -m venv $VIRTUAL_ENV

# Set working directory
WORKDIR /app

# Copy project files
COPY . /app/

# Install Python dependencies
RUN pip install -U pip setuptools wheel && \
    pip install pyyaml && \
    pip install --no-cache-dir -e . && \
    pip install --no-cache-dir --no-deps -r requirements.txt

# Set the entry point
CMD ["bash"]