# Use CUDA-enabled base image
FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04

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
    pip install -e .

# Set the entry point
CMD ["bash"]