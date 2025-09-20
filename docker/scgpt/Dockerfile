# Dockerfile for scGPT embedding generation
# Based on PyTorch with CUDA support for GPU inference

FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies
RUN pip install --no-cache-dir \
    scanpy==1.9.3 \
    anndata==0.9.1 \
    pandas \
    numpy \
    scipy \
    scikit-learn \
    h5py \
    pyarrow \
    tqdm \
    boto3 \
    duckdb \
    "flash-attn<1.0.5"

# Install scGPT from GitHub
RUN git clone https://github.com/bowang-lab/scGPT.git /tmp/scGPT && \
    cd /tmp/scGPT && \
    pip install . && \
    rm -rf /tmp/scGPT

# Create directories for data and models
RUN mkdir -p /data /models /output

# Copy our scripts (will be added later)
COPY scripts/generate_scgpt_embeddings.py /app/
COPY scripts/scgpt_wrapper.py /app/

# Set environment variables
ENV PYTHONPATH=/app:$PYTHONPATH
ENV CUDA_VISIBLE_DEVICES=0

# Default command
CMD ["python", "/app/scgpt_wrapper.py"]