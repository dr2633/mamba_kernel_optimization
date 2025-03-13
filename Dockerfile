FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# Install system packages and CUDA profiling tools
RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-venv git curl vim wget \
    cuda-command-line-tools-12-1 \
    cuda-tools-12-1 \
    cuda-cupti-12-1 \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables for CUDA tools
ENV PATH=/usr/local/cuda-12.1/bin${PATH:+:${PATH}}
ENV LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

# Working directory
WORKDIR /workspace

# Upgrade pip, Install PyTorch (CUDA 12.1), Triton, cartesia-pytorch, and mem0ai (with increased timeout)
RUN pip3 install --upgrade pip \
    && pip3 install --default-timeout=300 torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 \
    && pip3 install --default-timeout=300 triton cartesia-pytorch mem0ai


# Copy repository into the containera
COPY . /workspace/mamba_kernel_optimization

# Install local submodules: Mem0 and Edge
RUN pip3 install ./mamba_kernel_optimization/mem0 \
                 ./mamba_kernel_optimization/edge

# Set final working directory clearly
WORKDIR /workspace/mamba_kernel_optimization

CMD ["/bin/bash"]
