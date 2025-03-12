# Comprehensive Dockerfile Setup for Lambda Cloud GPU Integration

This document outlines a complete Dockerfile setup tailored for profiling and optimizing inference latency and memory retrieval for state-space models (SSMs), specifically targeting [Llamba-3B](https://huggingface.co/cartesia-ai/Llamba-3B), using CUDA and Triton kernels integrated with [Mem0](https://github.com/jameshaydon/mem0) and [Cartesia Edge](https://github.com/cartesia-ai/edge).

---

## Dockerfile Setup in VSCode

### Step 1: Create and Edit Dockerfile

- Open your project in VSCode.
- Create a new file named `Dockerfile` in your root directory (`mamba_kernel_optimization`).
- Paste the following Dockerfile clearly into VSCode:

```Dockerfile
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

# Install PyTorch (CUDA 12.1)
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install Triton
RUN pip3 install triton

# Install NVIDIA Nsight Systems for GPU profiling
RUN wget -qO- https://developer.download.nvidia.com/devtools/repos/ubuntu2204/amd64/nsight-systems-2024.1.list | tee /etc/apt/sources.list.d/nsight-systems.list \
    && apt-get update && apt-get install -y nsight-systems-2024.1 \
    && rm -rf /var/lib/apt/lists/*

# Copy the repository into the container
COPY . /workspace/mamba_kernel_optimization

# Install local submodules: Mem0 and Edge
RUN pip3 install ./mamba_kernel_optimization/mem0 \
                 ./mamba_kernel_optimization/edge

CMD ["/bin/bash"]
```

- Save your file clearly (`Cmd+S`).

---

## Test Docker Image Locally (Apple M3 Pro)

### Step 2: Build Docker Image (Specify Platform)

Ensure you specify the `--platform linux/amd64` clearly in your Docker build command to handle cross-architecture compatibility:

```bash
docker build --platform linux/amd64 -t mamba_kernel_optimization .
```

### Verify Docker Image

```bash
docker images
```

### Run Docker Container

```bash
docker run --gpus all -it mamba_kernel_optimization
```

### Inside Container: Verify GPU Access

```bash
nvidia-smi
python3 -c 'import torch; print(torch.cuda.is_available())'
```

---

## Lambda Cloud GPU Integration

### Step 3: Launch and Setup Lambda Cloud GPU Instance

- Register and create a GPU instance at [Lambda Cloud](https://lambdalabs.com/service/gpu-cloud).
- Add your SSH public key via the Lambda Cloud dashboard.

Lambda will provide an SSH command similar to:

```bash
ssh -i /path/to/your_private_key ubuntu@your_lambda_instance_ip
```

Replace clearly the placeholders (`your_private_key`, `your_lambda_instance_ip`) once obtained from Lambda Cloud.

---

### Step 4: Deploy Docker Image to Lambda Cloud

#### On Lambda Ubuntu Instance:

Install Docker and GPU support:

```bash
sudo apt-get update
sudo apt-get install -y docker.io

# Enable Docker GPU support
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

Pull and run your Docker image:

```bash
sudo docker pull your_dockerhub_username/mamba_kernel_optimization:latest
sudo docker run --gpus all -it your_dockerhub_username/mamba_kernel_optimization:latest
```

Replace clearly your Docker Hub username.

---

## Components to Update Post-Instance Setup

Replace the following placeholders provided by Lambda clearly:

- **SSH Key Path:** `/path/to/your_private_key`
- **Lambda Instance IP:** `your_lambda_instance_ip`
- **Docker Registry Username**: `your_dockerhub_username`

---

## Additional Profiling and Documentation Resources

- [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)
- [CUDA Profiler](https://developer.nvidia.com/nvidia-visual-profiler)
- [CUDA Profiling Tools Interface (CUPTI)](https://developer.nvidia.com/cuda-profiling-tools-interface)
- [Nsight Systems](https://developer.nvidia.com/nsight-systems)
- [Mamba](https://github.com/state-spaces/mamba)
- [Llamba-3B](https://huggingface.co/cartesia-ai/Llamba-3B)
- [Cartesia](https://cartesia.ai)
- [Edge - On-Device SSMs](https://github.com/cartesia-ai/edge)

---

This revised setup clearly addresses additional considerations for Apple M3 Pro users and ensures smooth Docker operations for GPU-based development workflows.

