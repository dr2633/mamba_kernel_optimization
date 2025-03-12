
[DeepGEMM](https://github.com/deepseek-ai/DeepGEMM)

[CUTLASS](https://github.com/NVIDIA/cutlass)

[CuTe's Support for Matrix-Multiply Accumulate](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cute/0t_mma_atom.md)



DeepGEMM is a library designed for clean and efficient FP8 General Matrix Multiplications (GEMMs) with fine-grained scaling, as proposed in [DeepSeek-V3](https://github.com/deepseek-ai/DeepSeek-V3). It supports both normal and Mix-of-Experts (MoE) grouped GEMMs. Written in CUDA, the library has no compilation need during installation, by compiling all kernels at runtime using a lightweight Just-In-Time (JIT) module.

Currently, DeepGEMM exclusively supports NVIDIA Hopper tensor cores. To address the imprecise FP8 tensor core accumulation, it employs CUDA-core two-level accumulation (promotion). While it leverages some concepts from [CUTLASS](https://github.com/nvidia/cutlass) and [CuTe](https://github.com/NVIDIA/cutlass/tree/main/include/cute), it avoids heavy reliance on their templates or algebras. Instead, the library is designed for simplicity, with only one core kernel function comprising around **~300 lines of code**. This makes it a clean and accessible resource for learning Hopper FP8 matrix multiplication and optimization techniques.


#### Repository Structure 

**/deep_gemm**


fp8_gemm.cuh
https://github.com/deepseek-ai/DeepGEMM/blob/main/deep_gemm/include/deep_gemm/fp8_gemm.cuh

mma_utils.cuh

scheduler.cuh 

tma_utils.cuh 

utils.cuh







/figures 

/tests

/third-party


