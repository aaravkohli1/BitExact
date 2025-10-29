#include "matmul.cuh"
#include "../utils/cuda_utils.cuh"
#include "../utils/reduction.cuh"
#include <cuda_runtime.h>

__global__ void matmul_kernel(const float* A, const float* B, float* output) {
    int batch_idx = blockIdx.x;
    
}