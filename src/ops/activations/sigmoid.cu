#include <cuda_runtime.h>
#include "../../utils/cuda_utils.cuh"
#include "sigmoid.cuh"

__global__ void sigmoid_kernel(
    const float* __restrict__ input, 
    float* __restrict__ output, 
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float x = input[idx];
    float val = expf(-x);
    float res = 1.0f / (1.0f + val);
    output[idx] = res;
}

void sigmoid_cuda(
    const float* input,
    float* output,
    int n
) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    sigmoid_kernel<<<blocks, threads>>>(input, output, n);
    CUDA_KERNEL_CHECK();
}