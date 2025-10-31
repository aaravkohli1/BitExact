#include <cuda_runtime.h>
#include "../../utils/cuda_utils.cuh"
#include "sigmoid.cuh"

__global__ void sigmoid_kernel(
    const float* __restrict__ input, 
    float* __restrict__ output, 
    int n
) {
    int vec_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int vec_size = n / 4;
    
    if (vec_idx < vec_size) {
        float4 vals = reinterpret_cast<const float4*>(input)[vec_idx];
        
        float4 result;
        result.x = 1.0f / (1.0f + expf(-vals.x));
        result.y = 1.0f / (1.0f + expf(-vals.y));
        result.z = 1.0f / (1.0f + expf(-vals.z));
        result.w = 1.0f / (1.0f + expf(-vals.w));
        
        reinterpret_cast<float4*>(output)[vec_idx] = result;
    }
    
    int scalar_idx = vec_size * 4 + (blockIdx.x * blockDim.x + threadIdx.x);
    if (scalar_idx < n) {
        float x = input[scalar_idx];
        output[scalar_idx] = 1.0f / (1.0f + expf(-x));
    }
}

void sigmoid_cuda(
    const float* input,
    float* output,
    int n
) {
    int threads = 256;
    int vec_size = n / 4;
    int blocks_vec = (vec_size + threads - 1) / threads;
    int blocks_scalar = ((n % 4) + threads - 1) / threads;
    int blocks = max(blocks_vec, blocks_scalar);
    
    sigmoid_kernel<<<blocks, threads>>>(input, output, n);
    CUDA_KERNEL_CHECK();
}