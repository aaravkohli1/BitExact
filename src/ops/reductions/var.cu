#include "var.cuh"
#include "mean.cuh"
#include "../../utils/cuda_utils.cuh"
#include "../../utils/reduction.cuh"
#include <cuda_runtime.h>

__global__ void var_kernel(
    const float* input,
    const float* temp_means,
    float* output,
    int batch_size,
    int hidden_dim
) {
    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;
    
    const float* x = input + batch_idx * hidden_dim;
    float mean = temp_means[batch_idx];

    int vec_size = hidden_dim / 4;
    float sum = 0.0f;
    
    for(int i = threadIdx.x; i < vec_size; i += blockDim.x) {
        float4 vals = reinterpret_cast<const float4*>(x)[i];
        float diff_x = vals.x - mean;
        float diff_y = vals.y - mean;
        float diff_z = vals.z - mean;
        float diff_w = vals.w - mean;

        sum += diff_x * diff_x;
        sum += diff_y * diff_y;
        sum += diff_z * diff_z;
        sum += diff_w * diff_w;
    }

    for(int i = vec_size * 4 + threadIdx.x; i < hidden_dim; i += blockDim.x){
        float diff = x[i] - mean;
        sum += diff * diff;
    }

    float warp_sum = warp_reduce_sum(sum);
    __shared__ float shared[8];  
    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;

    if (lane == 0) {
        shared[warp_id] = warp_sum;
    }
    __syncthreads();

    if (warp_id == 0) {
        float val = (lane < 8) ? shared[lane] : 0.0f;
        val = warp_reduce_sum(val);
        if (lane == 0) {
            shared[0] = val / hidden_dim;
        }
    }
    __syncthreads();
    
    if(threadIdx.x == 0) {
        output[batch_idx] = shared[0];
    }
}

void var_cuda(
    const float* input,
    float* output,
    int batch_size,
    int hidden_dim
) {
    float* temp_means;
    cudaMalloc(&temp_means, batch_size * sizeof(float));
    mean_cuda(input, temp_means, batch_size, hidden_dim);
    
    int threads = 256;
    int blocks = batch_size;
    var_kernel<<<blocks, threads>>>(input, temp_means, output, batch_size, hidden_dim);
    CUDA_KERNEL_CHECK();
    cudaFree(temp_means);
}