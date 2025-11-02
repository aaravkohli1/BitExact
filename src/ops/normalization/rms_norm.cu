#include "rms_norm.cuh"
#include "../../utils/cuda_utils.cuh"
#include "../../utils/reduction.cuh"
#include <cuda_runtime.h>

__global__ void rms_norm_kernel(
    const float* input,
    const float* weight,
    float* output,
    int batch_size,
    int hidden_dim,
    float eps
) {
    int batch_idx = blockIdx.x;
    if(batch_idx >= batch_size) return;

    const float* x = input + batch_idx * hidden_dim;
    float* y = output + batch_idx * hidden_dim;

    int vec_size = hidden_dim / 4;
    float sum = 0.0f;
    
    // Vectorize computation (squares)
    for(int i = threadIdx.x; i < vec_size; i += blockDim.x) {
        float4 vals = reinterpret_cast<const float4*>(x)[i];
        sum += vals.x * vals.x;
        sum += vals.y * vals.y;
        sum += vals.z * vals.z;
        sum += vals.w * vals.w;
    }

    // Compute remaining elements
    for(int i = vec_size * 4 + threadIdx.x; i < hidden_dim; i += blockDim.x){
        float val = x[i];
        sum += val * val;
    }

    // Warp synchronous reduction
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
            shared[0] = val;
        }
    }
    __syncthreads();

    float rms_val = rsqrtf(shared[0] / hidden_dim + eps);

    // Vectorize full computation
    for(int i = threadIdx.x; i < vec_size; i += blockDim.x) {
        float4 weight_vals = reinterpret_cast<const float4*>(weight)[i];
        float4 x_vals = reinterpret_cast<const float4*>(x)[i];
        float4 y_vals;
        
        y_vals.x = x_vals.x * rms_val * weight_vals.x;
        y_vals.y = x_vals.y * rms_val * weight_vals.y;
        y_vals.z = x_vals.z * rms_val * weight_vals.z;
        y_vals.w = x_vals.w * rms_val * weight_vals.w;

        reinterpret_cast<float4*>(y)[i] = y_vals;
    }

    // Compute remaining elements
    for(int i = vec_size * 4 + threadIdx.x; i < hidden_dim; i += blockDim.x) {
        y[i] = x[i] * rms_val * weight[i];
    }
}

void rms_norm_cuda(
    const float* input,
    const float* weight,
    float* output,
    int batch_size,
    int hidden_dim,
    float eps
) {
    int threads = 256;
    int blocks = batch_size;
    
    rms_norm_kernel<<<blocks, threads>>>(
        input, weight, output, batch_size, hidden_dim, eps
    );
    
    CUDA_KERNEL_CHECK();
}