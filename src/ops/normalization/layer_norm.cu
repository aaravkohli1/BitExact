#include "layer_norm.cuh"
#include "../reductions/mean.cuh"
#include "../reductions/var.cuh"
#include "../../utils/cuda_utils.cuh"
#include <cuda_runtime.h>

__global__ void layer_norm_apply_kernel(
    const float* input,
    const float* means,
    const float* vars,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int hidden_dim,
    float eps
) {
    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;
    
    const float* x = input + batch_idx * hidden_dim;
    float* y = output + batch_idx * hidden_dim;
    
    float mean = means[batch_idx];
    float var = vars[batch_idx];
    float inv_std = rsqrtf(var + eps);
    
    int vec_size = hidden_dim / 4;
    
    for(int i = threadIdx.x; i < vec_size; i += blockDim.x) {
         float4 x_vals = reinterpret_cast<const float4*>(x)[i];
         float4 w_vals = reinterpret_cast<const float4*>(weight)[i];
         float4 b_vals = reinterpret_cast<const float4*>(bias)[i];
        
         float4 y_vals;
         y_vals.x = (x_vals.x - mean) * inv_std * w_vals.x + b_vals.x;
         y_vals.y = (x_vals.y - mean) * inv_std * w_vals.y + b_vals.y;
         y_vals.z = (x_vals.z - mean) * inv_std * w_vals.z + b_vals.z;
         y_vals.w = (x_vals.w - mean) * inv_std * w_vals.w + b_vals.w;

         reinterpret_cast<float4*>(y)[i] = y_vals;
    }

    for(int i = vec_size * 4 + threadIdx.x; i < hidden_dim; i += blockDim.x){
        y[i] = (x[i] - mean) * inv_std * weight[i] + bias[i];
    }
}

void layer_norm_cuda(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int hidden_dim,
    float eps
) {
    float *temp_means, *temp_vars;
    cudaMalloc(&temp_means, batch_size * sizeof(float));
    cudaMalloc(&temp_vars, batch_size * sizeof(float));
    
    mean_cuda(input, temp_means, batch_size, hidden_dim);
    var_cuda(input, temp_vars, batch_size, hidden_dim);
    
    int threads = 256;
    int blocks = batch_size;
    layer_norm_apply_kernel<<<blocks, threads>>>(
        input, temp_means, temp_vars, weight, bias, output,
        batch_size, hidden_dim, eps
    );
    CUDA_KERNEL_CHECK();
    
    cudaFree(temp_means);
    cudaFree(temp_vars);
}