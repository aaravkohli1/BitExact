#include "min.cuh"
#include "../../utils/cuda_utils.cuh"
#include "../../utils/reduction.cuh"
#include <cuda_runtime.h>

__global__ void min_kernel(
    const float* input,
    float* output,
    int batch_size,
    int hidden_dim
) {
    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;

    const float* x = input + batch_idx * hidden_dim;

    int vec_size = hidden_dim / 4;
    float min_val = INFINITY;
    
    for(int i = threadIdx.x; i < vec_size; i += blockDim.x) {
        float4 vals = reinterpret_cast<const float4*>(x)[i];
        min_val = fminf(vals.x, min_val);
        min_val = fminf(vals.y, min_val);
        min_val = fminf(vals.z, min_val);
        min_val = fminf(vals.w, min_val);
    }

    for(int i = vec_size * 4 + threadIdx.x; i < hidden_dim; i += blockDim.x){
        min_val = fminf(min_val, x[i]);
    }

    float min= warp_reduce_min(min_val);

    __shared__ float shared[8];  
    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;

    if(lane == 0) {
        shared[warp_id] = min;
    }

     __syncthreads();

    if (warp_id == 0) {
        float val = (lane < 8) ? shared[lane] : INFINITY;
        val = warp_reduce_min(val); 
        if (lane == 0) {
            shared[0] = val;
        }
    }
    __syncthreads();
    if(threadIdx.x == 0) {
        output[batch_idx] = shared[0];
    }
}

void min_cuda(
    const float* input,
    float* output,
    int batch_size,
    int hidden_dim
) {
    int threads = 256;
    int blocks = batch_size;
    
    min_kernel<<<blocks, threads>>>(input, output, batch_size, hidden_dim);
    CUDA_KERNEL_CHECK();
}