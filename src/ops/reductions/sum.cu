#include "sum.cuh"
#include "../../utils/cuda_utils.cuh"
#include "../../utils/reduction.cuh"
#include <cuda_runtime.h>

__global__ void sum_kernel(
    const float* input,
    float* output,
    int batch_size,
    int hidden_dim
) {
    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;

    const float* x = input + batch_idx * hidden_dim;

    int vec_size = hidden_dim / 4;
    float sum = 0.0f;
    
    for(int i = threadIdx.x; i < vec_size; i += blockDim.x) {
        float4 vals = reinterpret_cast<const float4*>(x)[i];
        sum += vals.x;
        sum += vals.y;
        sum += vals.z;
        sum += vals.w;
    }

    for(int i = vec_size * 4 + threadIdx.x; i < hidden_dim; i += blockDim.x){
        sum += x[i];
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
            shared[0] = val;
        }
    }
    __syncthreads();
    if(threadIdx.x == 0) {
        output[batch_idx] = shared[0];
    }
}

void sum_cuda(
    const float* input,
    float* output,
    int batch_size,
    int hidden_dim
) {
    int threads = 256;
    int blocks = batch_size;

    sum_kernel<<<blocks, threads>>>(input, output, batch_size, hidden_dim);
    CUDA_KERNEL_CHECK();
}