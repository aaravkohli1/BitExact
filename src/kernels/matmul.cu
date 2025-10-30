#include "matmul.cuh"
#include "../utils/cuda_utils.cuh"
#include "../utils/reduction.cuh"
#include <cuda_runtime.h>

__global__ void matmul_kernel(
    const float* A, 
    const float* B, 
    float* C, 
    int M, 
    int K, 
    int N
) {
    int row = blockIdx.y;
    int col = blockIdx.x;
    
    if (row >= M || col >= N) return;
    
    float sum = 0.0f;
    for (int k = threadIdx.x; k < K; k += blockDim.x) {
        sum += A[row * K + k] * B[k * N + col];
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

    if (threadIdx.x == 0) {
        C[row * N + col] = shared[0];
    }
}

void matmul_cuda(
    const float* A,
    const float* B,
    float* C,
    int M, int K, int N
) {
    dim3 grid(N, M); 
    dim3 block(256);
    
    matmul_kernel<<<grid, block>>>(A, B, C, M, K, N);
    CUDA_KERNEL_CHECK();
}