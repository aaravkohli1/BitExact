#pragma once
#include <cuda_runtime.h>

__device__ __forceinline__ float warp_reduce_sum(float val) {
     for(int offset = 16; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(0xffffffff, val, offset);   
    } return val;
}

__device__ __forceinline__ float warp_reduce_max(float val) {
    for(int offset = 16; offset > 0; offset >>= 1) {
        float other = __shfl_down_sync(0xffffffff, val, offset);
        val = fmaxf(val, other); 
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_min(float val) {
    for(int offset = 16; offset > 0; offset >>= 1) {
        float other = __shfl_down_sync(0xffffffff, val, offset);
        val = fminf(val, other); 
    }
    return val;
}