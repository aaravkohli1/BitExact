#pragma once
#include <cuda_runtime.h>

__device__ __forceinline__ float warp_reduce_sum(float val) {
     for(int offset = 16; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(0xffffffff, val, offset);   
    } return val;
}