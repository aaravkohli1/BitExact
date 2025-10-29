#pragma once
#include <cuda_runtime.h>
#include <stdio.h>

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,  \
                    cudaGetErrorString(err));                                 \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

#define CUDA_KERNEL_CHECK()                                                   \
    do {                                                                      \
        cudaError_t err = cudaGetLastError();                                 \
        if(err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA kernel launch error at %s:%d: %s\n",        \
                __FILE__, __LINE__, cudaGetErrorString(err));                 \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while(0)                                                                
