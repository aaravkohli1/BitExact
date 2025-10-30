#pragma once

void matmul_cuda(
    const float* A, // [M, K]
    const float* B, // [K, N]
    float* C,        // [M, N]
    int M,
    int K,
    int N
);