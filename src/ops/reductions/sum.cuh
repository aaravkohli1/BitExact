#pragma once

void sum_cuda(
    const float* input,
    float* output,
    int batch_size,
    int hidden_dim
);