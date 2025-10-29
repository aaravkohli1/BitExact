#pragma once

void rms_norm_cuda(
    const float* input,
    const float* weight,
    float* output,
    int batch_size,
    int hidden_dim,
    float eps
);