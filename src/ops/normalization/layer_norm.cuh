void layer_norm_cuda(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int hidden_dim,
    float eps
);