#include <torch/extension.h>
#include "kernels/rms_norm.cuh"

torch::Tensor rms_norm(
    torch::Tensor input,
    torch::Tensor weight,
    double eps = 1e-6
) {
    TORCH_CHECK(input.is_cuda(), "input must be CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    TORCH_CHECK(input.dim() == 2, "input must be 2D");
    TORCH_CHECK(weight.dim() == 1, "weight must be 1D");
    TORCH_CHECK(input.size(1) == weight.size(0), "hidden_dim mismatch");
    
    auto output = torch::empty_like(input);
    
    int batch_size = input.size(0);
    int hidden_dim = input.size(1);
    
    rms_norm_cuda(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        hidden_dim,
        static_cast<float>(eps)
    );
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rms_norm", &rms_norm, "Batch-invariant RMS normalization",
          py::arg("input"), py::arg("weight"), py::arg("eps") = 1e-6);
}