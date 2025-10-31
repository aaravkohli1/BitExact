# BitExact

_Deterministic CUDA Kernels for Reproducible Deep Learning_

BitExact is a research-driven CUDA library providing bit-exact deterministic GPU tensor operations.
It ensures identical floating-point results across runs, batches, and devices, removing nondeterminism from key deep-learning computations.

The library is designed to be plug and play with PyTorch. This means it can serve as a drop-in replacement for selected PyTorch tensor operations while guaranteeing bit-level reproducibility.

BitExact is particularly suited for:

- Model reproducibility research - verifying training consistency across runs
- Numerical analysis and benchmarking - comparing model outputs with precision guarantees
- Deployment pipelines where deterministic inference is required for compliance or scientific validation

# Quick Links

- [API Reference 📘](./docs/api.md)
- [Design Reference ✏️](./docs/design.md)
- [Performance Reference 💨](./docs/performance.md)
- [Testing 🧪](./docs/testing.md)
- [Contributing 💡](#contributing)

# Current Features

| Category       | Kernel Operation      | Reference                          |
| -------------- | --------------------- | ---------------------------------- |
| Linear Algebra | Matrix Multiplication | [MatMul](./docs/api.md#matmul)     |
| Normalization  | RMS Normalization     | [RmsNorm](./docs/api.md#rmsnorm)   |
| Normalization  | Layer Normalization   | [RmsNorm](./docs/api.md#layernorm) |
| Reductions     | Sum                   | [Sum](./docs/api.md#sum)           |
| Reductions     | Mean                  | [Mean](./docs/api.md#mean)         |
| Reductions     | Max                   | [Max](./docs/api.md#max)           |
| Reductions     | Min                   | [Min](./docs/api.md#min)           |
| Activations    | Sigmoid               | [Sigmoid](./docs/api.md#sigmoid)   |

> _More Determinsitic Kernels Coming Soon_

# Installation

## Prerequisites

- Python $\geq 3.9$
- CUDA $\geq 12.0$
- PyTorch $\geq2.1$
- A C++ Compiler (MSVC 2022 / gcc $\geq 9$)

## From Source

```bash
git clone https://github.com/aaravkohli1/BitExact.git
cd BitExact
pip install . --no-build-isolation
```

## PyPI

```bash
pip install bitexact
```

# Performance at a Glance

Note that the tensors, and PyTorch is optimized for large batch workloads.

| Operation             | Throughput (vs PyTorch) | Notes                                                                                 |
| --------------------- | ----------------------- | ------------------------------------------------------------------------------------- |
| Matrix Multiplication | 0.47x                   | Slower than cuBLAS; PyTorch’s highly tuned GEMM outperforms deterministic reduction.  |
| RMS Normalization     | 5.09x                   | Fused mean, sqrt, and scaling operations reduce kernel launches and memory access.    |
| Layer Normalization   | 1.66x                   | Single-pass population variance computation yields faster execution on small tensors. |
| Sum                   | 1.98x                   | Optimized shared-memory reduction with fixed traversal order for determinism.         |
| Mean                  | 1.69x                   | Builds on the Sum kernel with deterministic normalization by element count.           |
| Max                   | 1.75x                   | Deterministic warp-level reduction; avoids divergent branching used in PyTorch.       |
| Min                   | 1.98x                   | Similar to Max; uses unified deterministic traversal for all elements.                |
| Variance              | 1.35x                   | Uses fused E[x²] - (E[x])² formulation with deterministic accumulation.               |
| Sigmoid               | 0.92x                   | Identical arithmetic to PyTorch; near-equal performance and perfect bit equivalence.  |
| **Average**           | **1.88x**               | Tests performed on small-scale tensors; PyTorch is optimized for large batch sizes.   |

> _(Benchmarked on NVIDIA GeForce RTX 4060 Ti, PyTorch 2.6.0, CUDA 12.5)_

# Contributions

Contributions are welcome! If you have an idea for a Kernel, feel free to implement it (the largest missing one is attention).

Please ensure new kernels:

1. Pass Deterministic equality tests (see [Testing](./test_determinism.py)).
2. Use Warp-synchronous, non-atomic reduction patterns.
3. Includes both .cu and .cuh files and a corresponding test.
