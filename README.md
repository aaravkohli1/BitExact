# BitExact

_Deterministic CUDA Kernels for Reproducible Deep Learning_

[![PyPI](https://img.shields.io/pypi/v/bitexact.svg)](https://pypi.org/project/bitexact/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE.md)
[![CUDA](https://img.shields.io/badge/CUDA-12.0%2B-green.svg)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-red.svg)]()

BitExact is a research-driven CUDA library providing bit-exact deterministic GPU tensor operations.
It ensures identical floating-point results across runs, batches, and devices, removing nondeterminism from key deep-learning computations.

The library is designed to be plug and play with PyTorch. This means it can serve as a drop-in replacement for selected PyTorch tensor operations while guaranteeing bit-level reproducibility.

BitExact is particularly suited for:

- Model reproducibility research - verifying training consistency across runs
- Numerical analysis and benchmarking - comparing model outputs with precision guarantees
- Deployment pipelines where deterministic inference is required for compliance or scientific validation

# Quick Links

- [Quick Start 🚀](#quick-start-example)
- [API Reference 📘](./docs/api.md)
- [Design Reference ✏️](./docs/design.md)
- [Performance Reference 💨](#performance-at-a-glance)
- [Testing 🧪](#testing)
- [Project Structure 🏛️](#project-structure)
- [Contributing 💡](#contributions)
- [Project Status ✅](#project-status)
- [Acknowledgements 🔍](#acknowledgements)

# Quick Start Example

```python
import torch, bitexact

x = torch.randn(4, 4, device="cuda")
w = torch.ones(4, device="cuda")

y = bitexact.rms_norm(x, w)
print(y)
```

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

> _More Determinsitic Kernels May Be Coming Soon_

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

| Operation             | Throughput (vs PyTorch) | Notes                                                                                          |
| --------------------- | ----------------------- | ---------------------------------------------------------------------------------------------- |
| Matrix Multiplication | 0.47x                   | Slower than cuBLAS; PyTorch’s highly tuned GEMM outperforms deterministic reduction.           |
| RMS Normalization     | 5.09x                   | Fused mean, sqrt, and scaling operations reduce kernel launches and memory access.             |
| Layer Normalization   | 1.66x                   | Fused single-kernel variance reduces global memory passes and improves speed on small tensors. |
| Sum                   | 1.98x                   | Optimized shared-memory reduction with fixed traversal order for determinism.                  |
| Mean                  | 1.69x                   | Builds on the Sum kernel with deterministic normalization by element count.                    |
| Max                   | 1.75x                   | Deterministic warp-level reduction; avoids divergent branching used in PyTorch.                |
| Min                   | 1.98x                   | Similar to Max; uses unified deterministic traversal for all elements.                         |
| Variance              | 1.35x                   | Uses fused E[x²] - (E[x])² formulation with deterministic accumulation.                        |
| Sigmoid               | 0.92x                   | Identical arithmetic to PyTorch; near-equal performance and perfect bit equivalence.           |
| **Average**           | **1.88x**               | Tests performed on small-scale tensors; PyTorch is optimized for large batch sizes.            |

> _(Benchmarked on NVIDIA GeForce RTX 4060 Ti, PyTorch 2.6.0, CUDA 12.5)_

## Interpretation of Results

BitExact’s performance advantage comes primarily from kernel fusion and deterministic reduction order, which minimize synchronization and memory traffic. However, PyTorch’s fused kernels outperform in large-batch GEMM and high-throughput workloads. These results emphasize that BitExact prioritizes determinism and reproducibility over raw FLOPS.

## Local Benchmarks

To see how BitExact benchmarks on your machine, run:

```bash
python benchmarks/benchmark.py
```

Example Output

```bash
BitExact vs PyTorch - Benchmark Suite

Operation      Torch (ms)  BitExact (ms)      Speed     Max Diff    Match
-------------------------------------------------------------------------
MatMul             0.0336         0.0692      0.48x     1.07e-04     True
Sum                0.0086         0.0117      0.73x     1.14e-05     True
Mean               0.0083         0.0079      1.05x     1.12e-08     True
Max                0.0087         0.0117      0.74x     0.00e+00     True
Min                0.0097         0.0080      1.21x     0.00e+00     True
Sigmoid            0.0074         0.0073      1.01x     0.00e+00     True
RMSNorm            0.0430         0.0084      5.12x     1.91e-06     True
LayerNorm          0.0881         0.0547      1.61x     1.91e-06     True
Variance           0.0311         0.0266      1.17x     2.38e-07     True

Note: Matches use atol=1e-4, rtol=1e-6 tolerance (within FP32 rounding).
-------------------------------------------------------------------------
Summary
-------------------------------------------------------------------------
Operations faster than PyTorch: 6/9
All operations deterministic: True
Average speedup: 1.46x
=========================================================================
```

# Testing

BitExact includes deterministic equality tests for all kernels.

To run the test suite, ensure you have PyTest installed. To install PyTest, run:

```bash
pip install -U pytest
```

Then you can run the test suite with:

```bash
pytest tests/
```

**Recommended Flags**

- `-v` - Verbose flag (shows results of each individual test)
- `-s` - Don’t capture output (allows setup logs from [conftest.py](./tests/conftest.py))

Example:

```bash
pytest tests/ -v -s
```

Because many tests utilize randomized tensors, running the suite multiple times can help verify reproducibility and numerical stability. You can run the tests any number of times, the examples below simply use 3 as a placeholder.

**Linux**

```bash
for i in {1..3}; do pytest -v; done
```

**Windows**

```powershell
for ($i = 1; $i -le 3; $i++) { pytest -v }
```

**Troubleshooting**

- CUDA OOM: close other GPU workloads, then re-run. Cache is auto-cleared; if needed, re-run with `-s` to confirm setup logs.
- No GPU: tests require a CUDA-capable device; CPU fallbacks are not provided.

> _All tests verify bit-exact equivalence to PyTorch’s reference implementations and ensure reproducibility across multiple runs and devices._

## Deterministic Inference

The [`examples/deterministic_inference.py`](./examples/deterministic_inference.py) script demonstrates a small neural network using BitExact kernels (`matmul`, `rms_norm`, and `sigmoid`). Running the example verifies that the network’s outputs are **bit-for-bit identical** across multiple runs, confirming complete GPU determinism.

Run the file with:

```bash
python examples/deterministic_inference.py
```

# Project Structure

```text

bitexact/
├── bitexact/                         # Python bindings and high-level API
│   └── __init__.py
│
├── benchmarks/                       # Benchmarking suite for performance comparison
│   ├── benchmark.py
│   └── utils.py
│
├── docs/                             # Technical documentation
│   ├── api.md
│   └── design.md
│
├── examples/                         # Minimal runnable examples
│   ├── basic_usage.py                # Simple demonstration of deterministic ops
│   └── deterministic_inference.py    # Reproducible model inference pipeline
│
├── src/                              # Core CUDA/C++ source
│   ├── bindings.cpp                  # PyTorch extension bindings (exposes kernels to Python)
│   │
│   └── ops/                          # Kernel implementations
│       ├── matmul/                   # Matrix multiplication kernels
│       │   ├── matmul.cu
│       │   └── matmul.cuh
│       │
│       ├── reductions/               # Deterministic reduction kernels
│       │   ├── sum.cu
│       │   ├── sum.cuh
│       │   ├── mean.cu
│       │   ├── mean.cuh
│       │   ├── max.cu
│       │   ├── max.cuh
│       │   ├── min.cu
│       │   ├── min.cuh
│       │   ├── var.cu
│       │   └── var.cuh
│       │
│       ├── normalization/            # Normalization kernels
│       │   ├── rms_norm.cu
│       │   ├── rms_norm.cuh
│       │   ├── layer_norm.cu
│       │   └── layer_norm.cuh
│       │
│       ├── activations/              # Activation kernels
│       │   ├── sigmoid.cu
│       │   └── sigmoid.cuh
│       │
│       └── utils/                    # Shared CUDA utilities
│           ├── cuda_utils.cuh        # Common device helpers (grid-stride loops, etc.)
│           ├── dtype_utils.cuh       # Type casting and precision utilities
│           └── reduction.cuh         # Shared reduction patterns for deterministic ops
│
├── tests/                            # Pytest suite
│   ├── conftest.py
│   └── test_determinism.py
│
├── LICENSE                           # License file
├── README.md                         # Project overview and documentation
└── setup.py                          # Build and installation script

```

# Contributions

Contributions are welcome! If you have an idea for a Kernel, feel free to implement it (the largest missing one is attention).

Please ensure new kernels:

1. Pass Deterministic equality tests (see [testing suite](./tests/test_determinism.py)).
2. Use Warp-synchronous, non-atomic reduction patterns.
3. Includes both .cu and .cuh files and a corresponding test.

# Project Status

This project was an experiment that followed a research article. I found it to be an interesting problem, so I spent a portion of my reading week making this library. I do find the problem of determinism to be really interesting so I will keep developing this library, but on no fixed schedule.

There are many ways the library could be expanded, outlined in the [design document](./docs/design.md). If you are interested, feel free to make a contribution.

### Acknowledgements

This project draws inspiration from research by [Thinking Machines Lab](https://thinkingmachines.ai/)
on deterministic GPU computation and reproducible deep learning.
Their exploration of bit-exact kernels and floating-point determinism informed the design philosophy of BitExact.

```

```
