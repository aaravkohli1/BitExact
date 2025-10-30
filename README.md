# BitExact

_Deterministic CUDA Kernels for Reproducible Deep Learning_

BitExact is a research-driven CUDA library providing bit-exact deterministic GPU tensor operations.
It ensures identical floating-point results across runs, batches, and devices, removing nondeterminism from key deep-learning computations.

--

# Quick Links

- [📘API Reference](./docs/api.md)
- [✏️Design Reference](./docs/design.md)
- [💨Performance Reference](./docs/performance.md)
- [🧪Testing](./docs/testing.md)
- [💡Contributing](#contributing)

--

# Current Features

| Category       | Kernel Operation      | Reference                        |
| -------------- | --------------------- | -------------------------------- |
| Linear Algebra | Matrix Multiplication | [MatMul](./docs/api.md#matmul)   |
| Normalization  | RMS Normalization     | [RmsNorm](./docs/api.md#rmsnorm) |
| Reductions     | Sum                   | [Sum](./docs/api.md#sum)         |
| Reductions     | Mean                  | [Mean](./docs/api.md#mean)       |
| Activations    | Sigmoid               | [Sigmoid](./docs/api.md#sigmoid) |

> _More Determinsitic Kernels Coming Soon_

# Installation

## Prerequisites

- Python $\geq 3.9$
- CUDA $\geq 12.0$
- PyTorch $\geq2.1$
- A C++ Compiler (MSVC 2022 / gcc $\geq 9$)

## From Source

```bash
git clone https://github.com/yourusername/BitExact.git
cd BitExact
pip install . --no-build-isolation
```

## PyPI

```bash
pip install bitexact
```

# Performance at a Glance

| Operation             | Throughput (vs PyTorch) | Notes                            |
| --------------------- | ----------------------- | -------------------------------- |
| Matrix Multiplication | 7.5x faster on average  | [MatMul](./docs/api.md#matmul)   |
| RMS Normalziation     | tbd                     | [RmsNorm](./docs/api.md#rmsnorm) |
| Sum                   |                         | [Sum](./docs/api.md#sum)         |
| Mean                  |                         | [Mean](./docs/api.md#mean)       |
| Sigmoid               |                         | [Sigmoid](./docs/api.md#sigmoid) |

> _(Benchmarked on NVIDIA GeForce RTX 4060 Ti, PyTorch 2.6.0, CUDA 12.5)_

# Contributions

Contributions are welcome!

Please ensure new kernels:

1. Pass Deterministic equality tests across 3 or more runs.
2. Use Warp-synchronous, non-atomic reduction patterns.
3. Includes both .cu and .cuh files and a corresponding test.
