# BitExact API Reference

## Overview

Brief description of the library, goals, and determinism guarantees

## Available Operations

- [RMSNorm](#rmsnorm)
- [MatMul](#matmul)
- [Sum](#sum)
- [Mean](#mean)
- [Sigmoid](#sigmoid)

---

## RMSNorm

### Function

```python
rmsnorm(input: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor
```

### Description

Computes the Root Mean Square Normalization (RMSNorm) of a tensor in a deterministic batch-invariant manner. Implements fixed order reduction to ensure bit-identical results across runs, GPUs, and batch sizes.

The Root Mean Square Normalization (RMSNorm) of a vector is defined as:

![](<https://latex.codecogs.com/png.image?\dpi{150}\large\color{white}\mathrm{RMSNorm}(x)=\frac{x}{\sqrt{\frac{1}{n}\sum_{i=1}^{n}x_i^2+\epsilon}}>)

### Parameters

- Input: The input vector (Must be continous and of dtype float32 or float16)
- Weight: The weight vector
- Eps: Epsilon constant

### Returns

- Output: Normalized tensor of same shape and dtype

### Example

```python
import torch, bitexact
x = torch.randn(1024, device='cuda')
w = torch.ones(128, device='cuda')
y = bitexact.rms_norm(x, w, eps=1e-6)
```

### Notes

- Determinism is only verified on a handful of random seeds (10/29/2025)
- Does not allocate host-side memory

## MatMul

### Function

```python
matmmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor
```

### Description

Performs Matrix Multiplication (abbreviated as MatMul) on two tensors in a deterministic batch-invariant manner. Implements fixed order reduction to ensure bit-identical results across runs, GPUs, and batch sizes.

Mathematically, matrix multiplication is:

![](<https://math.vercel.app/?from=\color{white}\mathrm{MatMul}(A,B)=C,\quad%20C_{ij}=\sum_{k=1}^{n}A_{ik}B_{kj}>)

### Parameters

- A: The first matrix for multiplication
- B: The second matrix for multiplication

### Returns

- C: The result of matrix multiplication with matrices A and B

### Example

```python
import torch
import bitexact
torch.manual_seed(42)
a = torch.randn(4, 8, device='cuda')
b = torch.randn(8, 16, device='cuda')
c = bitexact.matmul(a, b)
```

### Notes

- Determinism is only verified on a handful of random seeds (10/29/2025)
- Does not allocate host-side memory

## Sum

### Function

```python
sum(input: torch.Tensor, dim: int = -1) -> tensor.Tensor
```

### Description

Sums a tensor in a deterministic batch-invariant manner. Implements fixed order reduction to ensure bit-identical results across runs, GPUs, and batch sizes.

### Parameters

- input: The input tensor
- dim: The dimension of reduction

### Returns

- A tensor with the sum.

### Example

```python
  import torch
  import bitexact
  x = torch.randn(32, 128, device='cuda')
  bit_sum = bitexact.sum(x, dim=-1)
```

### Notes

- Determinism is only verified on a handful of random seeds (10/29/2025)
- Does not allocate host-side memory

## Mean

### Function

```python
mean(input: torch.Tensor, dim: int = -1) -> torch.Tensor
```

### Description

Calculates the mean of a tensor in a deterministic batch-invariant manner. Implements fixed order reduction to ensure bit-identical results across runs, GPUs, and batch sizes.

### Parameters

- input: the input tensor
- dim: the dimension to reduce along

### Returns

- A tensor containing the mean

### Example

```python
import torch
import bitexact
x = torch.randn(32, 128, device='cuda')
bitexact.mean(x, dim=-1)
```

### Notes

- Determinism is only verified on a handful of random seeds (10/29/2025)
- Does not allocate host-side memory

## Sigmoid

### Function

```python
sigmoid(input: torch.Tensor) -> torch.Tensor
```

### Description

Calculates the sigmoid activation of a tensor in a deterministic manner. Sigmoid not perform reductions, uses IEEE expf conventions.

### Parameters

- input: the input tensor

### Returns

- a tensor containing the sigmoid activations

### Example

```python
import torch
import bitexact
x = torch.randn(4096, device="cuda", dtype=torch.float32)
bitexact.sigmoid(x)
```

### Notes

- Determinism is only verified on a handful of random seeds (10/29/2025)
- Does not allocate host-side memory
