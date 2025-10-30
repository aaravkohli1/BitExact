# BitExact API Reference

## Overview

Brief description of the library, goals, and determinism guarantees

## Available Opeerations

    - [RMSNorm](#rmsnorm)
    - [MatMul](#matmul)
    - [Attention](#attention)

---

## RMSNorm

### Function

```python
rmsnorm(input: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor
```

### Description

Computes the Root Mean Square Normalization (RMSNorm) of a tensor in a deterministic batch-invariant manner. Implements fixed order reduction to ensure bit-identical results across runs, GPUs, and batch sizes.

The Root Mean Square Normalization (RMSNorm) of a vector is defined as:

![](<https://latex.codecogs.com/png.image?\dpi{120}\mathrm{RMSNorm}(x)=\frac{x}{\sqrt{\frac{1}{n}\sum_{i=1}^{n}x_i^2+\epsilon}}>)

### Parameters

    - Input: The input vector
      Must be continous and of dtype float32 or float16
    - Weight: The weight vector
    - Eps: Epsilon constant

### Returns

    - Output: Normalized tensor of same shape and dtype

### Example

    ```import torch, batchinv
    x = torch.randn(1024, device='cuda')
    w = torch.ones(128, device='cuda')
    y = batchinv.rms_norm(x, w, eps=1e-6)
    ```

### Notes

    - Determinism is only verified on a handful of random seeds (10/29/2025)
    - Does not allocate host-side memory

## MatMul

## Attention
