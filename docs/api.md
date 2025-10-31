# BitExact API Reference

## Overview

Brief description of the library, goals, and determinism guarantees

## Available Operations

- [RMSNorm](#rmsnorm)
- [LayerNorm](#layernorm)
- [MatMul](#matmul)
- [Sum](#sum)
- [Mean](#mean)
- [Max](#max)
- [Min](#min)
- [Sigmoid](#sigmoid)

## RMSNorm

```python
bitexact.rmsnorm(input: torch.Tensor, weight: torch.Tensor, eps=1e-6) -> torch.Tensor:
```

The RMSNorm Kernel computes the Root Mean Square Layer Normalization over a small batch of inputs. This kernel implements the operation as described in the [Design Reference](#./design.md#rmsnorm).

Mathetmatically, given an input vector $x \in IR^n$, RMSNorm is defined as:

$$
RMSNorm(x) = \frac{x}{\sqrt{\frac{1}{n} \sum_{i=1}^{n} x_i^{2} + \epsilon}} \odot w
$$

where:

- $x_i$ is the $i^{th}$ element of the input vector
- $n$ is the dimensionality of the vector
- $\epsilon$ is a small constant added for numerical stability
- $w$ is a learned scaling parameter (the weight)

### Parameters

| Parameter | Type         | Function                                   |
| --------- | ------------ | ------------------------------------------ |
| input     | torch.Tensor | The input batch to apply RMSNorm to        |
| weight    | torch.Tensor | The weight tensor for the RMSNorm formula  |
| eps       | float        | The constant (optional - defaults to 1e-6) |

### Returns

A tensor containing the RMSNorm applications.

### Example

```python
import torch
import bitexact as bxt

 x = torch.randn(4, 8, device='cuda')
 w = torch.ones(8, device='cuda')

 y = bxt.rms_norm(x, w)
```

## LayerNorm

## MatMul

## Sum

## Mean

## Max

## Min

## Sigmoid
