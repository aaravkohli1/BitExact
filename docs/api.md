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
bitexact.rms_norm(input: torch.Tensor, weight: torch.Tensor, eps=1e-6) -> torch.Tensor:
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

```python
bitexact.layer_norm(input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float=1e-6) -> torch.Tensor
```

The LayerNorm Kernel computes the Layer Normalization over a small batch of inputs. This kernel implements the operation as described in the [Design Reference](#./design.md#layernorm).

Mathetmatically, given an input vector $x \in IR^n$, LayerNorm is defined as:

$$
LayerNorm(x) = \frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}\odot w+b
$$

where:

- $x_i$ is the $i^{th}$ element of the input vector
- $n$ is the dimensionality of the vector
- $\mu$ is the mean of the input
- $\sigma^2$ is the variance
- $\epsilon$ is a small constant for numerical stabillity
- $w$ and $b$ are learned per-element affine parameters

### Parameters

| Parameter | Type         | Function                                   |
| --------- | ------------ | ------------------------------------------ |
| input     | torch.Tensor | The input batch to apply RMSNorm to        |
| weight    | torch.Tensor | The weight tensor for the RMSNorm formula  |
| bias      | torch.Tensor | the bias tensor for element parameters     |
| eps       | float        | The constant (optional - defaults to 1e-6) |

### Returns

A tensor containing the LayerNorm applications.

### Example

```python
import torch
import bitexact as bxt

torch.manual_seed(42)
x = torch.randn(shape, dtype=torch.float32, device="cuda")
weight = torch.ones(x.shape[-1], device="cuda")
bias = torch.zeros(x.shape[-1], device="cuda")

 y = bxt.layer_norm(x, weight, bias)
```

## MatMul

```python
bitexact.matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor
```

The MatMul Kernel performs matrix multiplication between two input tensors. This kernel provides deterministic, bit-exact results across runs and devices, implementing the operation as described in the [Design Reference](#./design.md#matmul).

Mathetmatically, given matrices $A \in IR^{k \bigtimes n}$ and $B \in IR^{m \bigtimes n}$, the matrix product $C \in IR^{m \bigtimes n}$
is

$$
C_{ij}=\sum_{r=1}^{k}A_{ir}B_{rj}
$$

where:

- $A_{ir}$ is the element in the $i^{th}$ row and $r^{th}$ column of $A$
- $B_{rj}$ is the element in the $r^{th}$ row and $J^{th}$ column of $B$
- $B_{rj}$ is the resulting element in the $i^{th}$ row and $J^{th}$ column of $C$

### Parameters

| Parameter | Type         | Function      |
| --------- | ------------ | ------------- |
| A         | torch.Tensor | First Matrix  |
| B         | torch.Tensor | Second Matrix |

### Returns

A tensor, C.

### Example

```python
import torch
import bitexact as bxt

torch.manual_seed(42)
a = torch.randn(4, 8, device='cuda')
b = torch.randn(8, 16, device='cuda')
c = bxt.matmul(a, b)
```

## Sum

```python
bitexact.sum(input: torch.Tensor, dim: int = -1) -> torch.Tensor:
```

The Sum Kernel computes the reduction of all elements along the specified dimension(s) of the input tensor. This operation is implemented deterministically to guarantee bit-exact reproducibility across threads, warps, and devices, as detailed in the
[Design Reference](#./design.md#sum).

Mathetmatically, given matrices $A \in IR^{k \bigtimes n}$ and $B \in IR^{m \bigtimes n}$, the matrix product $C \in IR^{m \bigtimes n}$
is

$$
S=\sum_{i_1=1}^{d_1}\sum_{i_2=1}^{d_2}\dots\sum_{i_n=1}^{d_n}X_{i_1i_2\dotsi_n}
$$

where:

- $X_{i_1 i_2...i_n}$ denotes the element of the tensor.

### Parameters

| Parameter | Type         | Function     |
| --------- | ------------ | ------------ |
| input     | torch.Tensor | Input Tensor |
| dim       | int          | Dimension    |

### Returns

A tensor with the sum.

### Example

```python
import torch
import bitexact as bxt

x = torch.randn(32, 128, device='cuda')
bit_sum = bxt.sum(x, dim=-1)
```

## Mean

```python
bitexact.mean(input: torch.Tensor, dim: int = -1) -> torch.Tensor
```

The Mean Kernel computes the mean of all elements along the specified dimension(s) of the input tensor. This operation is implemented deterministically to guarantee bit-exact reproducibility across threads, warps, and devices, as detailed in the
[Design Reference](#./design.md#mean).

Mathetmatically, given an input tensor $X \in IR^{d_1 \bigtimes d_2 \ bigtimes ... \bigtimes d_n}$, the mean is

$$
M=\frac{1}{N}\sum_{i_1=1}^{d_1}\sum_{i_2=1}^{d_2}\dots\sum_{i_n=1}^{d_n}X_{i_1i_2\dotsi_n}
$$

where:

- $N$ is the total number of elements
- $X_{i_1 i_2 ... i_n}$ is the element of the input tensor

### Parameters

| Parameter | Type         | Function     |
| --------- | ------------ | ------------ |
| input     | torch.Tensor | Input Tensor |
| dim       | int          | Dimension    |

### Returns

A tensor with the mean.

### Example

```python
import torch
import bitexact as bxt

x = torch.randn(32, 128, device='cuda')
bit_mean = bxt.mean(x, dim=-1)
```

## Max

```python
max(input: torch.Tensor, dim: int = -1) -> torch.Tensor
```

The Max Kernel computes the maxmimum of all elements along the specified dimension(s) of the input tensor. This operation is implemented deterministically to guarantee bit-exact reproducibility across threads, warps, and devices, as detailed in the
[Design Reference](#./design.md#max).

Mathetmatically, given an input tensor $X \in IR^{d_1 \bigtimes d_2 \ bigtimes ... \bigtimes d_n}$, the max is

$$
M=\max_{1\le i_1\le d_1,\,1\le i_2\le d_2,\,\dots,\,1\le i_n\le d_n}X_{i_1i_2\dotsi_n}
$$

where:

- $X_{i_1 i_2 ... i_n}$ is the element of the input tensor

### Parameters

| Parameter | Type         | Function     |
| --------- | ------------ | ------------ |
| input     | torch.Tensor | Input Tensor |
| dim       | int          | Dimension    |

### Returns

A tensor with the max.

### Example

```python
import torch
import bitexact as bxt

x = torch.randn(32, 128, device='cuda')
bit_max = bxt.max(x, dim=-1)
```

## Min

```python
min(input: torch.Tensor, dim: int = -1) -> torch.Tensor
```

The Max Kernel computes the minimum of all elements along the specified dimension(s) of the input tensor. This operation is implemented deterministically to guarantee bit-exact reproducibility across threads, warps, and devices, as detailed in the
[Design Reference](#./design.md#min).

Mathetmatically, given an input tensor $X \in IR^{d_1 \bigtimes d_2 \ bigtimes ... \bigtimes d_n}$, the min is

$$
m=\min_{1\le i_1\le d_1,\,1\le i_2\le d_2,\,\dots,\,1\le i_n\le d_n}X_{i_1i_2\dotsi_n}
$$

where:

- $X_{i_1 i_2 ... i_n}$ is the element of the input tensor

### Parameters

| Parameter | Type         | Function     |
| --------- | ------------ | ------------ |
| input     | torch.Tensor | Input Tensor |
| dim       | int          | Dimension    |

### Returns

A tensor with the min.

### Example

```python
import torch
import bitexact as bxt

x = torch.randn(32, 128, device='cuda')
bit_min = bxt.min(x, dim=-1)
```

## Sigmoid

```python
bitexact.sigmoid(input: torch.Tensor) -> torch.Tensor
```

The Sigmoid Kernelthe elementwise logistic activation function to the input tensor.
This operation is implemented deterministically, ensuring bit-exact reproducibility across threads, warps, and devices, as described in the [Design Reference](#./design.md#sigmoid).

Mathetmatically, given an input tensor $X \in IR^{d_1 \bigtimes d_2 \ bigtimes ... \bigtimes d_n}$, the sigmoid function is

$$
\sigma(x)=\frac{1}{1+e^{-x}}
$$

where:

- $x$ is element of the input tensor (applied elementwise)

### Parameters

| Parameter | Type         | Function     |
| --------- | ------------ | ------------ |
| input     | torch.Tensor | Input Tensor |

### Returns

A tensor with the sigmoid activation.

### Example

```python
import torch
import bitexact as bxt

x = torch.linspace(-10, 10, steps=1000, dtype=torch.float32)
bit_sigmoid = bxt.sigmoid(x)
```

## Further Reference

BitExact is built on top of PyTorch, whose functions are not documented in this repository. For more information, visit the [official PyTorch website](https://pytorch.org)

To see what makes these kernels determinsistic and why its important, check out the [design reference](./design.md). To see performance metrics on the kernels, check out the [performance docs](./performance.md). Finally, for determinism guarantees check out the [testing docs](./testing.md).
