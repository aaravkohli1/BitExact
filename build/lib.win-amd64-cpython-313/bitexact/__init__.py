from bitexact import _C
import torch

def rms_norm(input: torch.Tensor, weight: torch.Tensor, eps: float=1e-6) -> torch.Tensor:
    """Batch Invariant RMS Norm"""
    return _C.rms_norm(input, weight, eps)

def matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Batch Invariant Matrix Multiplication"""
    return _C.matmul(A, B)

def sum(input: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Batch Invariant Sum"""
    return _C.sum(input, dim)

def mean(input: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Batch-invariant mean reduction along a dimension."""
    return _C.mean(input, dim)

def sigmoid(input: torch.Tensor) -> torch.Tensor:
    """Batch Invariant Sigmoid Activation"""
    return _C.sigmoid(input)

def max(input: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Batch-invariant max reduction along a dimension."""
    return _C.max(input, dim)

def min(input: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Batch-invariant min reduction along a dimension."""
    return _C.min(input, dim)

def layer_norm(input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float=1e-6) -> torch.Tensor:
    """Batch invariant layer normalization"""
    return _C.layer_norm(input, weight, bias, eps)

def var(input: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Bath Invariant Variance"""
    return _C.var(input, dim)