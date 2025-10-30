from bitexact import _C
import torch

def rms_norm(input: torch.tensor, weight: torch.tensor, eps=1e-6) -> torch.tensor:
    return _C.rms_norm(input, weight, eps)

def matmul(A: torch.tensor, B: torch.tensor) -> torch.tensor:
    return _C.matmul(A, B)