from batchinv import _C
import torch

def rms_norm(input, weight, eps=1e-6) -> torch.Tensor:
    return _C.rms_norm(input, weight, eps)