import torch
import bitexact

def test_correctness():
    x = torch.randn(32, 128, device='cuda')
    bit_max = bitexact.max(x, dim=-1)
    torch_max = torch.max(x, dim=-1, keepdim=True)[0]
    assert (bit_max - torch_max).abs().max().item() == 0.0

if __name__ == "__main__":
    test_correctness()