import torch
import bitexact

def test_correctness():
    x = torch.randn(32, 128, device='cuda')
    bit_min = bitexact.min(x, dim=-1)
    torch_min = torch.min(x, dim=-1, keepdim=True)[0]
    diff = (bit_min - torch_min).abs().max().item()
    assert torch.allclose(bit_min, torch_min, rtol=1e-5, atol=1e-6)

def test_invariance():
    x = torch.randn(32, 128, device='cuda')
    min_big = bitexact.min(x, dim=-1)
    min_small = bitexact.min(x[:1], dim=-1)
    invariance_diff = (min_big[0] - min_small[0]).abs().item()
    assert invariance_diff == 0.0