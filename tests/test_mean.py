import torch
import bitexact

def check_correctness() -> None:
    x = torch.randn(32, 128, device='cuda')
    bit_mean = bitexact.mean(x, dim=-1)
    torch_mean = torch.mean(x, dim=-1, keepdim=True)
    assert torch.allclose(bit_mean, torch_mean, rtol=1e-5, atol=1e-6)

if __name__ == "__main__":
    check_correctness()