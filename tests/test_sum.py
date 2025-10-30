import torch
import bitexact

def check_correctness() -> None:
    x = torch.randn(32, 128, device='cuda')
    bit_sum = bitexact.sum(x, dim=-1)
    torch_sum = torch.sum(x, dim=-1, keepdim=True)
    assert torch.allclose(bit_sum, torch_sum, rtol=1e-5, atol=1e-6)

def check_batch_invariance() -> None:
    x = torch.randn(32, 128, device='cuda')
    sum_big = bitexact.sum(x, dim=-1)
    sum_small = bitexact.sum(x[:1], dim=-1)
    diff = (sum_big[0] - sum_small[0]).abs().item()
    assert diff == 0.0, "Not batch invariant :("

if __name__ == "__main__":
    check_correctness()
    check_batch_invariance()