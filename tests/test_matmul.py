import torch
import bitexact

def test_correctness() -> None:
    """Test that matmul gives correct results"""
    torch.manual_seed(42)
    a = torch.randn(4, 8, device='cuda')
    b = torch.randn(8, 16, device='cuda')

    c = bitexact.matmul(a, b)
    reference = torch.matmul(a, b)
    assert torch.allclose(c, reference, atol=1e-4, rtol=1e-4)

def test_batch_invariance() -> None:
    """Test that matmul is batch invariant"""
    torch.manual_seed(42)
    a_big = torch.randn(32, 128, device='cuda')
    b = torch.randn(128, 64, device='cuda')

    c_big = bitexact.matmul(a_big, b)
    c_small = bitexact.matmul(a_big[:1], b)
    diff = (c_big[0] - c_small[0]).abs().max().item()

    assert diff == 0.0

if __name__ == "__main__":
    test_correctness()
    test_batch_invariance()