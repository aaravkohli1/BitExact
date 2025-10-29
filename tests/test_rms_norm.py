import torch
import bitexact

def test_basic() -> None:
    """Test that RMS_NORM runs without crashing"""
    x = torch.randn(4, 8, device='cuda')
    w = torch.ones(8, device='cuda')
    y = bitexact.rms_norm(x, w)

    assert y.shape == x.shape, "Invalid Shapes"
    assert not torch.isnan(y).any(), "Invalid Output Tensor"

def test_normalization() -> None:
    """Test that output is correct"""
    x = torch.randn(8, 128, device='cuda')
    w = torch.ones(128, device='cuda')
    y = bitexact.rms_norm(x, w, eps=1e-6)

    mean_sq = (y ** 2).mean(dim=1)
    assert torch.allclose(mean_sq, torch.ones_like(mean_sq), atol=0.1)

def test_batch_invariance() -> None:
    """Test that the kernel is batch invariant"""
    torch.manual_seed(42)
    x_big = torch.randn(32, 256, device='cuda')
    w = torch.ones(256, device='cuda')

    y_big = bitexact.rms_norm(x_big, w)
    y_small = bitexact.rms_norm(x_big[:1], w)

    diff = (y_big[0] - y_small[0]).abs().max().item()
    assert diff == 0.0, "Not batch invariant :("

if __name__ == "__main__":
    test_basic()
    test_normalization()
    test_batch_invariance()
