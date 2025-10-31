import torch
import bitexact

def test_matmul() -> None:
    torch.manual_seed(42)

    a = torch.randn(4, 8, device='cuda')
    b = torch.randn(8, 16, device='cuda')
    c = bitexact.matmul(a, b)
    reference = torch.matmul(a, b)

    a_big = torch.randn(32, 128, device='cuda')
    b = torch.randn(128, 64, device='cuda')

    c_big = bitexact.matmul(a_big, b)
    c_small = bitexact.matmul(a_big[:1], b)
    diff = (c_big[0] - c_small[0]).abs().max().item()

    assert diff == 0.0
    assert torch.allclose(c, reference, atol=1e-4, rtol=1e-4)

def test_max() -> None:
    x = torch.randn(32, 128, device='cuda')
    bit_max = bitexact.max(x, dim=-1)
    torch_max = torch.max(x, dim=-1, keepdim=True)[0]
    assert (bit_max - torch_max).abs().max().item() == 0.0

def test_min() -> None:
    x = torch.randn(32, 128, device='cuda')

    bit_min = bitexact.min(x, dim=-1)
    torch_min = torch.min(x, dim=-1, keepdim=True)[0]

    diff = (bit_min - torch_min).abs().max().item()

    min_big = bitexact.min(x, dim=-1)
    min_small = bitexact.min(x[:1], dim=-1)

    invariance_diff = (min_big[0] - min_small[0]).abs().item()

    assert invariance_diff == 0.0
    assert torch.allclose(bit_min, torch_min, rtol=1e-5, atol=1e-6)

def test_mean() -> None:
    x = torch.randn(32, 128, device='cuda')
    bit_mean = bitexact.mean(x, dim=-1)
    torch_mean = torch.mean(x, dim=-1, keepdim=True)
    assert torch.allclose(bit_mean, torch_mean, rtol=1e-5, atol=1e-6)

def test_rmsnorm() -> None:
    x = torch.randn(4, 8, device='cuda')
    w = torch.ones(8, device='cuda')
    y = bitexact.rms_norm(x, w)

    x1 = torch.randn(8, 128, device='cuda')
    w1 = torch.ones(128, device='cuda')
    y1 = bitexact.rms_norm(x1, w1, eps=1e-6)
    mean_sq = (y1 ** 2).mean(dim=1)

    torch.manual_seed(42)
    x_big = torch.randn(32, 256, device='cuda')
    w2 = torch.ones(256, device='cuda')

    y_big = bitexact.rms_norm(x_big, w2)
    y_small = bitexact.rms_norm(x_big[:1], w2)
    diff = (y_big[0] - y_small[0]).abs().max().item()

    assert torch.allclose(mean_sq, torch.ones_like(mean_sq), atol=0.1)
    assert y.shape == x.shape, "Invalid Shapes"
    assert not torch.isnan(y).any(), "Invalid Output Tensor"
    assert diff == 0.0, "Not batch invariant :("

def test_layernorm() -> None:
    return

def test_sum() -> None:
    x = torch.randn(32, 128, device='cuda')
    bit_sum = bitexact.sum(x, dim=-1)
    torch_sum = torch.sum(x, dim=-1, keepdim=True)
    sum_big = bitexact.sum(x, dim=-1)
    sum_small = bitexact.sum(x[:1], dim=-1)
    diff = (sum_big[0] - sum_small[0]).abs().item()
    assert diff == 0.0, "Not batch invariant :("
    assert torch.allclose(bit_sum, torch_sum, atol=1e-5, rtol=0)

def run_suite(iterations=1000):
    for _ in range(iterations):
        if _ % 100 == 0:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        test_rmsnorm()
        test_matmul()
        test_mean()
        test_max()
        test_min()
        test_sum()

if __name__ == "__main__":
    run_suite()