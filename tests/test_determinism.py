import torch
import bitexact
import pytest

def test_matmul() -> None:
    """Test matrix multiplication correctness and determinism"""
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

@pytest.mark.parametrize("shape", [(32, 128), (16, 64), (8, 256)])
@pytest.mark.parametrize("seed", [42, 123, 456])
def test_max(shape, seed):
    """Test the max operation correctness and determinism"""
    torch.manual_seed(seed)
    x = torch.randn(shape, device='cuda')
    bit_max = bitexact.max(x, dim=-1)
    torch_max = torch.max(x, dim=-1, keepdim=True)[0]
    max_big = bitexact.max(x, dim=-1)
    max_small = bitexact.max(x[:1], dim=-1)
    invariance = (max_big[0] - max_small[0]).abs().item()
    assert invariance == 0.0
    assert (bit_max - torch_max).abs().max().item() == 0.0

@pytest.mark.parametrize("shape", [(32, 128), (16, 64), (8, 256)])
@pytest.mark.parametrize("seed", [42, 123, 456])
def test_min(shape, seed):
    """Test the min operation correctness and determinism"""
    torch.manual_seed(seed)
    x = torch.randn(shape, device='cuda')
    bit_min = bitexact.min(x, dim=-1)
    torch_min = torch.min(x, dim=-1, keepdim=True)[0]
    min_big = bitexact.min(x, dim=-1)
    min_small = bitexact.min(x[:1], dim=-1)
    invariance = (min_big[0] - min_small[0]).abs().item()
    assert invariance == 0.0
    assert torch.allclose(bit_min, torch_min, atol=1e-5, rtol=0)

@pytest.mark.parametrize("batch_size,hidden_dim", [(32, 128), (16, 64), (8, 256)])
def test_mean(batch_size, hidden_dim) -> None:
    """Test the mean operation correctness and determinism"""
    x = torch.randn(batch_size, hidden_dim, device='cuda')
    bit_mean = bitexact.mean(x, dim=-1)
    torch_mean = torch.mean(x, dim=-1, keepdim=True)
    assert torch.allclose(bit_mean, torch_mean, rtol=1e-5, atol=1e-6)

def test_rmsnorm() -> None:
    """Test the mean operation correctness and determinism"""
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
   

@pytest.mark.parametrize("shape", [(32, 128), (16, 64), (8, 256)])
@pytest.mark.parametrize("seed", [42, 123, 456, 789])
def test_sum(shape, seed):
    """Test the sum operation correctness and determinism"""
    torch.manual_seed(seed)
    x = torch.randn(shape, device='cuda')
    bit_sum = bitexact.sum(x, dim=-1)
    torch_sum = torch.sum(x, dim=-1, keepdim=True)
    sum_big = bitexact.sum(x, dim=-1)
    sum_small = bitexact.sum(x[:1], dim=-1)
    diff = (sum_big[0] - sum_small[0]).abs().item()
    assert diff == 0.0, "Not batch invariant :("
    assert torch.allclose(bit_sum, torch_sum, atol=1e-5, rtol=0)

@pytest.mark.parametrize("shape", [(32, 64), (4, 16, 128)])
@pytest.mark.parametrize("eps", [1e-5, 1e-6])
def test_layernorm_determinism(shape, eps):
    """Test the layernorm operation correctness and determinism"""
    torch.manual_seed(42)
    x = torch.randn(shape, dtype=torch.float32, device="cuda")

    ref = torch.nn.functional.layer_norm(
    x, normalized_shape=x.shape[-1:], eps=eps
    )

    weight = torch.ones(x.shape[-1], device="cuda")
    bias = torch.zeros(x.shape[-1], device="cuda")

    y1 = bitexact.layer_norm(x, weight, bias, eps=eps)
    y2 = bitexact.layer_norm(x, weight, bias, eps=eps)

    assert torch.allclose(y1, ref, atol=1e-5)
    assert torch.allclose(y1, y2, atol=0)


@pytest.mark.parametrize("shape", [(32, 128), (16, 64)])
@pytest.mark.parametrize("seed", [42, 123, 456])
def test_var(shape, seed):
    """Test the variance operation correctness and determinism"""
    torch.manual_seed(seed)
    x = torch.randn(shape, device='cuda')
    bit_var = bitexact.var(x, dim=-1)
    torch_var = x.var(dim=-1, keepdim=True, unbiased=False)
    assert torch.allclose(bit_var, torch_var, atol=1e-5)


@pytest.mark.parametrize("shape", [(32, 64), (16, 128)])
@pytest.mark.parametrize("dtype", [torch.float32])
def test_sigmoid(shape, dtype):
    """Test the sigmoid operation correctness and determinism"""
    x = torch.randn(shape, dtype=dtype, device='cuda')
    out = bitexact.sigmoid(x)
    ref = torch.sigmoid(x)
    assert torch.allclose(out, ref, atol=1e-3)

