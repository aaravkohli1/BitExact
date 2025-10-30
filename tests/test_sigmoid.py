import torch
import bitexact


class SigmoidFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor):
        ctx.save_for_backward(input)
        return bitexact.sigmoid(input)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (input,) = ctx.saved_tensors
        y = torch.sigmoid(input)
        grad_input = grad_output * y * (1 - y)
        return grad_input


def test_sigmoid_correctness_and_determinism():
    torch.manual_seed(42)

    x = torch.randn(4096, device="cuda", dtype=torch.float32)

    y1 = bitexact.sigmoid(x)
    y2 = bitexact.sigmoid(x)

    diff = torch.abs(y1 - y2).max().item()
    assert diff == 0.0, f"Nondeterministic sigmoid: max diff = {diff}"

    y_ref = torch.sigmoid(x)
    err = torch.abs(y1 - y_ref).max().item()
    assert err < 1e-6, f"❌ Incorrect output: max error = {err}"

    x = x.detach().clone().requires_grad_(True)
    y_custom = SigmoidFunction.apply(x)
    y_ref = torch.sigmoid(x)

    y_custom.sum().backward(retain_graph=True)
    grad_custom = x.grad.clone()
    x.grad.zero_()
    y_ref.sum().backward()
    grad_ref = x.grad

    grad_err = torch.abs(grad_custom - grad_ref).max().item()
    assert grad_err < 1e-6, f"Gradient mismatch: max error = {grad_err}"


if __name__ == "__main__":
    test_sigmoid_correctness_and_determinism()
