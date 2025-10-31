import torch
import time
import bitexact


def benchmark_op(name, fn_ref, fn_bitexact, *args, runs=100):
    # Warm-up (stabilize clocks / JIT)
    for _ in range(10):
        fn_ref(*args)
        fn_bitexact(*args)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Reference timing
    t0 = time.perf_counter()
    for _ in range(runs):
        fn_ref(*args)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    ref_ms = (time.perf_counter() - t0) * 1000.0 / runs

    # BitExact timing
    t0 = time.perf_counter()
    for _ in range(runs):
        fn_bitexact(*args)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    be_ms = (time.perf_counter() - t0) * 1000.0 / runs

    # Outputs for comparison
    ref_out = fn_ref(*args)
    be_out  = fn_bitexact(*args)

    # Basic sanity
    same_shape = ref_out.shape == be_out.shape
    same_dtype = ref_out.dtype == be_out.dtype

    # Bit-Exact check
    match = torch.allclose(ref_out, be_out, atol=0, rtol=0) if same_shape else False
    diff = (ref_out - be_out).abs().max().item() if same_shape else float('inf')

    print(f"\n{name}")
    print("-" * len(name))
    print(f"  PyTorch:   {ref_ms:.4f} ms")
    print(f"  BitExact:  {be_ms:.4f} ms")
    print(f"  Speed:     {ref_ms / be_ms:.2f}x vs PyTorch")
    print(f"  Shapes:    ref={tuple(ref_out.shape)}  bitexact={tuple(be_out.shape)}  dtypes: ref={ref_out.dtype}, be={be_out.dtype}")
    print(f"  Bit-Exact: {match} (max diff = {diff:.3e})")

    # Heuristic hint for var mismatch
    if name.lower().startswith("variance") and not match and same_shape and diff > 1e-3:
        print("  Hint: Large diff suggests biased/unbiased or denominator mismatch. "
              "Ensure population variance: mean((x-mean)^2, dim).")


if __name__ == "__main__":
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Sample tensors (contiguous)
    x = torch.randn(1024, 1024, dtype=torch.float32, device=device).contiguous()
    w = torch.randn(1024, dtype=torch.float32, device=device).contiguous()
    b = torch.randn(1024, dtype=torch.float32, device=device).contiguous()
    A = torch.randn(512, 512, dtype=torch.float32, device=device).contiguous()
    B = torch.randn(512, 512, dtype=torch.float32, device=device).contiguous()

    print("BitExact Benchmark Suite\n==========================")

    # ---- MatMul (keep as-is) ----
    benchmark_op("MatMul", torch.matmul, bitexact.matmul, A, B)

    # ---- Sum / Mean / Max / Min / Sigmoid (keep) ----
    # Reductions with dim parameter
    benchmark_op("Sum", 
             lambda X: torch.sum(X, dim=-1, keepdim=True), 
             lambda X: bitexact.sum(X, dim=-1), x)

    benchmark_op("Mean", 
             lambda X: torch.mean(X, dim=-1, keepdim=True), 
             lambda X: bitexact.mean(X, dim=-1), x)

    benchmark_op("Max", 
             lambda X: torch.max(X, dim=-1, keepdim=True)[0],
             lambda X: bitexact.max(X, dim=-1), x)

    benchmark_op("Min", 
             lambda X: torch.min(X, dim=-1, keepdim=True)[0],
             lambda X: bitexact.min(X, dim=-1), x)

    benchmark_op("Sigmoid", torch.sigmoid, bitexact.sigmoid, x)

    # ---- RMSNorm: formula-aligned reference ----
    def ref_rmsnorm(inp, wt, eps=1e-6):
        rms = torch.sqrt(inp.pow(2).mean(dim=-1, keepdim=True) + eps)  # population-style
        return (inp / rms) * wt

    benchmark_op("RMSNorm", lambda X, W: ref_rmsnorm(X, W, 1e-6), bitexact.rms_norm, x, w)

    # ---- LayerNorm: formula-aligned reference (population variance, eps inside sqrt) ----
    def ref_layernorm(inp, wt, bias, eps=1e-6):
        mu  = inp.mean(dim=-1, keepdim=True)
        var = (inp - mu).pow(2).mean(dim=-1, keepdim=True)
        y   = (inp - mu) / torch.sqrt(var + eps)
        return y * wt + bias

    benchmark_op("LayerNorm", lambda X, W, B: ref_layernorm(X, W, B, 1e-6),
                 bitexact.layer_norm, x, w, b)

    # ---- Variance: explicit population variance reference ----
    def ref_variance(inp, dim=-1):
        mu  = inp.mean(dim=dim, keepdim=True)
        var = (inp - mu).pow(2).mean(dim=dim, keepdim=True) 
        return var

    benchmark_op("Variance", lambda X: ref_variance(X, dim=-1), bitexact.var, x)
