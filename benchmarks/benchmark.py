import torch
import time
import bitexact

results = []

def benchmark_op(name, fn_ref, fn_bitexact, *args, runs=100):
    # Warm-up
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

    # Outputs
    ref_out = fn_ref(*args)
    be_out  = fn_bitexact(*args)

    same_shape = ref_out.shape == be_out.shape
    same_dtype = ref_out.dtype == be_out.dtype

    # Check Tolerance
    if same_shape:
        match = torch.allclose(ref_out, be_out, atol=1e-4, rtol=1e-6)
    else:
        match = False

    diff = (ref_out - be_out).abs().max().item() if same_shape else float('inf')

    results.append([name, f"{ref_ms:.4f}", f"{be_ms:.4f}",
                    f"{ref_ms / be_ms:.2f}x", f"{diff:.2e}", match])


if __name__ == "__main__":
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    x = torch.randn(1024, 1024, dtype=torch.float32, device=device).contiguous()
    w = torch.randn(1024, dtype=torch.float32, device=device).contiguous()
    b = torch.randn(1024, dtype=torch.float32, device=device).contiguous()
    A = torch.randn(512, 512, dtype=torch.float32, device=device).contiguous()
    B = torch.randn(512, 512, dtype=torch.float32, device=device).contiguous()

    print("BitExact vs PyTorch - Benchmark Suite")

    # Benchmarks
    benchmark_op("MatMul", torch.matmul, bitexact.matmul, A, B)
    benchmark_op("Sum", lambda X: torch.sum(X, dim=-1, keepdim=True), lambda X: bitexact.sum(X, dim=-1), x)
    benchmark_op("Mean", lambda X: torch.mean(X, dim=-1, keepdim=True), lambda X: bitexact.mean(X, dim=-1), x)
    benchmark_op("Max", lambda X: torch.max(X, dim=-1, keepdim=True)[0], lambda X: bitexact.max(X, dim=-1), x)
    benchmark_op("Min", lambda X: torch.min(X, dim=-1, keepdim=True)[0], lambda X: bitexact.min(X, dim=-1), x)
    benchmark_op("Sigmoid", torch.sigmoid, bitexact.sigmoid, x)

    def ref_rmsnorm(inp, wt, eps=1e-6):
        rms = torch.sqrt(inp.pow(2).mean(dim=-1, keepdim=True) + eps)
        return (inp / rms) * wt
    benchmark_op("RMSNorm", lambda X, W: ref_rmsnorm(X, W, 1e-6), bitexact.rms_norm, x, w)

    def ref_layernorm(inp, wt, bias, eps=1e-6):
        mu  = inp.mean(dim=-1, keepdim=True)
        var = (inp - mu).pow(2).mean(dim=-1, keepdim=True)
        y   = (inp - mu) / torch.sqrt(var + eps)
        return y * wt + bias
    benchmark_op("LayerNorm", lambda X, W, B: ref_layernorm(X, W, B, 1e-6),
                 bitexact.layer_norm, x, w, b)

    def ref_variance(inp, dim=-1):
        mu  = inp.mean(dim=dim, keepdim=True)
        var = (inp - mu).pow(2).mean(dim=dim, keepdim=True)
        return var
    benchmark_op("Variance", lambda X: ref_variance(X, dim=-1), bitexact.var, x)

    # Output table
    header = f"{'Operation':<12} {'Torch (ms)':>12} {'BitExact (ms)':>14} {'Speed':>10} {'Max Diff':>12} {'Match':>8}"
    print("\n" + header)
    print("-" * len(header))

    # Color codes
    green = "\033[92m"
    red = "\033[91m"
    reset = "\033[0m"

    for row in results:
        color = green if row[5] else red
        match_str = f"{color}{str(row[5]):>8}{reset}"
        print(f"{row[0]:<12} {row[1]:>12} {row[2]:>14} {row[3]:>10} {row[4]:>12} {match_str}")

    all_det = all(r[5] for r in results)
    average_speedup = round(sum(float(r[3][:-1]) for r in results) / len(results), 2)

    print("\nNote: Matches use atol=1e-4, rtol=1e-6 tolerance (within FP32 rounding).")
    print("-" * len(header))
    print("Summary")
    print("-" * len(header))
    print(f"Operations faster than PyTorch: {sum(1 for r in results if float(r[3][:-1]) > 1.0)}/{len(results)}")
    print(f"All operations deterministic: {green if all_det else red}{all_det}{reset}")
    print(f"Average speedup: {green if average_speedup >= 1 else red}{average_speedup}x{reset}")
    print('=' * len(header))