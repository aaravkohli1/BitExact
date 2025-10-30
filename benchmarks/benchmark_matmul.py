import torch
import time
import bitexact

def benchmark_matmul(M, K, N, num_iters=100):
    """Benchmark matmul for given shape"""
    A = torch.randn(M, K, device='cuda', dtype=torch.float32)
    B = torch.randn(K, N, device='cuda', dtype=torch.float32)
    
    # Warmup
    for _ in range(10):
        _ = bitexact.matmul(A, B)
    torch.cuda.synchronize()
    
    # Benchmark BitExact
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(num_iters):
        _ = bitexact.matmul(A, B)
    end.record()
    torch.cuda.synchronize()
    bit_time = start.elapsed_time(end) / num_iters
    
    # Benchmark PyTorch
    for _ in range(10):
        _ = torch.matmul(A, B)
    torch.cuda.synchronize()
    
    start.record()
    for _ in range(num_iters):
        _ = torch.matmul(A, B)
    end.record()
    torch.cuda.synchronize()
    torch_time = start.elapsed_time(end) / num_iters
    
    # Compute FLOPS
    flops = 2 * M * K * N  # 2 ops per multiply-add
    bit_gflops = (flops / 1e9) / (bit_time / 1000)
    torch_gflops = (flops / 1e9) / (torch_time / 1000)
    
    print(f"Shape [{M:4d}, {K:4d}] x [{K:4d}, {N:4d}]")
    print(f"  PyTorch:  {torch_time:7.3f} ms  ({torch_gflops:7.1f} GFLOPS)")
    print(f"  BitExact:     {bit_time:7.3f} ms  ({bit_gflops:7.1f} GFLOPS)")
    print(f"  Slowdown: {bit_time/torch_time:.2f}x")
    print()

if __name__ == "__main__":
    print("=" * 60)
    print("Matmul Benchmark")
    print("=" * 60)
    print()
    
    # Small matrices
    benchmark_matmul(64, 64, 64)
    benchmark_matmul(128, 128, 128)
    benchmark_matmul(256, 256, 256)
    
    # LLM-like shapes (batch, hidden_dim)
    benchmark_matmul(32, 4096, 4096)
    benchmark_matmul(64, 4096, 4096)