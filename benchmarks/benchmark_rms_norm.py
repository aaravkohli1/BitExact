import torch
import time
import bitexact

def benchmark_kernel(func, x, weight, num_warmup=10, num_iters=100):
    """Benchmark a function with CUDA events for accurate timing"""
    # Warmup
    for _ in range(num_warmup):
        _ = func(x, weight)
    
    torch.cuda.synchronize()
    
    # Benchmark
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(num_iters):
        _ = func(x, weight)
    end.record()
    
    torch.cuda.synchronize()
    
    elapsed_ms = start.elapsed_time(end)
    avg_ms = elapsed_ms / num_iters
    
    return avg_ms

def pytorch_rms_norm(x, weight, eps=1e-6):
    """PyTorch reference implementation"""
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x_normalized = x * torch.rsqrt(variance + eps)
    return x_normalized * weight

def compute_bandwidth(batch_size, hidden_dim, time_ms):
    """Compute effective bandwidth in GB/s"""
    # Reads: input (B*D*4 bytes), weight (D*4 bytes)
    # Writes: output (B*D*4 bytes)
    bytes_accessed = (2 * batch_size * hidden_dim + hidden_dim) * 4
    bytes_gb = bytes_accessed / 1e9
    time_s = time_ms / 1000
    bandwidth = bytes_gb / time_s
    return bandwidth

def run_benchmark(batch_size, hidden_dim):
    """Run benchmark for a specific shape"""
    x = torch.randn(batch_size, hidden_dim, device='cuda', dtype=torch.float32)
    weight = torch.ones(hidden_dim, device='cuda', dtype=torch.float32)
    
    # Benchmark PyTorch
    torch_time = benchmark_kernel(pytorch_rms_norm, x, weight)
    torch_bw = compute_bandwidth(batch_size, hidden_dim, torch_time)
    
    # Benchmark our implementation
    our_time = benchmark_kernel(bitexact.rms_norm, x, weight)
    our_bw = compute_bandwidth(batch_size, hidden_dim, our_time)
    
    speedup = torch_time / our_time
    
    print(f"Shape: [{batch_size:4d}, {hidden_dim:4d}]")
    print(f"  PyTorch:  {torch_time:.4f} ms  ({torch_bw:.2f} GB/s)")
    print(f"  Ours:     {our_time:.4f} ms  ({our_bw:.2f} GB/s)")
    print(f"  Speedup:  {speedup:.2f}x")
    print()
    
    return {
        'batch_size': batch_size,
        'hidden_dim': hidden_dim,
        'torch_time': torch_time,
        'our_time': our_time,
        'speedup': speedup,
        'torch_bw': torch_bw,
        'our_bw': our_bw
    }

if __name__ == "__main__":
    print("=" * 60)
    print("RMSNorm Benchmark")
    print("=" * 60)
    print()
    
    # Test various shapes
    configs = [
        # Small batch
        (1, 4096),
        (4, 4096),
        (8, 4096),
        
        # Medium batch
        (16, 4096),
        (32, 4096),
        (64, 4096),
        
        # Large batch
        (128, 4096),
        (256, 4096),
        
        # Different hidden dims
        (32, 2048),
        (32, 8192),
        (32, 16384),
    ]
    
    results = []
    for batch_size, hidden_dim in configs:
        result = run_benchmark(batch_size, hidden_dim)
        results.append(result)
    
    # Summary
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    avg_speedup = sum(r['speedup'] for r in results) / len(results)
    print(f"Average speedup: {avg_speedup:.2f}x")
    print(f"Best speedup: {max(r['speedup'] for r in results):.2f}x")
    print(f"Worst speedup: {min(r['speedup'] for r in results):.2f}x")