import os
import sys
import torch
import bitexact


# Fix Windows terminal output
os.system("")
sys.stdout.reconfigure(encoding="utf-8")

# Config
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32
EPS = 1e-6

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"

def forward_bitexact(x, W1, b1, W2, b2):
    h = bitexact.matmul(x, W1) + b1
    scale = torch.ones(h.shape[-1], device=DEVICE, dtype=DTYPE)
    h = bitexact.rms_norm(h, scale, EPS)
    y_lin = bitexact.matmul(h, W2) + b2
    y = bitexact.sigmoid(y_lin)
    return y


def make_fixed_weights(in_dim=64, hidden=32, out_dim=10):
    W1 = torch.linspace(-0.5, 0.5, steps=in_dim * hidden, device=DEVICE, dtype=DTYPE).reshape(in_dim, hidden).contiguous()
    b1 = torch.linspace(-0.1, 0.1, steps=hidden, device=DEVICE, dtype=DTYPE).contiguous()
    W2 = torch.linspace(-0.3, 0.3, steps=hidden * out_dim, device=DEVICE, dtype=DTYPE).reshape(hidden, out_dim).contiguous()
    b2 = torch.linspace(-0.05, 0.05, steps=out_dim, device=DEVICE, dtype=DTYPE).contiguous()
    return W1, b1, W2, b2


def make_input(batch=128, in_dim=64):
    base = torch.arange(batch * in_dim, device=DEVICE, dtype=DTYPE).reshape(batch, in_dim)
    x = (base % 97) / 97.0
    return x.contiguous()


def compare_runs(x, W1, b1, W2, b2, runs=3):
    outputs = [forward_bitexact(x, W1, b1, W2, b2) for _ in range(runs)]
    all_equal = True
    for i in range(1, runs):
        equal = torch.equal(outputs[0], outputs[i])
        color = GREEN if equal else RED
        print(f"{color}Run 0 vs Run {i}: bit-equal = {equal}{RESET}")
        if not equal:
            all_equal = False
            max_diff = (outputs[0] - outputs[i]).abs().max().item()
            print(f"  Max abs diff: {max_diff:.9e}")
    return all_equal


def main():
    if DEVICE != "cuda":
        print("WARNING: CUDA not available; BitExact kernels require CUDA. Exiting.")
        return

    in_dim, hidden, out_dim = 64, 32, 10
    x = make_input(batch=128, in_dim=in_dim)
    W1, b1, W2, b2 = make_fixed_weights(in_dim, hidden, out_dim)

    print("----------------------------------------")
    print("BitExact Deterministic Inference Demo")
    print("----------------------------------------\n")
    print(f"Device: {DEVICE}  dtype: {DTYPE}  input: {tuple(x.shape)}\n")

    ok = compare_runs(x, W1, b1, W2, b2, runs=3)

    if ok:
        print(f"\n{GREEN}RESULT: Bitwise Reproducible Across Runs ✅{RESET}")
    else:
        print(f"\n{RED}RESULT: Not Reproducible – See Diagnostics ❌{RESET}")


if __name__ == "__main__":
    main()
