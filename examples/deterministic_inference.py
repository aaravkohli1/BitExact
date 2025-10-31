import os
import torch
import bitexact

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32
EPS = 1e-6

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# -----------------------------------------------------------------------------
# Deterministic MLP using BitExact ops only
#   y = sigmoid( ( RMSNorm( x @ W1 + b1 ) ) @ W2 + b2 )
# -----------------------------------------------------------------------------
def forward_bitexact(x, W1, b1, W2, b2):
    """
    Forward pass using BitExact ops:
      - Linear layers implemented with bitexact.matmul
      - Bias via tensor add
      - Normalization via bitexact.rms_norm
      - Activation via bitexact.sigmoid
    """
    # First linear: x @ W1 + b1
    h = bitexact.matmul(x, W1) + b1

    # RMSNorm on hidden (per last dim) with scale=ones
    # weight must be (hidden_dim,)
    scale = torch.ones(h.shape[-1], device=DEVICE, dtype=DTYPE)
    h = bitexact.rms_norm(h, scale, EPS)

    # Second linear: h @ W2 + b2
    y_lin = bitexact.matmul(h, W2) + b2

    # Activation
    y = bitexact.sigmoid(y_lin)
    return y


def make_fixed_weights(in_dim=64, hidden=32, out_dim=10):
    """
    Create deterministic weights/biases without RNG, so we're testing kernel determinism,
    not initialization noise.
    """
    W1 = torch.linspace(-0.5, 0.5, steps=in_dim * hidden, device=DEVICE, dtype=DTYPE).reshape(in_dim, hidden).contiguous()
    b1 = torch.linspace(-0.1, 0.1, steps=hidden, device=DEVICE, dtype=DTYPE).contiguous()
    W2 = torch.linspace(-0.3, 0.3, steps=hidden * out_dim, device=DEVICE, dtype=DTYPE).reshape(hidden, out_dim).contiguous()
    b2 = torch.linspace(-0.05, 0.05, steps=out_dim, device=DEVICE, dtype=DTYPE).contiguous()
    return W1, b1, W2, b2


def make_input(batch=128, in_dim=64):
    # Use a fixed, non-random pattern to remove any RNG from the story
    base = torch.arange(batch * in_dim, device=DEVICE, dtype=DTYPE).reshape(batch, in_dim)
    x = (base % 97) / 97.0  # bounded [0,1)
    return x.contiguous()


def compare_runs(x, W1, b1, W2, b2, runs=3):
    """
    Run forward multiple times and check bitwise equality across all runs.
    If any mismatch occurs, print op-wise diffs to help debug.
    """
    outputs = []
    for i in range(runs):
        # make sure inputs are the exact same tensors each run
        y = forward_bitexact(x, W1, b1, W2, b2)
        outputs.append(y)

    all_equal = True
    for i in range(1, runs):
        equal = torch.equal(outputs[0], outputs[i])
        print(f"Run 0 vs Run {i}: bit-equal = {equal}")
        if not equal:
            all_equal = False
            max_diff = (outputs[0] - outputs[i]).abs().max().item()
            print(f"  Max abs diff: {max_diff:.9e}")

    # Quick per-op probe (only if a mismatch is detected)
    if not all_equal:
        print("\nPer-op diagnostic:")
        # Recompute each op twice on the same inputs to isolate nondeterminism
        # 1) First linear
        h1a = bitexact.matmul(x, W1) + b1
        h1b = bitexact.matmul(x, W1) + b1
        print(f"  matmul #1 bit-equal: {torch.equal(h1a, h1b)}  max diff: {(h1a - h1b).abs().max().item():.3e}")

        # 2) RMSNorm
        scale = torch.ones(h1a.shape[-1], device=DEVICE, dtype=DTYPE)
        n1a = bitexact.rms_norm(h1a, scale, EPS)
        n1b = bitexact.rms_norm(h1b, scale, EPS)
        print(f"  rms_norm bit-equal: {torch.equal(n1a, n1b)}  max diff: {(n1a - n1b).abs().max().item():.3e}")

        # 3) Second linear
        h2a = bitexact.matmul(n1a, W2) + b2
        h2b = bitexact.matmul(n1b, W2) + b2
        print(f"  matmul #2 bit-equal: {torch.equal(h2a, h2b)}  max diff: {(h2a - h2b).abs().max().item():.3e}")

        # 4) Sigmoid
        s1a = bitexact.sigmoid(h2a)
        s1b = bitexact.sigmoid(h2b)
        print(f"  sigmoid bit-equal: {torch.equal(s1a, s1b)}  max diff: {(s1a - s1b).abs().max().item():.3e}")

    return all_equal


def main():
    if DEVICE != "cuda":
        print("WARNING: CUDA not available; BitExact kernels require CUDA. Exiting.")
        return

    # Construct deterministic data + weights
    in_dim, hidden, out_dim = 64, 32, 10
    x = make_input(batch=128, in_dim=in_dim)
    W1, b1, W2, b2 = make_fixed_weights(in_dim, hidden, out_dim)

    print("BitExact deterministic inference demo")
    print(f"Device: {DEVICE}  dtype: {DTYPE}  input: {tuple(x.shape)}")

    ok = compare_runs(x, W1, b1, W2, b2, runs=3)
    print("\nRESULT:", "Bitwise reproducible across runs" if ok else "Not bitwise reproducible — see diagnostics above")


if __name__ == "__main__":
    main()
