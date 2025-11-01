import torch, pytest

@pytest.fixture(autouse=True, scope="session")
def global_cuda_setup(pytestconfig):
    """
    Automatically runs once per test session.
    - Sets manual seed for reproducibility.
    - Configures deterministic mode if enabled.
    - Clears GPU memory before and after tests.
    """
    if (not torch.cuda.is_available()):
        print("You do not have CUDA installed - exiting tests")
        return

    print("\n[BitExact] Initializing CUDA test environment...")
    torch.manual_seed(42)

    # Clear any previous GPU allocations
    torch.cuda.empty_cache()

    yield

    print("\n[BitExact] Cleaning up GPU cache...\n")
    torch.cuda.empty_cache()