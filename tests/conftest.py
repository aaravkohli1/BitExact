import torch, pytest

def pytest_configure(config):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available, skipping BitExact tests", allow_module_level=True)
