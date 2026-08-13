def pytest_configure(config) -> None:
    config.addinivalue_line("markers", "gpu_smoke: requires the production NVIDIA CUDA runtime")
