"""Device detection and utilities."""

import os
import torch

# Default device (auto-detected on import)
DEFAULT_DEVICE = None


def get_device(device: str | None = None) -> torch.device:
    """Get the best available device.

    Args:
        device: If specified, use this device. Otherwise, auto-detect.

    Returns:
        torch.device: The device to use.
    """
    global DEFAULT_DEVICE

    if device is not None:
        return torch.device(device)

    if DEFAULT_DEVICE is None:
        if torch.cuda.is_available():
            DEFAULT_DEVICE = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            DEFAULT_DEVICE = torch.device("mps")
        else:
            DEFAULT_DEVICE = torch.device("cpu")

    return DEFAULT_DEVICE


def set_num_threads(num_threads: int | None) -> None:
    """Set the number of CPU threads PyTorch uses.

    This controls PyTorch's CPU parallelism and can help limit CPU usage
    on systems with many cores (e.g., MacBook Pro) to avoid overwhelming
    the system or to allocate cores between training and serving.

    Args:
        num_threads: Number of threads to use. None uses all available cores.
    """
    if num_threads is not None:
        torch.set_num_threads(num_threads)
        torch.set_num_interop_threads(max(1, num_threads // 2))
        os.environ["OMP_NUM_THREADS"] = str(num_threads)
        os.environ["MKL_NUM_THREADS"] = str(num_threads)
