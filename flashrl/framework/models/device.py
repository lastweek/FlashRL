"""Device detection and utilities."""

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
