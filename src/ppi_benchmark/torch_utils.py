"""Small shared helpers for optional Torch-backed components."""

from __future__ import annotations


TORCH_DEVICE_CHOICES = ("auto", "cpu", "cuda", "mps")
TORCH_TRAINING_PRECISIONS = ("float32", "float16", "bfloat16")


def resolve_torch_device(requested_device: str):
    """Resolve and validate a requested Torch device lazily."""
    import torch

    if requested_device not in TORCH_DEVICE_CHOICES:
        raise ValueError(
            f"device must be one of: {', '.join(TORCH_DEVICE_CHOICES)}."
        )
    if requested_device == "auto":
        if torch.cuda.is_available():
            device_name = "cuda"
        elif torch.backends.mps.is_available():
            device_name = "mps"
        else:
            device_name = "cpu"
    else:
        device_name = requested_device
    if device_name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available.")
    if device_name == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested but is not available.")
    return torch.device(device_name)
