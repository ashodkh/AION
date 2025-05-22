import torch


def range_compression(
    sample: torch.Tensor, div_factor: float | int = 0.01
) -> torch.Tensor:
    """Applies arcsinh compression on each band of the input."""
    return torch.arcsinh(sample / div_factor) * div_factor


def reverse_range_compression(
    sample: torch.Tensor, div_factor: float | int = 0.01
) -> torch.Tensor:
    """Undoes arcsinh compression on each band of the input."""
    return torch.sinh(sample / div_factor) * div_factor