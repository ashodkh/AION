import torch
from torch import package

from .image import ImageCodec


def load_tokenizer(path: str, device: str = "cpu") -> torch.nn.Module:
    importer = package.PackageImporter(path)
    model = importer.load_pickle("network", "network.pkl", map_location=device)
    return model


__all__ = ["load_tokenizer", "ImageCodec"]
