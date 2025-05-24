from collections import OrderedDict
from typing import Type, Optional, Dict, List

from huggingface_hub import PyTorchModelHubMixin
import torch
from jaxtyping import Float
from torch import Tensor

from aion.codecs.base import Codec
from aion.codecs.quantizers import Quantizer
from aion.codecs.quantizers.scalar import ComposedScalarQuantizer
from aion.modalities import Catalog

__all__ = ["CatalogIdentityCodec"]


class CatalogIdentityCodec(Codec, PyTorchModelHubMixin):
    """Codec for catalog quantities.

    A codec that embeds catalog quantities through an identity mapping. A
    quantizer is applied if specified.

    Args:
        catalog_keys: List[str]
            List of catalog keys to encode.
        quantizers: Optional[List[Quantizer]]
            Optional list of quantizers for each catalog key.
        mask_value: int
            Value used to indicate masked/missing data.
    """

    def __init__(
        self,
        catalog_keys: List[str],
        quantizers: Optional[List[Quantizer]] = None,
        mask_value: int = 9999,
    ):
        super().__init__()
        self._modality = Catalog
        self._catalog_keys = catalog_keys
        self.mask_value = mask_value
        if quantizers:
            assert len(catalog_keys) == len(quantizers), (
                "Number of catalog keys and quantizers must match"
            )
            _quantizer = OrderedDict()
            for key, quantizer in zip(catalog_keys, quantizers):
                _quantizer[key] = quantizer
            self._quantizer = ComposedScalarQuantizer(_quantizer)
        else:
            self._quantizer = None

    @property
    def modality(self) -> Type[Catalog]:
        return self._modality

    @property
    def quantizer(self) -> Optional[Quantizer]:
        return self._quantizer

    def _encode(self, x: Catalog) -> Dict[str, Tensor]:
        encoded = OrderedDict()
        for key in self._catalog_keys:
            catalog_value = x[self.modality][key]
            mask = catalog_value != self.mask_value
            catalog_value = catalog_value[mask]
            encoded[key] = catalog_value
        encoded["mask"] = mask
        return encoded

    def encode(self, x: Catalog) -> Float[Tensor, "b c1 *code_shape"]:
        """Encodes a given batch of samples into latent space."""
        embedding = self._encode(x)
        _encoded = self.quantizer.encode(
            embedding
        )  # (b, C), where b is the number of non-masked samples

        mask = embedding["mask"]
        # B: batch size, L: sequence length (20) for each catalog key
        B, L = mask.shape
        C = len(self._catalog_keys)
        encoded = self.mask_value * torch.ones(
            B, L, C, dtype=_encoded.dtype, device=_encoded.device
        )
        encoded[mask] = _encoded
        encoded = encoded.reshape(B, -1)
        return encoded

    def _decode(self, z: Dict[str, Tensor]) -> Catalog:
        return Catalog(data=z)

    def decode(self, z: Float[Tensor, "b c1 *code_shape"]) -> Catalog:
        B, LC = z.shape
        C = len(self._catalog_keys)
        L = LC // C
        z = z[:, : C * L]  # Truncate the z if it is longer than the expected length
        z = z.reshape(B * L, C)
        if self._quantizer is not None:
            z = self.quantizer.decode(z)
        for key in self._catalog_keys:
            z[key] = z[key].reshape(B, L)
        return self._decode(z)
