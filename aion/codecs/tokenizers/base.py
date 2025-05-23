from abc import ABC, abstractmethod

import torch
from jaxtyping import Float
from torch import Tensor
from typing import Dict, Type, Optional, Any

from aion.modalities import ModalityType, Modality
from aion.codecs.quantizers import Quantizer


class Codec(ABC, torch.nn.Module):
    """Abstract definition of the Codec API.

    A codec embeds a specific type of data into a sequence of either
    discrete tokens or continuous embedddings, and then decode it back.
    """

    @property
    @abstractmethod
    def modality(self) -> Type[Modality]:
        """Returns the modality key that this codec can operate on."""
        raise NotImplementedError

    @abstractmethod
    def _encode(self, x: ModalityType) -> Float[Tensor, "b c1 *code_shape"]:
        """Function to be implemented by subclasses which
        takes a batch of input samples (as a ModalityType instance)
        and embedds it into a latent space, before any quantization.
        """
        raise NotImplementedError

    @abstractmethod
    def _decode(
        self, z: Float[Tensor, "b c1 *code_shape"], **metadata: Optional[Dict[str, Any]]
    ) -> ModalityType:
        """Function to be implemented by subclasses which
        takes a batch of latent space embeddings (after dequantization)
        and decodes it into the original input space as a ModalityType instance.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def quantizer(self) -> "Quantizer":
        """If the codec is quantized, return the quantizer,
        otherwise returns None.
        """
        raise NotImplementedError

    def encode(self, x: ModalityType) -> Float[Tensor, "b c1 *code_shape"]:
        """Encodes a given batch of samples into latent space."""
        # Verify that the input type corresponds to the modality of the codec
        if not isinstance(x, self.modality):
            raise ValueError(
                f"Input type {type(x).__name__} does not match the modality of the codec {self.modality.__name__}"
            )
        embedding = self._encode(x)
        if self.is_quantized:
            return self.quantizer.encode(embedding)
        else:
            return embedding

    def decode(
        self, z: Float[Tensor, "b c1 *code_shape"], **metadata: Optional[Dict[str, Any]]
    ) -> ModalityType:
        """Decodes a given batch of samples from latent space."""
        if self.quantizer is not None:
            z = self.quantizer.decode(z)
        return self._decode(z, **metadata)

    @property
    def is_quantized(self) -> bool:
        """Whether this codec is quantizer or not."""
        return self.quantizer is not None
