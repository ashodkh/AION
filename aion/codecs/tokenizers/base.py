from abc import ABC, abstractmethod

import torch
from jaxtyping import Float, Dict
from torch import Tensor

from aion.codecs.quantizers import Quantizer


class Codec(ABC, torch.nn.Module):
    """Abstract definition of the Codec API.

    A codec embeds a specific type of data into a sequence of either
    discrete tokens or continuous embedddings, and then decode it back.
    """

    @property
    @abstractmethod
    def modality(self) -> str:
        """Returns the modality key that this codec can operate on."""
        raise NotImplementedError

    @abstractmethod
    def _encode(
        self, x: Dict[str, Dict[str, Float[Tensor, "b c *input_shape"]]]
    ) -> Float[Tensor, "b c1 *code_shape"]:
        """Function to be implemented by subclasses which
        takes a batch of input samples and embedds it into a
        latent space, before any quantization.
        """
        raise NotImplementedError

    @abstractmethod
    def _decode(
        self, z: Float[Tensor, "b c1 *code_shape"]
    ) -> Dict[str, Dict[str, Float[Tensor, "b c *input_shape"]]]:
        """Function to be implemented by subclasses which
        takes a batch of latent space embeddings (after dequantization)
        and decodes it into the original input space.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def quantizer(self) -> "Quantizer":
        """If the codec is quantized, return the quantizer,
        otherwise returns None.
        """
        raise NotImplementedError

    def encode(
        self, x: Dict[str, Dict[str, Float[Tensor, "b c *input_shape"]]]
    ) -> Float[Tensor, "b c1 *code_shape"]:
        """Encodes a given batch of samples into latent space."""
        embedding = self._encode(x)
        if self.is_quantized:
            return self.quantizer.encode(embedding)
        else:
            return embedding

    def decode(
        self, z: Float[Tensor, "b c1 *code_shape"]
    ) -> Dict[str, Dict[str, Float[Tensor, "b c *input_shape"]]]:
        """Encodes a given batch of samples into latent space."""
        if self._quantizer is not None:
            z = self.quantizer.decode(z)
        return self._decode(z)

    @property
    def is_quantized(self) -> bool:
        """Whether this codec is quantizer or not."""
        return self.quantizer is not None
