from abc import ABC, abstractmethod

import torch
from jaxtyping import Float, Bool

from aion.codecs.quantizers import Quantizer


class Codec(ABC, torch.nn.Module):
    """Abstract definition of a Codec.

    A codec embeds a specific type of data into a sequence of either
    discrete tokens or continuous embeddings, and then decode it back.
    """

    @property
    @abstractmethod
    def modality(self) -> str:
        """Returns the modality key that this codec can operate on."""
        raise NotImplementedError

    @abstractmethod
    def _encode(
        self, x: Float[torch.Tensor, " b c *input_shape"]
    ) -> Float[torch.Tensor, " b c1 *code_shape"]:
        """Function to be implemented by subclasses which
        takes a batch of input samples and embedds it into a
        latent space, before any quantization.
        """
        raise NotImplementedError

    @abstractmethod
    def _decode(
        self, z: Float[torch.Tensor, " b c1 *code_shape"]
    ) -> Float[torch.Tensor, " b c *input_shape"]:
        """Function to be implemented by subclasses which
        takes a batch of latent space embeddings (after dequantization)
        and decodes it into the original input space.
        """
        raise NotImplementedError

    def encode(
        self,
        x: Float[torch.Tensor, " b c *input_shape"],
        channel_mask: Bool[torch.Tensor, " b c"],
    ) -> Float[torch.Tensor, " b c1 *code_shape"]:
        """Encodes a given batch of samples into latent space."""
        return self._encode(x, channel_mask)

    def decode(
        self, z: Float[torch.Tensor, " b c1 *code_shape"]
    ) -> Float[torch.Tensor, " b c *input_shape"]:
        """Encodes a given batch of samples into latent space."""
        return self._decode(z)

    def forward(
        self,
        x: Float[torch.Tensor, " b c *input_shape"],
        channel_mask: Bool[torch.Tensor, " b c"],
    ) -> Float[torch.Tensor, " b c1 *code_shape"]:
        return self.encode(x, channel_mask)


class QuantizedCodec(Codec):
    def __init__(self, quantizer: Quantizer):
        super().__init__()
        self.quantizer = quantizer

    def decode(
        self, z: Float[torch.Tensor, " b c1 *code_shape"]
    ) -> Float[torch.Tensor, " b c *input_shape"]:
        z = self.quantizer.decode(z)
        return self._decode(z)

    def encode(
        self,
        x: Float[torch.Tensor, " b c *input_shape"],
        channel_mask: Bool[torch.Tensor, " b c"],
    ) -> Float[torch.Tensor, " b c1 *code_shape"]:
        embedding = super().encode(x, channel_mask)
        return self.quantizer.encode(embedding)
