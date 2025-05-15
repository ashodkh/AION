from abc import abstractmethod, ABC
import pytorch_lightning as L
from typing import Dict, Optional
import math
import scipy.interpolate

from jaxtyping import Float

from torch import Tensor
import torch
from torch import nn


class Codec(L.LightningModule, ABC):
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


class Quantizer(nn.Module):
    """Abstract interface for all quantizer modules."""

    @abstractmethod
    def forward(self, z_e: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Performs a forward pass through the vector quantizer.
        Args:
            z_e: Tensor (B, C, ...)
                The input tensor to be quantized.
        Returns:
            z_q: Tensor
                The quantized tensor.
            loss: Tensor
                The embedding loss for the quantization.
            codebook_usage: Tensor
                The fraction of codes used in the codebook.
        """
        raise NotImplementedError

    def quantize(self, z: Tensor) -> Tensor:
        """Quantize the input tensor z, returns corresponding
        codebook entry.
        """
        return self.decode(self.encode(z))

    @abstractmethod
    def encode(self, z: Tensor) -> Tensor:
        """Encodes the input tensor z, returns the corresponding
        codebook index.
        """
        raise NotImplementedError

    @abstractmethod
    def decode(self, codes: Tensor) -> Tensor:
        """Decodes the input code index into corresponding codebook entry of
        dimension (embedding_dim).
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def codebook_size(self) -> int:
        """Returns the size of the codebook."""
        raise NotImplementedError

    @property
    def codebook(self) -> Tensor:
        """Returns the codebook."""
        return self.decode(torch.arange(self.codebook_size))

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Returns the dimension of the codebook entries."""
        raise NotImplementedError


class ScalarReservoirQuantizer(Quantizer):
    """
    Scalar quantizer module.

    The sclar quantizer module takes a batch of scalars and quantizes them using a CDF codebook.
    The CDF estimate is updated using reservoir sampling, allowing you to stream through data.

    Args:
        codebook_size: int
            The number of codes in the codebook.
        reservoir_size: int
            The size of the reservoir to keep in memory.
        reservoir_default: float
            Optional default value of reservoir samples. Only relevant if there
            are fewer samples in your dataset than the size of your codebook.
    """

    def __init__(
        self,
        codebook_size: int,
        reservoir_size: int,
        reservoir_default: Optional[float] = 0.0,
    ):
        super().__init__()

        self._codebook_size = codebook_size
        self._reservoir_size = reservoir_size
        _reservoir = torch.ones(reservoir_size) * reservoir_default
        self.register_buffer("_reservoir", _reservoir)

        # Qunatiles for CDF reconstruction.
        self._reservoir_quantile = torch.linspace(0, 1, self._reservoir_size)
        self._quantile = torch.linspace(0, 1, self._codebook_size)

        # Initialize index_to_val.
        self.register_buffer("_index_to_val", None)
        self._generate_index_to_val()

        self._n_total_samples = 0

    @property
    def codebook_size(self) -> int:
        """Returns the size of the codebook."""
        return self._codebook_size

    @property
    def codebook(self) -> torch.Tensor:
        """Returns the codebook."""
        return self._index_to_val

    @property
    def embedding_dim(self) -> int:
        """Returns the dimension of the codebook entries."""
        return 1

    def _generate_index_to_val(self):
        """
        Generate the indices for quantization from reservoir.
        """
        sorted_reservoir, _ = torch.sort(self._reservoir)
        get_inverse_cumulative = scipy.interpolate.interp1d(
            self._reservoir_quantile.cpu().numpy(),
            sorted_reservoir.cpu().numpy(),
            fill_value=(0.0, 1.0),
            bounds_error=False,
        )
        self._index_to_val = torch.tensor(
            get_inverse_cumulative(self._quantile),
            dtype=self._reservoir.dtype,
            device=self._reservoir.device,
        )

    def _update_reservoirs(self, z_e: torch.Tensor):
        """
        Update the reservoirs using current sample.

        Args:
            z_e: torch.Tensor (B)
                The input tensor to be quantized.
        """
        n_samples = len(z_e)
        # Fill in the reservoir before resampling.
        if self._n_total_samples < self._reservoir_size:
            # The number of new samples is not guaranteed to bring us to the
            # codebook size, so drop any samples that would exceed the reamining
            # reservoir.
            offset = min(self._reservoir_size - self._n_total_samples, n_samples)
            self._reservoir[self._n_total_samples : self._n_total_samples + offset] = (
                z_e[:offset]
            )
        else:
            # If the same index is drawn twice, only one of the draws will be
            # kept. This is the desired behavior.
            rep_ind = torch.randint(0, self._n_total_samples - 1, (n_samples,))
            rep_mask = rep_ind < self._reservoir_size
            self._reservoir[rep_ind[rep_mask]] = z_e[rep_mask]

        self._n_total_samples += n_samples

        # Update our cdf estimate using our new reservoir.
        self._generate_index_to_val()

    def forward(
        self, z_e: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Performs a forward pass through the vector quantizer.
        Args:
            z_e: torch.Tensor (B)
                The input tensor to be quantized.
        Returns:
            z_q: torch.Tensor (B)
                The quantized tensor.
            loss: torch.Tensor
                The embedding loss for the quantization.
            codebook_usage: torch.Tensor
                The fraction of codes used in the codebook.
        """
        # Update the reservoirs with the samples
        self._update_reservoirs(z_e)
        z_q = self.quantize(z_e)
        codes = self.encode(z_e)
        codebook_usage = len(torch.unique(codes)) / self.codebook_size
        return z_q, torch.nn.functional.mse_loss(z_q, z_e), torch.tensor(codebook_usage)

    def quantize(self, z: torch.Tensor) -> torch.Tensor:
        """Quantize the input tensor z, returns corresponding
        codebook entry.

        Args:
            z: torch.Tensor (B)
                The input tensor to be quantized.

        Returns:
            z: torch.Tensor (B)
                Quantized tensor.
        """
        return self.decode(self.encode(z))

    def encode(self, z: torch.Tensor) -> torch.Tensor:
        """Encodes the input tensor z, returns the corresponding
        codebook index.

        Args:
            z: torch.Tensor (B)
                The input tensor to be encoded.

        Returns:
            codes: torch.Tensor (B)
                Encoded tensor.
        """
        # Ignoring the last value so bucketize doesn't assign one larger than
        # the boundary values.
        codes = torch.bucketize(z, self._index_to_val[:-1], right=False)
        return codes

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """Decodes the input code index into corresponding codebook entry of
        dimension (embedding_dim).

        Args:
            codes: torch.Tensor (B)
                Codes to be decoded.

        Returns:
            z: torch.Tensor (B)
                Decoded sample.
        """
        samples = self._index_to_val[codes.type(torch.int)]
        return samples


class ScalarLogReservoirQuantizer(ScalarReservoirQuantizer):
    """
    Scalar quantizer module.

    The sclar quantizer module takes a batch of scalars and quantizes them using a CDF codebook.
    The CDF estimate is updated using reservoir sampling, allowing you to stream through data.

    Args:
        codebook_size: int
            The number of codes in the codebook.
        reservoir_size: int
            The size of the reservoir to keep in memory.
        reservoir_default: float
            Optional default (log) value of reservoir samples. Only relevant if
            there are fewer samples in your dataset than the size of your
            codebook.
        min_log_value: float
            Minimum log value to allow in reservoir. Values below this threshold
            will be set to this threshold. Important for scalars that have
            values near zero or that are negative.

    Notes:
        All logs in base e.
    """

    def __init__(
        self,
        codebook_size: int,
        reservoir_size: int,
        reservoir_default: Optional[float] = -3.0,
        min_log_value: Optional[float] = -3.0,
    ):
        # Reservoir should not default to values below the minimum.
        assert reservoir_default >= min_log_value

        super().__init__(codebook_size, reservoir_size, reservoir_default)
        self._min_value = math.exp(min_log_value)

    def _log_and_apply_min(self, z_e: torch.Tensor) -> torch.Tensor:
        """Takes the log base 10 of tensor and applies log minimum.

        Args:
            z_e: torch.Tensor (B)
                The input tensor to be logged.
        Returns:
            z_log: torch.Tensor (B)
                The logged tensor with minimum enforced.
        """
        z_e = z_e.clone()
        z_e[z_e <= self._min_value] = self._min_value
        return torch.log(z_e)

    def _update_reservoirs(self, z_e: torch.Tensor):
        """
        Update the reservoirs using current sample.

        Args:
            z_e: torch.Tensor (B)
                The input tensor to be quantized.
        """
        z_e = self._log_and_apply_min(z_e)
        super()._update_reservoirs(z_e)

    def encode(self, z: torch.Tensor) -> torch.Tensor:
        """Encodes the input tensor z, returns the corresponding
        codebook index.

        Args:
            z: torch.Tensor (B)
                The input tensor to be encoded.

        Returns:
            codes: torch.Tensor (B)
                Encoded tensor.
        """
        # Ignoring the last value so bucketize doesn't assign one larger than
        # the boundary values.
        z = self._log_and_apply_min(z)
        return super().encode(z)

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """Decodes the input code index into corresponding codebook entry of
        dimension (embedding_dim).

        Args:
            codes: torch.Tensor (B)
                Codes to be decoded.

        Returns:
            z: torch.Tensor (B)
                Decoded sample.
        """
        return torch.exp(super().decode(codes))


class ScalarIdentityCodec(Codec):
    """Codec for scalar quantities.

    A codec that embeds scalar quantities through an identity mapping. A
    quantizer is applied if specified.

    Args:
        modality: str
            Name of the modality the codec is designed for.
        quantizer: Quantizer
            Optional quantizer for the scalar values.
    """

    def __init__(self, modality: str, quantizer: Optional[Quantizer] = None):
        super().__init__()
        self._modality = modality
        self._quantizer = quantizer

    @property
    def modality(self):
        return self._modality

    @property
    def quantizer(self):
        return self._quantizer

    def _encode(self, x: Dict[str, Dict[str, Float[Tensor, "b t"]]]) -> Tensor:
        return x[self.modality]

    def _decode(self, z: torch.FloatTensor) -> Dict[str, torch.FloatTensor]:
        return {self.modality: z}
