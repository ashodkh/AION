import math
from typing import Optional

import scipy.interpolate
import torch
import torch.nn as nn

from aion.codecs.quantizers import Quantizer


class ScalarReservoirQuantizer(Quantizer):
    """
    Scalar quantizer module.

    The scalar quantizer module takes a batch of scalars and quantizes them using a CDF codebook.
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

    The scalar quantizer module takes a batch of scalars and quantizes them using a CDF codebook.
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


class ScalarCompressedReservoirQuantizer(ScalarReservoirQuantizer):
    """
    Scalar quantizer module with compression/decompression functions.

    The scalar quantizer module takes a batch of scalars, applies compression functions,
    and quantizes them using a CDF codebook. The CDF estimate is updated using reservoir
    sampling, allowing you to stream through data.

    Args:
        compression_fns: list[str]
            List of torch function names to apply for compression (e.g., ['arcsinh']).
        decompression_fns: list[str]
            List of torch function names to apply for decompression (e.g., ['sinh']).
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
        compression_fns: list[str],
        decompression_fns: list[str],
        codebook_size: int,
        reservoir_size: int,
        reservoir_default: Optional[float] = 0.0,
    ):
        super().__init__(codebook_size, reservoir_size, reservoir_default)
        assert len(compression_fns) == len(decompression_fns), (
            "Mismatched compression/decompression functions"
        )
        self.compression_fns = compression_fns
        self.decompression_fns = decompression_fns

        assert self._check_identity(torch.tensor([1.0])), (
            "Identity check failed, compression/decompression functions are not inverses."
        )

    def compress(self, x: torch.Tensor) -> torch.Tensor:
        """Apply compression functions to input tensor.

        Args:
            x: torch.Tensor
                Input tensor to compress.

        Returns:
            torch.Tensor
                Compressed tensor.
        """
        for c in self.compression_fns:
            x = getattr(torch, c)(x)
        return x

    def decompress(self, x: torch.Tensor) -> torch.Tensor:
        """Apply decompression functions to input tensor.

        Args:
            x: torch.Tensor
                Input tensor to decompress.

        Returns:
            torch.Tensor
                Decompressed tensor.
        """
        for c in self.decompression_fns[::-1]:
            x = getattr(torch, c)(x)
        return x

    def _check_identity(self, x: torch.Tensor) -> bool:
        """Check if compression and decompression are inverses.

        Args:
            x: torch.Tensor
                Test tensor.

        Returns:
            bool
                True if compress(decompress(x)) ≈ x.
        """
        return torch.allclose(self.decompress(self.compress(x)), x)

    def _update_reservoirs(self, z_e: torch.Tensor):
        z_e = self.compress(z_e)
        super()._update_reservoirs(z_e)

    def encode(self, z: torch.Tensor) -> torch.Tensor:
        z = self.compress(z)
        return super().encode(z)

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        return self.decompress(super().decode(codes))


class MultiScalarCompressedReservoirQuantizer(Quantizer):
    """
    Multi-channel scalar quantizer with compression.

    Wraps multiple ScalarCompressedReservoirQuantizers to quantize multi-channel tensors.
    Each channel is quantized independently with its own reservoir.

    Args:
        compression_fns: list[str]
            List of torch function names to apply for compression (e.g., ['arcsinh']).
        decompression_fns: list[str]
            List of torch function names to apply for decompression (e.g., ['sinh']).
        codebook_size: int
            The number of codes in the codebook.
        reservoir_size: int
            The size of the reservoir to keep in memory.
        reservoir_default: float
            Optional default value of reservoir samples.
        num_quantizers: int
            Number of channels/quantizers to create.
    """

    def __init__(
        self,
        compression_fns: list[str],
        decompression_fns: list[str],
        codebook_size: int,
        reservoir_size: int,
        reservoir_default: Optional[float] = 0.0,
        num_quantizers: int = 1,
    ):
        super().__init__()
        self.quantizers = nn.ModuleList(
            [
                ScalarCompressedReservoirQuantizer(
                    compression_fns,
                    decompression_fns,
                    codebook_size,
                    reservoir_size,
                    reservoir_default,
                )
                for _ in range(num_quantizers)
            ]
        )
        self.num_quantizers = num_quantizers

    def encode(self, z: torch.Tensor) -> torch.Tensor:
        """Encodes the input tensor z, returns the corresponding
        codebook index.

        Args:
            z: torch.Tensor (B, C)
                The input tensor to be encoded.

        Returns:
            codes: torch.Tensor (B, C)
                Encoded tensor.
        """
        return torch.stack(
            [q.encode(z[:, i]) for i, q in enumerate(self.quantizers)],
            dim=1,
        )

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """Decodes the input code index into corresponding codebook entry of
        dimension (embedding_dim).

        Args:
            codes: torch.Tensor (B, C)
                Codes to be decoded.

        Returns:
            z: torch.Tensor (B, C)
                Decoded sample.
        """
        return torch.stack(
            [q.decode(codes[:, i]) for i, q in enumerate(self.quantizers)],
            dim=1,
        )

    def quantize(self, z: torch.Tensor) -> torch.Tensor:
        """Quantize the input tensor z, returns corresponding
        codebook entry.

        Args:
            z: torch.Tensor (B, C)
                The input tensor to be quantized.

        Returns:
            z: torch.Tensor (B, C)
                Quantized tensor.
        """
        return self.decode(self.encode(z))

    def _update_reservoirs(self, z_e: torch.Tensor):
        for i, q in enumerate(self.quantizers):
            q._update_reservoirs(z_e[:, i])

    def forward(
        self, z_e: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Performs a forward pass through the vector quantizer.
        Args:
            z_e: torch.Tensor (B, C, ...)
                The input tensor to be quantized.
        Returns:
            z_q: torch.Tensor
                The quantized tensor.
            loss: torch.Tensor
                The embedding loss for the quantization.
            codebook_usage: torch.Tensor
                The fraction of codes used in the codebook.
        """
        self._update_reservoirs(z_e)
        indices = self.encode(z_e)
        z_q = self.decode(indices)
        num_unique = sum([len(torch.unique(c)) for c in indices.T])
        codebook_usage = num_unique / (self.codebook_size * self.num_quantizers)
        return z_q, torch.nn.functional.mse_loss(z_q, z_e), torch.tensor(codebook_usage)

    @property
    def codebook_size(self) -> int:
        """Returns the size of the codebook."""
        return self.quantizers[0].codebook_size

    @property
    def codebook(self) -> torch.Tensor:
        """Returns the codebook."""
        return self.quantizers[0].codebook

    @property
    def embedding_dim(self) -> int:
        """Returns the dimension of the codebook entries."""
        return 1
