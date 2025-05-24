from typing import Type, Optional, Dict, Any

from huggingface_hub import PyTorchModelHubMixin
from jaxtyping import Float
from torch import Tensor

from aion.codecs.quantizers import Quantizer
from aion.codecs.quantizers.scalar import (
    ScalarLogReservoirQuantizer,
    ScalarReservoirQuantizer,
    MultiScalarCompressedReservoirQuantizer,
)
from aion.codecs.base import Codec
from aion.modalities import ScalarModality, ScalarModalities


class BaseScalarIdentityCodec(Codec, PyTorchModelHubMixin):
    """Codec for scalar quantities.

    A codec that embeds scalar quantities through an identity mapping. A
    quantizer is applied if specified.

    Args:
        modality_class: Type[ScalarModality]
            The modality class this codec is designed for.
        quantizer: Quantizer
            Optional quantizer for the scalar values.
    """

    @property
    def quantizer(self) -> Quantizer:
        return self._quantizer

    @property
    def modality(self) -> Type[ScalarModality]:
        return self._modality_class

    def _encode(self, x: ScalarModality) -> Float[Tensor, " b"]:
        return x.value

    def _decode(
        self, z: Float[Tensor, " b"], **metadata: Optional[Dict[str, Any]]
    ) -> ScalarModality:
        return self._modality_class(value=z)

    def load_state_dict(self, state_dict, strict=True):
        # This function is just because the scalar codecs were saved with 'quantizer' instead of '_quantizer'
        remapped_state_dict = {
            (
                k.replace("quantizer", "_quantizer", 1)
                if k.startswith("quantizer")
                else k
            ): v
            for k, v in state_dict.items()
        }
        return super().load_state_dict(remapped_state_dict, strict=strict)


class ScalarCodec(BaseScalarIdentityCodec):
    def __init__(
        self,
        modality: str,
        codebook_size: int,
        reservoir_size: int,
    ):
        super().__init__()
        self._modality_class = next(m for m in ScalarModalities if m.name == modality)
        self._quantizer = ScalarReservoirQuantizer(
            codebook_size=codebook_size,
            reservoir_size=reservoir_size,
        )


class LogScalarCodec(BaseScalarIdentityCodec):
    def __init__(
        self,
        modality: str,
        codebook_size: int,
        reservoir_size: int,
        min_log_value: float | None = -3,
    ):
        super().__init__()
        self._modality_class = next(m for m in ScalarModalities if m.name == modality)
        self._quantizer = ScalarLogReservoirQuantizer(
            codebook_size=codebook_size,
            reservoir_size=reservoir_size,
            min_log_value=min_log_value,
        )


class MultiScalarCodec(BaseScalarIdentityCodec):
    """Codec for multi-channel scalar quantities with compression.

    A codec that handles multi-channel scalar modalities using compression
    and decompression functions before quantization. This is particularly useful
    for spectral coefficients or other multi-dimensional scalar data that
    benefits from preprocessing transformations.

    Each channel is quantized independently using a compressed reservoir quantizer,
    allowing for different statistical distributions across channels while
    maintaining the ability to handle streaming data.

    Args:
        modality: str
            The name of the modality this codec is designed for. Must match
            a modality name defined in the ScalarModalities registry.
        compression_fns: list[str]
            List of PyTorch function names to apply for compression (e.g., ['arcsinh']).
            These functions are applied in order to transform the data before quantization.
        decompression_fns: list[str]
            List of PyTorch function names to apply for decompression (e.g., ['sinh']).
            These functions are applied in reverse order during decoding to restore
            the original data range.
        codebook_size: int
            The number of codes in each quantizer's codebook.
        reservoir_size: int
            The size of the reservoir to keep in memory for each channel's quantizer.
        num_quantizers: int
            Number of channels/quantizers to create, corresponding to the number
            of dimensions in the multi-channel scalar data.

    Note:
        The compression and decompression functions must be mathematical inverses
        of each other. The codec will verify this during initialization and raise
        an assertion error if the functions are not properly inverse.

    Example:
        >>> codec = MultiScalarCodec(
        ...     modality="bp_coefficients",
        ...     compression_fns=["arcsinh"],
        ...     decompression_fns=["sinh"],
        ...     codebook_size=1024,
        ...     reservoir_size=10000,
        ...     num_quantizers=55
        ... )
    """

    def __init__(
        self,
        modality: str,
        compression_fns: list[str],
        decompression_fns: list[str],
        codebook_size: int,
        reservoir_size: int,
        num_quantizers: int,
    ):
        super().__init__()
        self._modality_class = next(m for m in ScalarModalities if m.name == modality)
        self._quantizer = MultiScalarCompressedReservoirQuantizer(
            compression_fns=compression_fns,
            decompression_fns=decompression_fns,
            codebook_size=codebook_size,
            reservoir_size=reservoir_size,
            num_quantizers=num_quantizers,
        )
