from typing import Type, Optional, Dict, Any

from huggingface_hub import PyTorchModelHubMixin
from jaxtyping import Float
from torch import Tensor

from aion.codecs.quantizers import Quantizer
from aion.codecs.quantizers.scalar import (
    ScalarLogReservoirQuantizer,
    ScalarReservoirQuantizer,
)
from aion.codecs.tokenizers.base import Codec
from aion.modalities import (
    ScalarModality,
    ScalarModalityType,
)


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

    def __init__(
        self,
        modality_class: Type[ScalarModality],
        quantizer: Optional[Quantizer] = None,
    ):
        super().__init__()
        self._quantizer = quantizer
        self._modality_class = modality_class

    @property
    def quantizer(self) -> Quantizer:
        return self._quantizer

    @property
    def modality(self) -> Type[ScalarModality]:
        return self._modality_class

    def _encode(self, x: ScalarModality) -> Float[Tensor, "b 1"]:
        # Scalar values need to be reshaped to have a channel dimension
        return x.value.unsqueeze(1)  # Shape: [batch] -> [batch, 1]

    def _decode(
        self, z: Float[Tensor, "b 1"], **metadata: Optional[Dict[str, Any]]
    ) -> ScalarModality:
        # Remove the channel dimension for scalar values
        value = z.squeeze(1)  # Shape: [batch, 1] -> [batch]
        return self._modality_class(value=value)

    @classmethod
    def from_pretrained(cls, model_id: str, **kwargs):
        """Load a pretrained codec from HuggingFace Hub.

        This method provides backward compatibility by extracting the modality
        name from the model_id and mapping it to the appropriate modality class.
        """
        # Extract modality name from model_id (e.g., "polymathic-ai/aion-scalar-flux-g-codec" -> "FLUX_G")
        parts = model_id.split("-")
        if len(parts) >= 4 and parts[2] == "scalar":
            # Handle model IDs like "aion-scalar-flux-g-codec"
            modality_parts = parts[3:]
            if modality_parts[-1] == "codec":
                modality_parts = modality_parts[:-1]
            modality_name = "_".join(part.upper() for part in modality_parts)

            # Map to modality class
            if modality_name in ScalarModalityType.__members__:
                modality_class = ScalarModalityType[modality_name]
                # Load the model state
                loaded = super(BaseScalarIdentityCodec, cls).from_pretrained(
                    model_id, **kwargs
                )
                # Update the modality class
                loaded._modality_class = modality_class
                return loaded

        # Fallback to default loading
        return super().from_pretrained(model_id, **kwargs)


class ScalarCodec(BaseScalarIdentityCodec):
    def __init__(
        self,
        modality_class: Type[ScalarModality],
        codebook_size: int,
        reservoir_size: int,
    ):
        quantizer = ScalarReservoirQuantizer(
            codebook_size=codebook_size,
            reservoir_size=reservoir_size,
        )
        super().__init__(modality_class, quantizer)


class LogScalarCodec(BaseScalarIdentityCodec):
    def __init__(
        self,
        modality_class: Type[ScalarModality],
        codebook_size: int,
        reservoir_size: int,
    ):
        quantizer = ScalarLogReservoirQuantizer(
            codebook_size=codebook_size,
            reservoir_size=reservoir_size,
        )
        super().__init__(modality_class, quantizer)
