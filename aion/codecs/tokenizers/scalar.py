from typing import Dict, Optional

import torch
from jaxtyping import Float
from torch import Tensor

from aion.codecs.quantizers import Quantizer
from aion.codecs.tokenizers.base import QuantizedCodec


class ScalarIdentityCodec(QuantizedCodec):
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
        super().__init__(quantizer)
        self._modality = modality

    @property
    def modality(self):
        return self._modality

    def _encode(self, x: Dict[str, Dict[str, Float[Tensor, "b t"]]]) -> Tensor:
        return x[self.modality]

    def _decode(self, z: torch.FloatTensor) -> Dict[str, torch.FloatTensor]:
        return {self.modality: z}
