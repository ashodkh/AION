import pytest
import torch

from aion.codecs.tokenizers.scalar import (
    LogScalarCodec,
    ScalarCodec,
)
from aion.modalities import (
    FluxG,
    FluxR,
    FluxI,
    FluxZ,
    FluxW1,
    FluxW2,
    FluxW3,
    FluxW4,
    ShapeR,
    ShapeE1,
    ShapeE2,
    EBV,
)


@pytest.mark.parametrize(
    "codec_class,modality",
    [
        # LogScalarCodec tests
        (LogScalarCodec, FluxG),
        (LogScalarCodec, FluxR),
        (LogScalarCodec, FluxI),
        (LogScalarCodec, FluxZ),
        (LogScalarCodec, FluxW1),
        (LogScalarCodec, FluxW2),
        (LogScalarCodec, FluxW3),
        (LogScalarCodec, FluxW4),
        (LogScalarCodec, ShapeR),
        # ScalarCodec tests
        (ScalarCodec, ShapeE1),
        (ScalarCodec, ShapeE2),
        (ScalarCodec, EBV),
    ],
)
def test_scalar_tokenizer(data_dir, codec_class, modality):
    codec = codec_class.from_pretrained(
        f"polymathic-ai/aion-scalar-{modality.name.lower().replace('_', '-')}-codec"
    )
    codec.eval()
    input_batch = torch.load(
        data_dir / f"{modality.name}_codec_input_batch.pt", weights_only=False
    )
    reference_encoded_batch = torch.load(
        data_dir / f"{modality.name}_codec_encoded_batch.pt", weights_only=False
    )
    reference_decoded_batch = torch.load(
        data_dir / f"{modality.name}_codec_decoded_batch.pt", weights_only=False
    )

    with torch.no_grad():
        output = codec.encode(modality(value=input_batch))
        decoded_output = codec.decode(output)

    assert torch.allclose(output, reference_encoded_batch)
    assert torch.allclose(decoded_output.value, reference_decoded_batch)
