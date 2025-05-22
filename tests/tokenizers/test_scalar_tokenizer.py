import pytest
import torch

from aion.codecs.tokenizers.scalar import (
    ScalarLogReservoirCodec,
    ScalarReservoirCodec,
)


@pytest.mark.parametrize(
    "modality",
    [
        "FLUX_G",
        "FLUX_R",
        "FLUX_I",
        "FLUX_Z",
        "FLUX_W1",
        "FLUX_W2",
        "FLUX_W3",
        "FLUX_W4",
        "SHAPE_R",
    ],
)
def test_log_reservoir_tokenizer(data_dir, modality):
    codec = ScalarLogReservoirCodec.from_pretrained(
        f"polymathic-ai/aion-scalar-{modality.lower().replace('_', '-')}-codec"
    )
    codec.eval()

    input_batch = torch.load(
        data_dir / f"{modality}_codec_input_batch.pt", weights_only=False
    )
    reference_encoded_batch = torch.load(
        data_dir / f"{modality}_codec_encoded_batch.pt", weights_only=False
    )
    reference_decoded_batch = torch.load(
        data_dir / f"{modality}_codec_decoded_batch.pt", weights_only=False
    )

    with torch.no_grad():
        output = codec.encode({modality: input_batch})
        decoded_output = codec.decode(output)
        decoded_output = decoded_output[modality]

    assert torch.allclose(output, reference_encoded_batch)
    assert torch.allclose(decoded_output, reference_decoded_batch)


@pytest.mark.parametrize("modality", ["SHAPE_E1", "SHAPE_E2", "EBV"])
def test_reservoir_tokenizer(data_dir, modality):
    codec = ScalarReservoirCodec.from_pretrained(
        f"polymathic-ai/aion-scalar-{modality.lower().replace('_', '-')}-codec"
    )
    codec.eval()

    input_batch = torch.load(
        data_dir / f"{modality}_codec_input_batch.pt", weights_only=False
    )
    reference_encoded_batch = torch.load(
        data_dir / f"{modality}_codec_encoded_batch.pt", weights_only=False
    )
    reference_decoded_batch = torch.load(
        data_dir / f"{modality}_codec_decoded_batch.pt", weights_only=False
    )

    with torch.no_grad():
        output = codec.encode({modality: input_batch})
        decoded_output = codec.decode(output)
        decoded_output = decoded_output[modality]

    assert torch.allclose(output, reference_encoded_batch)
    assert torch.allclose(decoded_output, reference_decoded_batch)
