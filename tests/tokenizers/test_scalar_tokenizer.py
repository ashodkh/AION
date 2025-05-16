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

    input_batch = torch.load(data_dir / f"{modality}_input.pt")
    output_batch = torch.load(data_dir / f"{modality}_output.pt")

    with torch.no_grad():
        output = codec.encode({modality: input_batch})
    assert torch.allclose(output, output_batch)


@pytest.mark.parametrize("modality", ["SHAPE_E1", "SHAPE_E2", "EBV"])
def test_reservoir_tokenizer(data_dir, modality):
    codec = ScalarReservoirCodec.from_pretrained(
        f"polymathic-ai/aion-scalar-{modality.lower().replace('_', '-')}-codec"
    )
    codec.eval()
    codec.load_state_dict(torch.load(data_dir / f"{modality}_codec.pt"))

    input_batch = torch.load(data_dir / f"{modality}_input.pt")
    output_batch = torch.load(data_dir / f"{modality}_output.pt")

    with torch.no_grad():
        output = codec.encode({modality: input_batch})
    assert torch.allclose(output, output_batch)
