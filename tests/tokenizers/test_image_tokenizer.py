import pytest
import torch

from aion.codecs.tokenizers.image import MagViTAEImageCodec


@pytest.mark.parametrize("n_bands", [3, 10])
@pytest.mark.parametrize("embedding_dim", [5, 10])
@pytest.mark.parametrize("multisurvey_projection_dims", [12, 24])
@pytest.mark.parametrize("hidden_dims", [8, 16])
def test_magvit_image_tokenizer(
    n_bands, embedding_dim, multisurvey_projection_dims, hidden_dims
):
    tokenizer = MagViTAEImageCodec(
        n_bands=n_bands,
        quantizer_levels=[1] * embedding_dim,
        hidden_dims=hidden_dims,
        multisurvey_projection_dims=multisurvey_projection_dims,
        n_compressions=2,
        num_consecutive=4,
        embedding_dim=embedding_dim,
        range_compression_factor=0.01,
        mult_factor=10,
    )
    batch_size = 4
    random_input = torch.randn(batch_size, n_bands, 96, 96)
    channel_mask = torch.ones(batch_size, n_bands)
    encoded = tokenizer.encode(random_input, channel_mask)
    assert encoded.shape == (batch_size, 24, 24)
    decoded = tokenizer.decode(encoded)
    assert decoded.shape == random_input.shape


def test_hf_previous_predictions(data_dir):
    codec = MagViTAEImageCodec.from_pretrained("polymathic-ai/aion-image-codec")

    input_batch = torch.load(
        data_dir / "image_codec_input_batch.pt", weights_only=False
    )
    reference_encoded_output = torch.load(
        data_dir / "image_codec_encoded_batch.pt", weights_only=False
    )
    reference_decoded_output = torch.load(
        data_dir / "image_codec_decoded_batch.pt", weights_only=False
    )

    with torch.no_grad():
        encoded_output = codec.encode(
            input_batch["image"]["array"], input_batch["image"]["channel_mask"]
        )
        decoded_output = codec.decode(encoded_output)

    assert encoded_output.shape == reference_encoded_output.shape
    assert torch.allclose(encoded_output, reference_encoded_output), (
        f"Encoded output is not close to reference output: {torch.abs(encoded_output - reference_encoded_output).sum()}, "
        f"{torch.where(torch.abs(encoded_output - reference_encoded_output) > 1e-5)}"
    )
    assert decoded_output.shape == reference_decoded_output.shape
    assert torch.allclose(decoded_output, reference_decoded_output), (
        f"Decoded output is not close to reference output: {(torch.abs(decoded_output - reference_decoded_output) > 1e-5).sum()}, "
        f"{torch.where(torch.abs(decoded_output - reference_decoded_output) > 1e-5)}"
    )
