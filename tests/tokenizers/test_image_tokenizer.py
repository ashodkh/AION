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

    input_batch = torch.load(data_dir / "image_codec_test_input.pt")
    reference_output = torch.load(data_dir / "image_codec_test_output.pt")

    with torch.no_grad():
        output = codec.encode(
            input_batch["image"]["array"], input_batch["image"]["channel_mask"]
        )
    assert torch.allclose(output, reference_output)
