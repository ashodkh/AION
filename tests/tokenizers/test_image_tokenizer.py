import pytest
import torch

from aion.codecs.tokenizers import ImageCodec


@pytest.mark.parametrize("embedding_dim", [5, 10])
@pytest.mark.parametrize("multisurvey_projection_dims", [12, 24])
@pytest.mark.parametrize("hidden_dims", [8, 16])
def test_magvit_image_tokenizer(
    embedding_dim, multisurvey_projection_dims, hidden_dims
):
    n_bands = 4
    tokenizer = ImageCodec(
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
    batch = {
        "image": {
            "flux": torch.randn(batch_size, n_bands, 96, 96),
            "bands": ["DES_G", "DES_R", "DES_I", "DES_Z"],
        }
    }
    encoded = tokenizer.encode(batch)
    assert encoded.shape == (batch_size, 24, 24)
    decoded = tokenizer.decode(encoded, bands=["DES_G", "DES_R", "DES_I", "DES_Z"])
    random_input = batch["image"]["flux"]
    assert decoded.shape == random_input.shape


def test_hf_previous_predictions(data_dir):
    codec = ImageCodec.from_pretrained("polymathic-ai/aion-image-codec")

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
        encoded_output = codec.encode(input_batch)
        decoded_output = codec.decode(
            encoded_output, bands=["DES_G", "DES_R", "DES_I", "DES_Z"]
        )

    assert encoded_output.shape == reference_encoded_output.shape
    assert torch.allclose(encoded_output, reference_encoded_output)
    assert decoded_output.shape == reference_decoded_output.shape
    assert torch.allclose(
        decoded_output, reference_decoded_output, rtol=1e-3, atol=1e-4
    )
