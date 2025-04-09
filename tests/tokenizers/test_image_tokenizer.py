import pytest
import torch

from aion.codecs.tokenizers.image import MagViTAEImageCodec
from aion.codecs.quantizers import FiniteScaleQuantizer


@pytest.mark.parametrize("n_bands", [3, 10])
@pytest.mark.parametrize("embedding_dim", [5, 10])
@pytest.mark.parametrize("multisurvey_projection_dims", [12, 24])
@pytest.mark.parametrize("hidden_dims", [8, 16])
def test_magvit_image_tokenizer(
    n_bands, embedding_dim, multisurvey_projection_dims, hidden_dims
):
    tokenizer = MagViTAEImageCodec(
        n_bands=n_bands,
        quantizer=FiniteScaleQuantizer(levels=[1] * embedding_dim),
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
    encoded = tokenizer.encode(random_input)
    assert encoded.shape == (batch_size, 24, 24)
    decoded = tokenizer.decode(encoded)
    assert decoded.shape == random_input.shape
