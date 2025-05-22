import torch
from huggingface_hub import PyTorchModelHubMixin
from jaxtyping import Float

from aion.codecs.modules.magvit import MagVitAE
from aion.codecs.modules.subsampler import SubsampledLinear
from aion.codecs.quantizers import FiniteScalarQuantizer, Quantizer
from aion.codecs.tokenizers.base import QuantizedCodec
from aion.codecs.utils import range_compression, reverse_range_compression
from aion.codecs.preprocessing.image import ImagePadder


class AutoencoderImageCodec(QuantizedCodec):
    """Meta-class for autoencoder codecs for images, does not actually contain a network."""

    def __init__(
        self,
        n_bands: int,
        quantizer: Quantizer,
        encoder: torch.nn.Module,
        decoder: torch.nn.Module,
        hidden_dims: int = 64,
        embedding_dim: int = 5,
        multisurvey_projection_dims: int = 54,
        range_compression_factor: float = 0.01,
        mult_factor: float = 10.0,
    ):
        super().__init__(quantizer)
        self.range_compression_factor = range_compression_factor
        self.mult_factor = mult_factor
        self.n_bands = n_bands
        self.encoder = encoder
        self.decoder = decoder

        # Handles multi-survey projections
        self.image_padder = ImagePadder()
        self.subsample_in = SubsampledLinear(
            dim_in=n_bands, dim_out=multisurvey_projection_dims, subsample_in=True
        )
        self.subsample_out = SubsampledLinear(
            dim_in=multisurvey_projection_dims, dim_out=n_bands, subsample_in=False
        )
        # Go down to size of levels
        self.pre_quant_proj = torch.nn.Conv2d(
            hidden_dims, embedding_dim, kernel_size=1, stride=1, padding=0
        )

        # Go back to the original size
        self.post_quant_proj = torch.nn.Conv2d(
            embedding_dim, hidden_dims, kernel_size=1, stride=1, padding=0
        )

    @property
    def modality(self) -> str:
        return "image"

    def _range_compress(self, x):
        x = range_compression(x, self.range_compression_factor)
        x = x * self.mult_factor
        return x

    def _reverse_range_compress(self, x):
        x = x / self.mult_factor
        x = reverse_range_compression(x, self.range_compression_factor)
        return x

    def _encode(
        self,
        batch: Float[torch.Tensor, " b c *input_shape"],
        bands: list[str],
    ) -> Float[torch.Tensor, " b c1 w1 h1"]:
        # handle multisurvey padding
        image, channel_mask = self.image_padder.forward(batch, bands)
        x = self._range_compress(image)
        x = self.subsample_in(image, channel_mask)
        h = self.encoder(x)
        h = self.pre_quant_proj(h)
        return h

    def _decode(
        self,
        z: Float[torch.Tensor, " b c1 w1 h1"],
        bands: list[str],
    ) -> Float[torch.Tensor, " b c *input_shape"]:
        # handle multisurvey padding
        h = self.post_quant_proj(z)
        dec = self.decoder(h)
        batch_size = z.shape[0]
        channel_mask = torch.ones((batch_size, self.n_bands), device=z.device)
        dec = self.subsample_out(dec, channel_mask)
        dec = self._reverse_range_compress(dec)
        return dec


class ImageCodec(AutoencoderImageCodec, PyTorchModelHubMixin):
    def __init__(
        self,
        n_bands: int,
        quantizer_levels: list[int],
        hidden_dims: int = 512,
        multisurvey_projection_dims: int = 54,
        n_compressions: int = 2,  # Number of compressions in the network
        num_consecutive: int = 4,  # Number of consecutive residual layers per compression
        embedding_dim: int = 5,
        range_compression_factor: float = 0.01,
        mult_factor: float = 10.0,
    ):
        """
        MagViT Autoencoder for images.

        Args:
            n_bands: Number of bands in the input images.
            quantizer: Quantizer to use.
            hidden_dims: Number of hidden dimensions in the network.
            n_compressions: Number of compressions in the network.
            num_consecutive: Number of consecutive residual layers per compression.
            embedding_dim: Dimension of the latent space.
            range_compression_factor: Range compression factor.
            mult_factor: Multiplication factor.
        """
        # Get MagViT architecture
        model = MagVitAE(
            n_bands=multisurvey_projection_dims,
            hidden_dims=hidden_dims,
            n_compressions=n_compressions,
            num_consecutive=num_consecutive,
        )
        quantizer = FiniteScalarQuantizer(levels=quantizer_levels)
        super().__init__(
            n_bands,
            quantizer,
            model.encode,
            model.decode,
            hidden_dims,
            embedding_dim,
            multisurvey_projection_dims,
            range_compression_factor,
            mult_factor,
        )
        self.model = model
