import torch
from huggingface_hub import PyTorchModelHubMixin
from jaxtyping import Bool, Float

from aion.codecs.modules.magvit import MagVitAE
from aion.codecs.modules.subsampler import SubsampledLinear
from aion.codecs.quantizers import FiniteScalarQuantizer, Quantizer
from aion.codecs.tokenizers.base import QuantizedCodec
from aion.codecs.preprocessing.image import ImagePadder, CenterCrop, RescaleToLegacySurvey, Clamp, RangeCompression


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

        # Preprocessing
        self.clamp = Clamp()
        self.center_crop = CenterCrop(crop_size=96)
        self.rescaler = RescaleToLegacySurvey()
        self.range_compressor = RangeCompression(
            div_factor=range_compression_factor, mult_factor=mult_factor
        )

        # Handle multi-survey projection
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
        
    def _get_survey(self, bands: list[str]) -> str:
        survey = bands[0].split("-")[0]
        return survey

    def _encode(
        self,
        x: Float[torch.Tensor, " b {self.n_bands} w h"],
        bands: list[str],
    ) -> Float[torch.Tensor, " b c1 w1 h1"]:
        
        # Preprocess the image
        x = self.center_crop(x)
        x = self.clamp(x)
        x = self.rescaler.forward(x, self._get_survey(bands))
        x = self.range_compressor.forward(x)

        # Handle multi-survey projection
        x, channel_mask = self.image_padder.forward(x, bands)
        x = self.subsample_in(x, channel_mask)

        # Encode the image
        h = self.encoder(x)
        h = self.pre_quant_proj(h)
        return h

    def _decode(
        self,
        z: Float[torch.Tensor, " b c1 w1 h1"],
        bands: list[str],
    ) -> Float[torch.Tensor, " b {self.n_bands} w h"]:
        # Decode the image
        h = self.post_quant_proj(z)
        dec = self.decoder(h)
        
        # Handle multi-survey projection
        channel_mask = torch.ones((z.shape[0], self.n_bands), device=z.device)
        dec = self.subsample_out(dec, channel_mask)

        # Postprocess the image
        dec = self.range_compressor.backward(dec)
        dec = self.image_padder.backward(dec, bands)
        dec = self.rescaler.backward(dec, self._get_survey(bands))
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