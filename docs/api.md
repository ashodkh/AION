# API Reference

This API reference covers the core components you'll actually use with AION-1, based on the working implementation.

## Core Model

### `aion.AION`

The main AION model class that provides multimodal astronomical analysis.

```python
from aion import AION

class AION(FM):
    """
    AION-1 multimodal astronomical foundation model.

    Inherits from FM (4M) architecture and adds astronomical-specific
    functionality for processing multiple data modalities.
    """

    @classmethod
    def from_pretrained(cls, model_name: str, **kwargs) -> 'AION':
        """
        Load a pre-trained AION model from HuggingFace Hub.

        Args:
            model_name: HuggingFace model identifier
                - 'polymathic-ai/aion-base': 300M parameter model

        Returns:
            AION model instance

        Example:
            >>> model = AION.from_pretrained('polymathic-ai/aion-base')
            >>> model = model.to('cuda').eval()
        """

    def forward(
        self,
        input_tokens: Dict[str, torch.Tensor],
        target_mask: Optional[Dict[str, torch.Tensor]] = None,
        num_encoder_tokens: int = 600,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.

        Args:
            input_tokens: Dictionary mapping modality token keys to token tensors
            target_mask: Dictionary specifying which tokens to predict
                Format: {"tok_z": torch.zeros(batch_size, num_target_tokens)}
            num_encoder_tokens: Number of tokens to use in encoder

        Returns:
            Dictionary mapping target keys to prediction logits

        Example:
            >>> predictions = model(
            ...     tokens,
            ...     target_mask={"tok_z": torch.zeros(32, 1)},
            ...     num_encoder_tokens=600
            ... )
            >>> redshift_probs = torch.softmax(predictions["tok_z"], dim=-1)
        """

    def encode(
        self,
        input_tokens: Dict[str, torch.Tensor],
        num_encoder_tokens: int = 600
    ) -> torch.Tensor:
        """
        Extract embeddings from input tokens.

        Args:
            input_tokens: Dictionary of tokenized modality data
            num_encoder_tokens: Number of tokens for encoder processing

        Returns:
            Encoder embeddings with shape [batch, seq_len, hidden_dim]

        Example:
            >>> embeddings = model.encode(tokens, num_encoder_tokens=600)
            >>> # Use embeddings for downstream tasks
            >>> pooled = embeddings.mean(dim=1)  # [batch, hidden_dim]
        """
```

## Codec Management

### `aion.codecs.CodecManager`

Manages automatic loading and application of modality-specific codecs.

```python
from aion.codecs import CodecManager

class CodecManager:
    """
    Central manager for encoding/decoding between modalities and tokens.
    """

    def __init__(self, device: str = 'cuda'):
        """
        Initialize codec manager.

        Args:
            device: Device to load codecs on ('cuda', 'cpu')

        Example:
            >>> codec_manager = CodecManager(device='cuda')
        """

    def encode(self, *modalities) -> Dict[str, torch.Tensor]:
        """
        Encode modalities into discrete tokens.

        Args:
            *modalities: Variable number of modality objects

        Returns:
            Dictionary mapping token keys to token tensors

        Example:
            >>> tokens = codec_manager.encode(image, spectrum, flux_g)
            >>> # Returns: {"tok_image": tensor(...), "tok_spectrum_sdss": tensor(...), "tok_flux_g": tensor(...)}
        """

    def decode(
        self,
        tokens: Dict[str, torch.Tensor],
        modality_class: type,
        **metadata
    ):
        """
        Decode tokens back to modality objects.

        Args:
            tokens: Dictionary of token tensors
            modality_class: Class of modality to decode (e.g., LegacySurveyImage)
            **metadata: Additional metadata required for reconstruction

        Returns:
            Reconstructed modality object

        Example:
            >>> reconstructed = codec_manager.decode(
            ...     tokens,
            ...     LegacySurveyImage,
            ...     bands=["DES-G", "DES-R", "DES-I", "DES-Z"]
            ... )
        """
```

## Modalities

AION-1 uses a typed modality system to ensure data compatibility and provenance tracking.

### Base Classes

```python
from aion.modalities import BaseModality

class BaseModality:
    """Base class for all astronomical modalities."""

    @property
    def token_key(self) -> str:
        """Unique identifier for this modality type in the model."""
```

### Image Modalities

```python
from aion.modalities import LegacySurveyImage, HSCImage

class LegacySurveyImage(BaseModality):
    """
    Legacy Survey multi-band image.

    Attributes:
        flux: Image tensor with shape [batch, 4, height, width] for g,r,i,z bands
        bands: List of band identifiers (e.g., ['DES-G', 'DES-R', 'DES-I', 'DES-Z'])
    """

    flux: torch.Tensor
    bands: List[str]

    @property
    def token_key(self) -> str:
        return "tok_image"

class HSCImage(BaseModality):
    """
    HSC multi-band image.

    Attributes:
        flux: Image tensor with shape [batch, 5, height, width] for g,r,i,z,y bands
        bands: List of band identifiers
    """

    flux: torch.Tensor
    bands: List[str]

    @property
    def token_key(self) -> str:
        return "tok_image"
```

### Spectrum Modalities

```python
from aion.modalities import DESISpectrum, SDSSSpectrum

class DESISpectrum(BaseModality):
    """
    DESI spectroscopic observation.

    Attributes:
        flux: Flux density array
        ivar: Inverse variance array
        mask: Boolean mask array
        wavelength: Wavelength array in Angstroms
    """

    flux: torch.Tensor
    ivar: torch.Tensor
    mask: torch.Tensor
    wavelength: torch.Tensor

    @property
    def token_key(self) -> str:
        return "tok_spectrum_desi"

class SDSSSpectrum(BaseModality):
    """SDSS spectroscopic observation."""

    @property
    def token_key(self) -> str:
        return "tok_spectrum_sdss"
```

### Scalar Modalities

```python
from aion.modalities import (
    LegacySurveyFluxG, LegacySurveyFluxR, LegacySurveyFluxI, LegacySurveyFluxZ,
    Z, GaiaParallax
)

class LegacySurveyFluxG(BaseModality):
    """Legacy Survey g-band flux measurement."""

    value: torch.Tensor

    @property
    def token_key(self) -> str:
        return "tok_flux_g"

class Z(BaseModality):
    """Spectroscopic redshift."""

    value: torch.Tensor

    @property
    def token_key(self) -> str:
        return "tok_z"
```

## Complete Usage Example

Here's a comprehensive example showing the full workflow:

```python
import torch
from aion import AION
from aion.codecs import CodecManager
from aion.modalities import (
    LegacySurveyImage, DESISpectrum,
    LegacySurveyFluxG, LegacySurveyFluxR, LegacySurveyFluxI, LegacySurveyFluxZ
)

# 1. Load model and codec manager
model = AION.from_pretrained('polymathic-ai/aion-base').to('cuda').eval()
codec_manager = CodecManager(device='cuda')

# 2. Prepare data
image = LegacySurveyImage(
    flux=torch.tensor(image_data, dtype=torch.float32),
    bands=['DES-G', 'DES-R', 'DES-I', 'DES-Z']
)

spectrum = DESISpectrum(
    flux=torch.tensor(flux_data),
    ivar=torch.tensor(ivar_data),
    mask=torch.tensor(mask_data, dtype=torch.bool),
    wavelength=torch.tensor(wavelength_data)
)

flux_g = LegacySurveyFluxG(value=torch.tensor([flux_g_value]))

# 3. Encode to tokens
tokens = codec_manager.encode(image, spectrum, flux_g)

# 4. Extract embeddings for downstream tasks
with torch.no_grad():
    embeddings = model.encode(tokens, num_encoder_tokens=600)
    pooled_embeddings = embeddings.mean(dim=1)  # [batch, hidden_dim]

# 5. Predict redshift
with torch.no_grad():
    predictions = model(
        tokens,
        target_mask={"tok_z": torch.zeros(1, 1)},
        num_encoder_tokens=600
    )
    redshift_probs = torch.softmax(predictions["tok_z"][0], dim=-1)

# 6. Decode tokens back to modalities
reconstructed_image = codec_manager.decode(
    tokens,
    LegacySurveyImage,
    bands=['DES-G', 'DES-R', 'DES-I', 'DES-Z']
)
```

## Model Variants

Currently available pre-trained models:

| Model | Parameters | HuggingFace ID |
|-------|------------|----------------|
| AION-Base | 300M | `polymathic-ai/aion-base` |

More model variants will be added as they become available.

## Common Patterns

### Similarity Search
```python
def compute_similarities(query_tokens, database_tokens, model):
    """Compute embedding similarities between query and database."""
    with torch.no_grad():
        query_emb = model.encode(query_tokens).mean(dim=1)
        db_embs = model.encode(database_tokens).mean(dim=1)

    from sklearn.metrics.pairwise import cosine_similarity
    return cosine_similarity(query_emb.cpu(), db_embs.cpu())
```

### Batch Processing
```python
def process_batch(batch_data, model, codec_manager):
    """Process a batch of astronomical objects."""
    batch_tokens = codec_manager.encode(*batch_data)

    with torch.no_grad():
        embeddings = model.encode(batch_tokens, num_encoder_tokens=600)

    return embeddings.mean(dim=1)  # Pooled embeddings
```

For more examples, see the [Usage Guide](usage.md) and [Tutorial Notebook](https://colab.research.google.com/github/PolymathicAI/AION/blob/main/notebooks/Tutorial.ipynb).
