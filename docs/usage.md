# AION-1 Usage Guide

This comprehensive guide demonstrates how to use AION-1 for various astronomical analysis tasks, based on the actual working implementation.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Loading and Preparing Data](#loading-and-preparing-data)
3. [Basic Workflows](#basic-workflows)
4. [Embedding Extraction](#embedding-extraction)
5. [Similarity Search](#similarity-search)
6. [Property Prediction](#property-prediction)
7. [Performance Tips](#performance-tips)

## Quick Start

Here's how to get started with AION-1 in just a few lines:

```python
import torch
import numpy as np
from aion import AION
from aion.codecs import CodecManager
from aion.modalities import LegacySurveyImage

# 1. Load model and codec manager
model = AION.from_pretrained('polymathic-ai/aion-base').to('cuda').eval()
codec_manager = CodecManager(device='cuda')

# 2. Prepare your astronomical data
image = LegacySurveyImage(
    flux=torch.tensor(your_image_data, dtype=torch.float32),  # Shape: [batch, 4, height, width]
    bands=['DES-G', 'DES-R', 'DES-I', 'DES-Z']
)

# 3. Encode to tokens
tokens = codec_manager.encode(image)

# 4. Extract embeddings for downstream analysis
with torch.no_grad():
    embeddings = model.encode(tokens, num_encoder_tokens=600)
    # Shape: [batch, sequence_length, 768]

# 5. Predict redshift distribution
with torch.no_grad():
    predictions = model(
        tokens,
        target_mask={"tok_z": torch.zeros(batch_size, 1)},
        num_encoder_tokens=600
    )
    redshift_logits = predictions["tok_z"]
    redshift_probs = torch.softmax(redshift_logits, dim=-1)
```

## Loading and Preparing Data

### Working with Images

AION-1 expects multi-band astronomical images with specific formatting:

```python
import torch
from astropy.io import fits
from aion.modalities import LegacySurveyImage, HSCImage

# Example 1: Legacy Survey (4-band: g,r,i,z)
def load_legacy_survey_image(fits_path):
    """Load and format Legacy Survey FITS data."""
    with fits.open(fits_path) as hdul:
        # Assuming bands are in separate extensions
        flux_data = np.array([hdul[i].data for i in range(1, 5)])  # 4 bands

    image = LegacySurveyImage(
        flux=torch.tensor(flux_data, dtype=torch.float32),
        bands=['DES-G', 'DES-R', 'DES-I', 'DES-Z']
    )
    return image

# Example 2: HSC (5-band: g,r,i,z,y)
def load_hsc_image(flux_array):
    """Load HSC 5-band image data."""
    image = HSCImage(
        flux=torch.tensor(flux_array, dtype=torch.float32),
        bands=['HSC-G', 'HSC-R', 'HSC-I', 'HSC-Z', 'HSC-Y']
    )
    return image

# Note: AION-1 automatically crops/pads images to 96x96 pixels
```

### Working with Spectra

Load and prepare spectroscopic observations:

```python
from aion.modalities import DESISpectrum, SDSSSpectrum

def load_desi_spectrum(flux, ivar, mask, wavelength):
    """Load DESI spectrum data."""
    spectrum = DESISpectrum(
        flux=torch.tensor(flux, dtype=torch.float32),
        ivar=torch.tensor(ivar, dtype=torch.float32),
        mask=torch.tensor(mask, dtype=torch.bool),
        wavelength=torch.tensor(wavelength, dtype=torch.float32)
    )
    return spectrum

def load_sdss_spectrum_from_fits(fits_path):
    """Load SDSS spectrum from FITS file."""
    with fits.open(fits_path) as hdul:
        data = hdul[1].data
        wavelength = 10**data['loglam']  # Convert from log wavelength
        flux = data['flux']
        ivar = data['ivar']

    # Create mask for bad pixels
    mask = (ivar > 0) & (flux > 0)

    spectrum = SDSSSpectrum(
        flux=torch.tensor(flux, dtype=torch.float32),
        ivar=torch.tensor(ivar, dtype=torch.float32),
        mask=torch.tensor(mask, dtype=torch.bool),
        wavelength=torch.tensor(wavelength, dtype=torch.float32)
    )
    return spectrum
```

### Working with Photometric Data

Prepare scalar measurements like fluxes and shape parameters:

```python
from aion.modalities import (
    LegacySurveyFluxG, LegacySurveyFluxR, LegacySurveyFluxI, LegacySurveyFluxZ,
    Z, GaiaParallax
)

def create_photometry_modalities(catalog_data):
    """Create modalities from catalog measurements."""
    modalities = []

    # Photometric fluxes
    if 'flux_g' in catalog_data:
        modalities.append(LegacySurveyFluxG(
            value=torch.tensor(catalog_data['flux_g'], dtype=torch.float32)
        ))

    if 'flux_r' in catalog_data:
        modalities.append(LegacySurveyFluxR(
            value=torch.tensor(catalog_data['flux_r'], dtype=torch.float32)
        ))

    # Redshift
    if 'redshift' in catalog_data:
        modalities.append(Z(
            value=torch.tensor(catalog_data['redshift'], dtype=torch.float32)
        ))

    return modalities
```

## Basic Workflows

### Workflow 1: Embedding Extraction

Extract learned representations for downstream machine learning:

```python
def extract_galaxy_embeddings(data_list, model, codec_manager):
    """Extract embeddings from a list of galaxy observations."""
    all_embeddings = []

    # Process in batches for efficiency
    batch_size = 32
    for i in range(0, len(data_list), batch_size):
        batch = data_list[i:i + batch_size]

        # Encode all modalities in the batch
        batch_tokens = codec_manager.encode(*batch)

        # Extract embeddings
        with torch.no_grad():
            embeddings = model.encode(batch_tokens, num_encoder_tokens=600)
            # Pool over sequence dimension
            pooled = embeddings.mean(dim=1)  # [batch, 768]

        all_embeddings.append(pooled.cpu().numpy())

    return np.vstack(all_embeddings)

# Usage example
galaxy_embeddings = extract_galaxy_embeddings(
    [image1, image2, image3, ...],
    model,
    codec_manager
)
```

### Workflow 2: Redshift Prediction

Predict redshift distributions from various input modalities:

```python
def predict_redshift_distribution(inputs, model, codec_manager):
    """Predict redshift probability distribution."""
    # Encode inputs
    tokens = codec_manager.encode(*inputs)

    # Predict redshift
    with torch.no_grad():
        predictions = model(
            tokens,
            target_mask={"tok_z": torch.zeros(len(inputs), 1)},
            num_encoder_tokens=600
        )

    # Convert to probabilities
    redshift_logits = predictions["tok_z"]
    redshift_probs = torch.softmax(redshift_logits, dim=-1)

    return redshift_probs

# Example: Predict from photometry
redshift_dist = predict_redshift_distribution(
    [flux_g, flux_r, flux_i, flux_z],
    model,
    codec_manager
)
```

### Workflow 3: Reconstruction

Reconstruct modalities through the encode-decode process:

```python
def reconstruct_modality(original_modality, model, codec_manager, modality_class, **metadata):
    """Reconstruct a modality through encode-decode cycle."""
    # Encode original
    tokens = codec_manager.encode(original_modality)

    # Decode back
    reconstructed = codec_manager.decode(
        tokens,
        modality_class,
        **metadata
    )

    return reconstructed

# Example: Reconstruct image
reconstructed_image = reconstruct_modality(
    original_image,
    model,
    codec_manager,
    LegacySurveyImage,
    bands=['DES-G', 'DES-R', 'DES-I', 'DES-Z']
)
```

## Embedding Extraction

### Basic Embedding Extraction

```python
def get_embeddings(modalities, model, codec_manager, pooling='mean'):
    """Extract embeddings with different pooling strategies."""
    tokens = codec_manager.encode(*modalities)

    with torch.no_grad():
        embeddings = model.encode(tokens, num_encoder_tokens=600)

    # Apply pooling
    if pooling == 'mean':
        return embeddings.mean(dim=1)
    elif pooling == 'max':
        return embeddings.max(dim=1)[0]
    elif pooling == 'cls':
        return embeddings[:, 0]  # First token
    else:
        return embeddings  # Return full sequence

# Usage
embeddings = get_embeddings([image, spectrum], model, codec_manager)
```

### Multi-Modal Embeddings

Combine embeddings from different modalities:

```python
def get_multimodal_embeddings(image, spectrum, photometry, model, codec_manager):
    """Extract embeddings from multiple modality types."""

    # Get embeddings from each modality type
    image_tokens = codec_manager.encode(image)
    spectrum_tokens = codec_manager.encode(spectrum)
    photo_tokens = codec_manager.encode(*photometry)

    embeddings = {}

    with torch.no_grad():
        # Image embeddings
        img_emb = model.encode(image_tokens, num_encoder_tokens=300)
        embeddings['image'] = img_emb.mean(dim=1)

        # Spectrum embeddings
        spec_emb = model.encode(spectrum_tokens, num_encoder_tokens=300)
        embeddings['spectrum'] = spec_emb.mean(dim=1)

        # Combined embeddings
        all_tokens = {**image_tokens, **spectrum_tokens, **photo_tokens}
        combined_emb = model.encode(all_tokens, num_encoder_tokens=900)
        embeddings['combined'] = combined_emb.mean(dim=1)

    return embeddings
```

## Similarity Search

Implement similarity search using AION embeddings:

```python
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

class AIONSimilaritySearch:
    def __init__(self, model, codec_manager):
        self.model = model
        self.codec_manager = codec_manager
        self.database_embeddings = []
        self.database_objects = []
        self.index = None

    def add_objects(self, objects):
        """Add objects to the search database."""
        for obj in objects:
            # Extract embedding
            tokens = self.codec_manager.encode(*obj['modalities'])
            with torch.no_grad():
                emb = self.model.encode(tokens, num_encoder_tokens=600)
                emb = emb.mean(dim=1).cpu().numpy()

            self.database_embeddings.append(emb)
            self.database_objects.append(obj)

        # Build search index
        if self.database_embeddings:
            embeddings_matrix = np.vstack(self.database_embeddings)
            self.index = NearestNeighbors(n_neighbors=10, metric='cosine')
            self.index.fit(embeddings_matrix)

    def search(self, query_modalities, k=5):
        """Search for similar objects."""
        # Get query embedding
        tokens = self.codec_manager.encode(*query_modalities)
        with torch.no_grad():
            query_emb = self.model.encode(tokens, num_encoder_tokens=600)
            query_emb = query_emb.mean(dim=1).cpu().numpy()

        # Find nearest neighbors
        distances, indices = self.index.kneighbors(query_emb, n_neighbors=k)

        results = []
        for i, idx in enumerate(indices[0]):
            results.append({
                'object': self.database_objects[idx],
                'similarity': 1 - distances[0][i],  # Convert distance to similarity
                'rank': i + 1
            })

        return results

# Usage example
searcher = AIONSimilaritySearch(model, codec_manager)

# Add objects to database
database_objects = [
    {'modalities': [image1, spectrum1], 'metadata': {'id': 'galaxy_1'}},
    {'modalities': [image2, spectrum2], 'metadata': {'id': 'galaxy_2'}},
    # ... more objects
]
searcher.add_objects(database_objects)

# Search for similar objects
query_galaxy = [query_image, query_spectrum]
similar_objects = searcher.search(query_galaxy, k=10)

print(f"Found {len(similar_objects)} similar objects:")
for result in similar_objects:
    print(f"Rank {result['rank']}: {result['object']['metadata']['id']} "
          f"(similarity: {result['similarity']:.3f})")
```

## Property Prediction

Use AION embeddings for various prediction tasks:

### Redshift Estimation with k-NN

```python
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

def train_redshift_predictor(galaxies_with_redshifts, model, codec_manager):
    """Train a k-NN regressor for redshift prediction."""

    # Extract embeddings and targets
    embeddings = []
    redshifts = []

    for galaxy in galaxies_with_redshifts:
        tokens = codec_manager.encode(*galaxy['modalities'])
        with torch.no_grad():
            emb = model.encode(tokens, num_encoder_tokens=600)
            emb = emb.mean(dim=1).cpu().numpy()

        embeddings.append(emb[0])  # Remove batch dimension
        redshifts.append(galaxy['redshift'])

    X = np.array(embeddings)
    y = np.array(redshifts)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train k-NN regressor
    knn = KNeighborsRegressor(n_neighbors=5)
    knn.fit(X_train, y_train)

    # Evaluate
    y_pred = knn.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Redshift prediction - MAE: {mae:.4f}, R²: {r2:.4f}")

    return knn

def predict_redshift(new_galaxy, trained_model, model, codec_manager):
    """Predict redshift for a new galaxy."""
    tokens = codec_manager.encode(*new_galaxy)
    with torch.no_grad():
        emb = model.encode(tokens, num_encoder_tokens=600)
        emb = emb.mean(dim=1).cpu().numpy()

    predicted_z = trained_model.predict(emb)[0]
    return predicted_z
```

### Stellar Mass Prediction

```python
from sklearn.ensemble import RandomForestRegressor

def train_stellar_mass_predictor(galaxies_with_masses, model, codec_manager):
    """Train predictor for stellar mass estimation."""

    # Similar to redshift prediction but for stellar mass
    embeddings = []
    masses = []

    for galaxy in galaxies_with_masses:
        tokens = codec_manager.encode(*galaxy['modalities'])
        with torch.no_grad():
            emb = model.encode(tokens, num_encoder_tokens=600)
            emb = emb.mean(dim=1).cpu().numpy()

        embeddings.append(emb[0])
        masses.append(np.log10(galaxy['stellar_mass']))  # Log stellar mass

    X = np.array(embeddings)
    y = np.array(masses)

    # Train Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)

    return rf
```

## Performance Tips

### Batch Processing

Process multiple objects efficiently:

```python
def process_batch_efficiently(object_list, model, codec_manager, batch_size=32):
    """Process objects in batches for better GPU utilization."""
    results = []

    for i in range(0, len(object_list), batch_size):
        batch = object_list[i:i + batch_size]

        # Group by modality type for efficient encoding
        images = [obj for obj in batch if 'image' in obj]
        spectra = [obj for obj in batch if 'spectrum' in obj]

        batch_results = []

        with torch.no_grad():
            # Process images
            if images:
                image_batch = [obj['image'] for obj in images]
                tokens = codec_manager.encode(*image_batch)
                embeddings = model.encode(tokens, num_encoder_tokens=600)
                batch_results.extend(embeddings.mean(dim=1).cpu().numpy())

            # Process spectra
            if spectra:
                spectrum_batch = [obj['spectrum'] for obj in spectra]
                tokens = codec_manager.encode(*spectrum_batch)
                embeddings = model.encode(tokens, num_encoder_tokens=300)
                batch_results.extend(embeddings.mean(dim=1).cpu().numpy())

        results.extend(batch_results)

    return results
```

### Memory Management

Handle large datasets with limited GPU memory:

```python
def process_large_dataset(dataset, model, codec_manager, max_batch_size=16):
    """Process large datasets with automatic memory management."""
    import gc

    current_batch_size = max_batch_size
    results = []

    i = 0
    while i < len(dataset):
        try:
            batch = dataset[i:i + current_batch_size]

            # Process batch
            batch_tokens = codec_manager.encode(*batch)
            with torch.no_grad():
                embeddings = model.encode(batch_tokens, num_encoder_tokens=600)
                results.append(embeddings.mean(dim=1).cpu())

            i += current_batch_size

        except torch.cuda.OutOfMemoryError:
            # Clear memory and reduce batch size
            torch.cuda.empty_cache()
            gc.collect()
            current_batch_size = max(1, current_batch_size // 2)
            print(f"Reduced batch size to {current_batch_size}")

            if current_batch_size == 0:
                raise RuntimeError("Cannot process even single example")

    return torch.cat(results, dim=0)
```

### Using Mixed Precision

Speed up inference with automatic mixed precision:

```python
def extract_embeddings_amp(modalities, model, codec_manager):
    """Extract embeddings using automatic mixed precision."""
    from torch.cuda.amp import autocast

    tokens = codec_manager.encode(*modalities)

    with torch.no_grad():
        with autocast():
            embeddings = model.encode(tokens, num_encoder_tokens=600)

    return embeddings.float()  # Convert back to float32
```

## Best Practices

1. **Always use `.eval()` mode** for inference to disable dropout and batch norm updates
2. **Use `torch.no_grad()`** to disable gradient computation and save memory
3. **Process in batches** when possible for better GPU utilization
4. **Pool embeddings appropriately** - mean pooling works well for most tasks
5. **Use consistent device placement** - ensure all tensors are on the same device
6. **Clear GPU cache** periodically when processing large datasets

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce batch size or use gradient checkpointing
2. **Slow processing**: Ensure data is on GPU and use batch processing
3. **Shape mismatches**: Check that tensor dimensions match expected format
4. **Device errors**: Ensure model, data, and codec_manager are on same device

### Debug Mode

```python
def debug_tokens(tokens, codec_manager):
    """Debug token shapes and contents."""
    print("Token summary:")
    for key, tensor in tokens.items():
        print(f"  {key}: shape={tensor.shape}, dtype={tensor.dtype}, device={tensor.device}")
        print(f"    range: [{tensor.min().item():.2f}, {tensor.max().item():.2f}]")
```

For more advanced examples and the latest updates, see the [Tutorial Notebook](https://colab.research.google.com/github/PolymathicAI/AION/blob/main/notebooks/Tutorial.ipynb).
