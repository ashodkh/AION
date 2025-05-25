"""Test the CodecManager class."""

import pytest
import torch

from aion.codecs.manager import CodecManager
from aion.modalities import (
    LegacySurveyFluxG,
    LegacySurveyShapeE1,
)


class TestCodecManager:
    """Test suite for CodecManager."""

    @pytest.fixture
    def manager(self):
        """Create a CodecManager instance."""
        return CodecManager(device="cpu")

    def test_codec_caching(self, manager):
        """Test that codecs are properly cached and reused."""
        # Create two modalities that use the same codec type
        flux_g1 = LegacySurveyFluxG(value=torch.randn(4, 1))
        flux_g2 = LegacySurveyFluxG(value=torch.randn(4, 1))

        # Encode both
        manager.encode(flux_g1)
        manager.encode(flux_g2)

        # Check that only one codec was loaded
        assert len(manager._modality_to_codec) == 1
        assert LegacySurveyFluxG in manager._modality_to_codec

        # Check that the same codec instance is used
        codec1 = manager._get_codec_for_modality(LegacySurveyFluxG)
        codec2 = manager._get_codec_for_modality(LegacySurveyFluxG)
        assert codec1 is codec2

    @pytest.mark.parametrize("batch_size", [1, 4, 16])
    def test_different_batch_sizes(self, manager, batch_size):
        """Test that CodecManager handles different batch sizes correctly."""
        # Create modalities with different batch sizes
        flux_g = LegacySurveyFluxG(value=torch.randn(batch_size, 1))
        shape_e1 = LegacySurveyShapeE1(value=torch.randn(batch_size, 1))

        # Encode
        tokens = manager.encode(flux_g, shape_e1)

        # Check batch sizes
        assert tokens["tok_flux_g"].shape[0] == batch_size
        assert tokens["tok_shape_e1"].shape[0] == batch_size

        # Decode and verify
        decoded_flux = manager.decode(tokens, LegacySurveyFluxG)
        decoded_shape = manager.decode(tokens, LegacySurveyShapeE1)

        assert decoded_flux.value.shape[0] == batch_size
        assert decoded_shape.value.shape[0] == batch_size
