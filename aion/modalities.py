"""Structured modality data types using Pydantic for type safety and validation."""

from typing import List, Union
from pydantic import BaseModel, Field, ConfigDict
from jaxtyping import Float
from torch import Tensor


class Modality(BaseModel):
    """Base class for all modality data types."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")


class Image(Modality):
    """Image modality data.

    Represents astronomical images with flux measurements, and band information.
    """

    flux: Float[Tensor, "batch num_bands height width"] = Field(
        description="Array of flux measurements of the image."
    )
    bands: List[str] = Field(
        description="List of band names, indicating the wavelength range of each channel in flux (e.g., ['DES-G', 'DES-R']). Its length must match num_bands."
    )

    def __repr__(self) -> str:
        repr_str = f"Image(flux_shape={list(self.flux.shape)}, bands={self.bands})"
        return repr_str


# Convenience type for any modality data
ModalityType = Union[Image,]
