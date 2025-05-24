"""Structured modality data types using Pydantic for type safety and validation."""

from typing import List, Union, ClassVar
from pydantic import BaseModel, Field, ConfigDict
from jaxtyping import Float, Bool
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


class Spectrum(Modality):
    """Spectrum modality data.

    Represents astronomical spectra with flux measurements, inverse variance, mask, and wavelength information.
    """

    flux: Float[Tensor, "batch length"] = Field(
        description="Array of flux measurements of the spectrum."
    )
    ivar: Float[Tensor, "batch length"] = Field(
        description="Array of inverse variance values for the spectrum."
    )
    mask: Bool[Tensor, "batch length"] = Field(
        description="Mask array indicating valid/invalid values in the spectrum."
    )
    wavelength: Float[Tensor, "batch length"] = Field(
        description="Array of wavelength values in Angstroms."
    )

    def __repr__(self) -> str:
        repr_str = f"Spectrum(flux_shape={list(self.flux.shape)}, wavelength_range=[{self.wavelength.min().item():.1f}, {self.wavelength.max().item():.1f}])"
        return repr_str


class ScalarModality(Modality):
    """Base class for scalar modality data.

    Represents a single scalar value per sample, typically used for
    flux measurements, shape parameters, or other single-valued properties.
    """

    name: ClassVar[str] = ""
    value: Float[Tensor, "..."] = Field(
        description="Scalar value for each sample in the batch."
    )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(shape={list(self.value.shape)})"


# Flux measurements in different bands
class FluxG(ScalarModality):
    """G-band flux measurement."""

    name: ClassVar[str] = "FLUX_G"


class FluxR(ScalarModality):
    """R-band flux measurement."""

    name: ClassVar[str] = "FLUX_R"


class FluxI(ScalarModality):
    """I-band flux measurement."""

    name: ClassVar[str] = "FLUX_I"


class FluxZ(ScalarModality):
    """Z-band flux measurement."""

    name: ClassVar[str] = "FLUX_Z"


class FluxW1(ScalarModality):
    """WISE W1-band flux measurement."""

    name: ClassVar[str] = "FLUX_W1"


class FluxW2(ScalarModality):
    """WISE W2-band flux measurement."""

    name: ClassVar[str] = "FLUX_W2"


class FluxW3(ScalarModality):
    """WISE W3-band flux measurement."""

    name: ClassVar[str] = "FLUX_W3"


class FluxW4(ScalarModality):
    """WISE W4-band flux measurement."""

    name: ClassVar[str] = "FLUX_W4"


# Shape parameters
class ShapeR(ScalarModality):
    """R-band shape measurement (e.g., half-light radius)."""

    name: ClassVar[str] = "SHAPE_R"


class ShapeE1(ScalarModality):
    """First ellipticity component."""

    name: ClassVar[str] = "SHAPE_E1"


class ShapeE2(ScalarModality):
    """Second ellipticity component."""

    name: ClassVar[str] = "SHAPE_E2"


# Other scalar properties
class EBV(ScalarModality):
    """E(B-V) extinction measurement."""

    name: ClassVar[str] = "EBV"


# Extinction values from HSC
class AG(ScalarModality):
    """HSC a_g extinction."""

    name: ClassVar[str] = "a_g"


class AR(ScalarModality):
    """HSC a_r extinction."""

    name: ClassVar[str] = "a_r"


class AI(ScalarModality):
    """HSC a_i extinction."""

    name: ClassVar[str] = "a_i"


class AZ(ScalarModality):
    """HSC a_z extinction."""

    name: ClassVar[str] = "a_z"


class AY(ScalarModality):
    """HSC a_y extinction."""

    name: ClassVar[str] = "a_y"


class MagG(ScalarModality):
    """HSC g-band cmodel magnitude."""

    name: ClassVar[str] = "g_cmodel_mag"


class MagR(ScalarModality):
    """HSC r-band cmodel magnitude."""

    name: ClassVar[str] = "r_cmodel_mag"


class MagI(ScalarModality):
    """HSC i-band cmodel magnitude."""

    name: ClassVar[str] = "i_cmodel_mag"


class MagZ(ScalarModality):
    """HSC z-band cmodel magnitude."""

    name: ClassVar[str] = "z_cmodel_mag"


class MagY(ScalarModality):
    """HSC y-band cmodel magnitude."""

    name: ClassVar[str] = "y_cmodel_mag"


class Shape11(ScalarModality):
    """HSC i-band SDSS shape 11 component."""

    name: ClassVar[str] = "i_sdssshape_shape11"


class Shape22(ScalarModality):
    """HSC i-band SDSS shape 22 component."""

    name: ClassVar[str] = "i_sdssshape_shape22"


class Shape12(ScalarModality):
    """HSC i-band SDSS shape 12 component."""

    name: ClassVar[str] = "i_sdssshape_shape12"


# Gaia modalities
class FluxGGaia(ScalarModality):
    """Gaia G-band mean flux."""

    name: ClassVar[str] = "phot_g_mean_flux"


class FluxBpGaia(ScalarModality):
    """Gaia BP-band mean flux."""

    name: ClassVar[str] = "phot_bp_mean_flux"


class FluxRpGaia(ScalarModality):
    """Gaia RP-band mean flux."""

    name: ClassVar[str] = "phot_rp_mean_flux"


class Parallax(ScalarModality):
    """Gaia parallax measurement."""

    name: ClassVar[str] = "parallax"


class Ra(ScalarModality):
    """Right ascension coordinate."""

    name: ClassVar[str] = "ra"


class Dec(ScalarModality):
    """Declination coordinate."""

    name: ClassVar[str] = "dec"


class XpBp(ScalarModality):
    """Gaia BP spectral coefficients."""

    name: ClassVar[str] = "bp_coefficients"


class XpRp(ScalarModality):
    """Gaia RP spectral coefficients."""

    name: ClassVar[str] = "rp_coefficients"


ScalarModalities = [
    FluxG,
    FluxR,
    FluxI,
    FluxZ,
    FluxW1,
    FluxW2,
    FluxW3,
    FluxW4,
    ShapeR,
    ShapeE1,
    ShapeE2,
    EBV,
    AG,
    AR,
    AI,
    AZ,
    AY,
    MagG,
    MagR,
    MagI,
    MagZ,
    MagY,
    Shape11,
    Shape22,
    Shape12,
    FluxGGaia,
    FluxBpGaia,
    FluxRpGaia,
    Parallax,
    Ra,
    Dec,
    XpBp,
    XpRp,
]

# Convenience type for any modality data
ModalityType = Union[Image, Spectrum, ScalarModality]
