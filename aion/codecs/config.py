from dataclasses import dataclass
from typing import TypeVar

from aion.codecs.catalog import CatalogCodec
from aion.codecs.image import ImageCodec
from aion.codecs.scalar import (
    GridScalarCodec,
    LogScalarCodec,
    MultiScalarCodec,
    ScalarCodec,
)
from aion.codecs.scalar_field import ScalarFieldCodec
from aion.codecs.spectrum import SpectrumCodec
from aion.modalities import (
    HSCAG,
    HSCAI,
    HSCAR,
    HSCAY,
    HSCAZ,
    Dec,
    GaiaFluxBp,
    GaiaFluxG,
    GaiaFluxRp,
    GaiaParallax,
    GaiaXpBp,
    GaiaXpRp,
    HSCMagG,
    HSCMagI,
    HSCMagR,
    HSCMagY,
    HSCMagZ,
    HSCShape11,
    HSCShape12,
    HSCShape22,
    Image,
    LegacySurveyCatalog,
    LegacySurveyEBV,
    LegacySurveyFluxG,
    LegacySurveyFluxI,
    LegacySurveyFluxR,
    LegacySurveyFluxW1,
    LegacySurveyFluxW2,
    LegacySurveyFluxW3,
    LegacySurveyFluxW4,
    LegacySurveyFluxZ,
    LegacySurveySegmentationMap,
    LegacySurveyShapeE1,
    LegacySurveyShapeE2,
    LegacySurveyShapeR,
    Ra,
    Spectrum,
    Z,
)

CodecType = TypeVar(
    "CodecModel",
    bound=type[
        CatalogCodec
        | GridScalarCodec
        | ImageCodec
        | LogScalarCodec
        | MultiScalarCodec
        | ScalarCodec
        | ScalarFieldCodec
        | SpectrumCodec
    ],
)


@dataclass
class CodecHFConfig:
    """Codec configuration for AION."""

    codec_class: CodecType
    repo_id: str


MODALITY_CODEC_MAPPING = {
    Image: ImageCodec,
    Spectrum: SpectrumCodec,
    LegacySurveyCatalog: CatalogCodec,
    LegacySurveySegmentationMap: ScalarFieldCodec,
    LegacySurveyFluxG: LogScalarCodec,
    LegacySurveyFluxR: LogScalarCodec,
    LegacySurveyFluxI: LogScalarCodec,
    LegacySurveyFluxZ: LogScalarCodec,
    LegacySurveyFluxW1: LogScalarCodec,
    LegacySurveyFluxW2: LogScalarCodec,
    LegacySurveyFluxW3: LogScalarCodec,
    LegacySurveyFluxW4: LogScalarCodec,
    LegacySurveyShapeR: LogScalarCodec,
    GaiaFluxG: LogScalarCodec,
    GaiaFluxBp: LogScalarCodec,
    GaiaFluxRp: LogScalarCodec,
    GaiaParallax: LogScalarCodec,
    LegacySurveyShapeE1: ScalarCodec,
    LegacySurveyShapeE2: ScalarCodec,
    LegacySurveyEBV: ScalarCodec,
    HSCMagG: ScalarCodec,
    HSCMagR: ScalarCodec,
    HSCMagI: ScalarCodec,
    HSCMagZ: ScalarCodec,
    HSCMagY: ScalarCodec,
    HSCShape11: ScalarCodec,
    HSCShape22: ScalarCodec,
    HSCShape12: ScalarCodec,
    HSCAG: ScalarCodec,
    HSCAR: ScalarCodec,
    HSCAI: ScalarCodec,
    HSCAZ: ScalarCodec,
    HSCAY: ScalarCodec,
    Ra: ScalarCodec,
    Dec: ScalarCodec,
    GaiaXpBp: MultiScalarCodec,
    GaiaXpRp: MultiScalarCodec,
    Z: GridScalarCodec,
}

HF_REPO_ID = "polymathic-ai/aion-base"
