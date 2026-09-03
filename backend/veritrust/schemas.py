"""Request and response models. Only used at the HTTP boundary."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Base64Request(BaseModel):
    image: str = Field(description="Raw base64 or a data URL.")
    heatmap: bool = Field(default=True, description="Compute the Grad-CAM overlay.")


class SignalOut(BaseModel):
    name: str
    p_ai: float
    weight: float
    kind: str
    detail: str = ""
    clamped: bool = False
    counted: bool = True


class FaceOut(BaseModel):
    index: int
    x: int
    y: int
    w: int
    h: int
    score: float
    p_ai: float


class MediaOut(BaseModel):
    """One shape for both modalities, with the fields the other cannot fill left as None.

    Two models here would mean AnalyzeResponse could not declare a single type for `image`, and a
    union would make the frontend test which arm it got before reading anything. Every field is
    optional for that reason, and mime is the one thing both always know.

    Anything the engine puts in image_info and this class does not declare is dropped by FastAPI
    with a 200 and no warning, exactly as for the top level response keys.
    """

    mime: str
    width: int | None = None
    height: int | None = None
    original_width: int | None = None
    original_height: int | None = None
    megapixels: float | None = None
    downscaled: bool | None = None
    duration: float | None = None
    original_duration: float | None = None
    sample_rate: int | None = None
    original_sample_rate: int | None = None
    original_channels: int | None = None
    downmixed: bool | None = None
    truncated: bool | None = None
    decoder: str | None = None
    windows_scored: int | None = None


class AnalyzeResponse(BaseModel):
    verdict: str = Field(description="likely_authentic, uncertain or likely_ai_generated.")
    score_ai: float = Field(description="Fused score in 0 to 1. An ordering unless calibrated is true.")
    confidence: float
    confidence_meaning: str
    calibrated: bool
    signals: list[SignalOut]
    notes: list[str]
    provenance: dict
    faces: list[FaceOut]
    heatmap: str | None
    image: MediaOut
    timing_ms: dict
    overridden_by: str | None
    escalated_by: str | None = None
    logit_spread: float = Field(
        default=0.0,
        description="Spread between the highest and lowest model reading, in logits, after clamping.",
    )
    spread_exceeds_limit: bool = Field(
        default=False,
        description="True when that spread passed logit_spread_limit. On a fused verdict it also "
        "means confidence was reduced by it. On a verdict taken from the face pathway alone there "
        "is nothing to reduce, and it means the disagreement was overruled rather than averaged.",
    )
    errors: list[dict]


class HealthResponse(BaseModel):
    status: str
    version: str
    models_loaded: int
    device: str
