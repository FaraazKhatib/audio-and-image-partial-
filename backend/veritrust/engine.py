"""Analysis orchestration. Owns the pipeline, knows nothing about HTTP.

Order of work for an image: decode, read provenance, run the synthetic ensemble on the whole image,
then if faces are present crop and score each one, then fuse. Provenance runs first because a signed
manifest makes the rest advisory.

Modality is decided by the magic bytes, not by a second endpoint or a client supplied hint. One
upload field that accepts whatever the reader has is the point, and a client is not a reliable
witness to what it is sending. sniff_audio_mime only matches audio container headers, so an image
cannot fall down the audio path: the WAV test requires the WAVE form type after RIFF, which is what
separates it from WebP, and the bare MP3 frame sync excludes the FF D8 that starts every JPEG.

Both modalities return the same Analysis, and therefore the same response shape. That is deliberate.
The alternative is a second schema and a second renderer, and the reader is being told the same
thing either way: which detectors read what, how much they disagreed, and which band the fused score
landed in. Fields that only one modality can fill are left at their empty value rather than dropped,
so the frontend never has to test which kind of result it received before reading a key.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


from . import provenance
from .config import (
    FACE,
    SYNTHETIC,
    AUDIO,
    CALIBRATION_PATH,
    VERDICT_AUTHENTIC,
    VERDICT_UNCERTAIN,
    Settings,
    settings as default_settings,
)
from .detectors.registry import Registry
from .explain import grad_cam
from .faces import FaceDetector
from .fusion import Calibration, Signal, fuse, logit, sigmoid, clamp_probability
from .preprocessing import crop_with_margin, decode_image
from .audio import sniff_audio_mime, decode_audio, make_windows, quantile

PROVENANCE = "provenance"


@dataclass
class Analysis:
    verdict: str
    score_ai: float
    confidence: float
    calibrated: bool
    signals: list[dict]
    notes: list[str]
    provenance: dict
    faces: list[dict]
    heatmap: str | None
    image_info: dict
    timing_ms: dict
    overridden_by: str | None
    escalated_by: str | None = None
    logit_spread: float = 0.0
    spread_exceeds_limit: bool = False
    errors: list[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "verdict": self.verdict,
            "score_ai": round(self.score_ai, 4),
            "confidence": round(self.confidence, 1),
            "confidence_meaning": (
                "Margin outside the uncertainty band, not a probability that the verdict is "
                "correct."
            ),
            "calibrated": self.calibrated,
            "signals": self.signals,
            "notes": self.notes,
            "provenance": self.provenance,
            "faces": self.faces,
            "heatmap": self.heatmap,
            "image": self.image_info,
            "timing_ms": self.timing_ms,
            "overridden_by": self.overridden_by,
            "escalated_by": self.escalated_by,
            "logit_spread": round(self.logit_spread, 3),
            "spread_exceeds_limit": self.spread_exceeds_limit,
            "errors": self.errors,
        }


class Engine:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or default_settings
        self.registry = Registry(self.settings)
                self.calibration = Calibration()

    def load(self) -> None:
        self.registry.load_all()
        self.face_detector.load()
        self.calibration = Calibration.load(CALIBRATION_PATH)

    def status(self) -> dict:
        return {
            **self.registry.status(),
            "face_backend": self.face_detector.backend,
            "face_note": self.face_detector.note,
            "calibrated": self.calibration.fitted,
            "calibration_source": self.calibration.source or None,
            "thresholds": {
                "authentic_max": self.settings.thresholds.authentic_max,
                "ai_min": self.settings.thresholds.ai_min,
            },
        }

    def analyze(self, data: bytes, want_heatmap: bool = False) -> Analysis:
        audio_mime = sniff_audio_mime(data)
        if audio_mime is None:
            raise ValueError("Only audio files are supported.")
        return self._analyze_audio(data)

    def _analyze_audio(self, data: bytes) -> Analysis:
        timing: dict[str, float] = {}
        errors: list[dict] = []

        started = time.perf_counter()
        decoded = decode_audio(data, self.settings)
        timing["decode"] = (time.perf_counter() - started) * 1000

        windows, notes = make_windows(decoded, self.settings)

        signals: list[Signal] = []
        started = time.perf_counter()
        for detector in self.registry.audio:
            window_scores: list[float] = []
            failure: str | None = None
            for window in windows:
                result = detector.predict(window.samples, decoded.sample_rate)
                if result.usable:
                    window_scores.append(result.p_fake)
                elif failure is None:
                    # One entry per detector, not one per window. A checkpoint that fails fails the
                    # same way on all 24 windows, and 24 identical rows in the error list reads as
                    # 24 separate faults.
                    failure = result.error

            if failure is not None:
                scored = f" It scored {len(window_scores)} of {len(windows)} window(s)."
                errors.append(
                    {
                        "detector": detector.key,
                        "error": f"{failure}{scored if window_scores else ''}",
                    }
                )
            if not window_scores:
                continue

            score = quantile(window_scores, self.settings.audio_quantile)
            signals.append(
                Signal(
                    name=detector.key,
                    p_fake=score,
                    weight=detector.spec.weight,
                    kind=AUDIO,
                    detail=(
                        f"{self.settings.audio_quantile:.0%} quantile of {len(window_scores)} "
                        f"window(s), highest {max(window_scores):.2f}. A quantile rather than the "
                        f"max, which would climb with clip length on its own."
                    ),
                )
            )
        timing["audio"] = (time.perf_counter() - started) * 1000

        started = time.perf_counter()
        result = fuse(signals, self.settings.thresholds, self.settings.fusion, self.calibration)
        timing["fusion"] = (time.perf_counter() - started) * 1000

        notes.extend(result.notes)

        if not self.settings.enable_audio:
            notes.append(
                "Audio analysis is switched off by configuration (VT_ENABLE_AUDIO), so no model "
                "read this file. Nothing about the verdict below is based on the audio itself."
            )
        elif not self.registry.audio:
            notes.append(
                "No audio detection model loaded, so nothing read this file. Check "
                "/api/v1/models and run scripts/verify_models.py."
            )

        if decoded.truncated:
            notes.append(
                f"This recording is {decoded.original_duration:.0f} s and was cut to the first "
                f"{self.settings.max_audio_seconds:.0f} s for analysis. Anything after that point "
                f"was not examined."
            )
        if decoded.resampled:
            notes.append(
                f"Audio was resampled from {decoded.original_sample_rate} Hz to "
                f"{decoded.sample_rate} Hz, which is what these checkpoints were trained on. "
                f"Resampling is anti-aliased, but it still attenuates the high frequency detail "
                f"they read, so a marginal reading may be weaker than it would be natively."
            )
        if decoded.downmixed:
            notes.append(
                f"{decoded.original_channels} channels were averaged to mono. A spoof present in "
                f"only one channel is quieter after mixing than it was in the original."
            )

        # Metadata is not read on this path. Returning the empty ProvenanceResult keeps one response
        # shape across both modalities, so the frontend reads the same keys either way, and saying
        # so in the notes stops the empty result from being read as "checked, found nothing".
        notes.append(
            "Container metadata was not examined. Provenance reading is implemented for images "
            "only, so this verdict rests on the model readings alone."
        )

        return Analysis(
            verdict=result.verdict,
            score_ai=result.score,
            confidence=result.confidence,
            calibrated=result.calibrated,
            signals=result.signals,
            notes=notes,
            provenance=provenance.ProvenanceResult().as_dict(),
            faces=[],
            heatmap=None,
            image_info={
                "mime": decoded.mime,
                "duration": decoded.duration,
                "original_duration": decoded.original_duration,
                "sample_rate": decoded.sample_rate,
                "original_sample_rate": decoded.original_sample_rate,
                "original_channels": decoded.original_channels,
                "downmixed": decoded.downmixed,
                "truncated": decoded.truncated,
                "decoder": decoded.decoder,
                "windows_scored": len(windows),
            },
            timing_ms={k: round(v, 1) for k, v in timing.items()},
            overridden_by=result.overridden_by,
            escalated_by=result.escalated_by,
            logit_spread=result.logit_spread,
            spread_exceeds_limit=result.spread_exceeds_limit,
            errors=errors,
        )

    def _analyze_image(self, data: bytes, want_heatmap: bool = True) -> Analysis:
        timing: dict[str, float] = {}
        errors: list[dict] = []

        started = time.perf_counter()
        decoded = decode_image(data, self.settings)
        timing["decode"] = (time.perf_counter() - started) * 1000

        started = time.perf_counter()
        prov = provenance.inspect(decoded.raw)
        timing["provenance"] = (time.perf_counter() - started) * 1000

        signals: list[Signal] = []

        if prov.p_fake is not None:
            signals.append(
                Signal(
                    name=PROVENANCE,
                    p_fake=prov.p_fake,
                    weight=1.0,
                    kind=PROVENANCE,
                    detail="; ".join(prov.evidence[:3]),
                    override=prov.override,
                )
            )

        started = time.perf_counter()
        best_synthetic = None
        best_score = -1.0
        for detector in self.registry.synthetic:
            result = detector.predict(decoded.image)
            if not result.usable:
                errors.append({"detector": result.key, "error": result.error})
                continue
            signals.append(
                Signal(
                    name=result.key,
                    p_fake=result.p_fake,
                    weight=detector.spec.weight,
                    kind=SYNTHETIC,
                    detail=result.detail,
                )
            )
            if result.p_fake > best_score:
                best_score = result.p_fake
                best_synthetic = detector
        timing["synthetic"] = (time.perf_counter() - started) * 1000

        started = time.perf_counter()
        face_payload: list[dict] = []
        detection = self.face_detector.detect(decoded.image)
        if detection.faces and self.registry.face:
            per_face_scores: list[float] = []
            for index, face in enumerate(detection.faces):
                try:
                    crop = crop_with_margin(decoded.image, face.box, self.settings.face_margin)
                except Exception as exc:
                    errors.append({"detector": "face_crop", "error": str(exc)})
                    continue
                scores = []
                for detector in self.registry.face:
                    result = detector.predict(crop)
                    if result.usable:
                        scores.append(result.p_fake)
                    else:
                        errors.append({"detector": result.key, "error": result.error})
                if not scores:
                    continue
                import logging
                log = logging.getLogger("veritrust")
                log.info("Face scores for crop: %s", scores)
                # Use logit averaging (like fusion.py) to properly balance confident models.
                # Linear averaging suppresses a confident detection (e.g. 0.99) if other models are mildly uncertain.
                clamp = self.settings.fusion.signal_clamp
                face_z = sum(logit(clamp_probability(s, clamp)) for s in scores) / len(scores)
                face_score = sigmoid(face_z)
                per_face_scores.append(face_score)
                entry = face.as_dict()
                entry.update({"index": index, "p_ai": round(face_score, 4)})
                face_payload.append(entry)

            if per_face_scores:
                signals.append(
                    Signal(
                        name="face_pathway",
                        p_fake=max(per_face_scores),
                        weight=self.registry.face[0].spec.weight,
                        kind=FACE,
                        detail=(
                            f"Highest of {len(per_face_scores)} face crop(s), detected with "
                            f"{detection.backend}. Max is used because one swapped face is enough."
                        ),
                    )
                )
        timing["faces"] = (time.perf_counter() - started) * 1000

        started = time.perf_counter()
        result = fuse(signals, self.settings.thresholds, self.settings.fusion, self.calibration)
        timing["fusion"] = (time.perf_counter() - started) * 1000

        # A current positive can be a useful review flag, but the configured image checkpoints
        # have neither local calibration nor a measured error rate on the generators people now
        # upload.  Their shared "real" reading must not be rendered as affirmative evidence of a
        # real photograph.  Provenance overrides are explicit evidence and intentionally bypass
        # this rule.  The raw score remains intact for calibration and diagnostics.
        if (
            not result.calibrated
            and not self.settings.allow_uncalibrated_authentic
            and result.overridden_by is None
            and result.verdict == VERDICT_AUTHENTIC
        ):
            result.verdict = VERDICT_UNCERTAIN
            result.confidence = 0.0
            result.notes.append(
                "The models read this as authentic, but they are uncalibrated on this deployment. "
                "Held at uncertain rather than affirming a real photograph. Set "
                "VT_ALLOW_UNCALIBRATED_AUTHENTIC=true only after a labelled local evaluation."
            )

        heatmap = None
        if want_heatmap and self.settings.enable_heatmap and best_synthetic is not None:
            started = time.perf_counter()
            heatmap = grad_cam(best_synthetic, decoded.image)
            timing["heatmap"] = (time.perf_counter() - started) * 1000

        notes = list(result.notes)
        if not detection.available and detection.note:
            notes.append(detection.note)
        elif detection.available and not detection.faces:
            notes.append(
                "No faces found, so the verdict rests on whole image generation cues only."
            )
        elif detection.faces and not self.registry.face:
            notes.append(
                f"{len(detection.faces)} face(s) found with {detection.backend}, but no face "
                f"model is loaded, so the face replacement pathway did not run. This image is "
                f"being judged on whole image cues alone. Check /api/v1/models."
            )
        if decoded.downscaled:
            notes.append(
                f"Image was downscaled to {decoded.width}x{decoded.height} for analysis. "
                f"Resampling weakens the high frequency cues these detectors rely on."
            )
        if not self.registry.any_ready:
            notes.append(
                "No detection model loaded. Check /api/v1/models and run "
                "scripts/verify_models.py."
            )

        return Analysis(
            verdict=result.verdict,
            score_ai=result.score,
            confidence=result.confidence,
            calibrated=result.calibrated,
            signals=result.signals,
            notes=notes,
            provenance=prov.as_dict(),
            faces=face_payload,
            heatmap=heatmap,
            image_info={
                "width": decoded.width,
                "height": decoded.height,
                "original_width": decoded.original_width,
                "original_height": decoded.original_height,
                "megapixels": decoded.megapixels,
                "mime": decoded.mime,
                "downscaled": decoded.downscaled,
            },
            timing_ms={k: round(v, 1) for k, v in timing.items()},
            overridden_by=result.overridden_by,
            escalated_by=result.escalated_by,
            logit_spread=result.logit_spread,
            spread_exceeds_limit=result.spread_exceeds_limit,
            errors=errors,
        )
