"""Hugging Face audio classification wrapper with safe label resolution.

This exists as a separate class rather than a branch inside HFImageClassifier because the two
differ in every part that touches the checkpoint: AutoFeatureExtractor instead of
AutoImageProcessor, AutoModelForAudioClassification instead of the image head, and a waveform plus
an explicit sampling rate instead of a PIL image. What they must not differ in is label resolution,
so resolve_fake_indices is imported rather than reimplemented. The audio checkpoints disagree with
each other about class order, one shipping 0=fake 1=real and another 0=real 1=fake, which is
exactly the silent inversion that name based resolution exists to prevent.

The sampling rate is passed through rather than assumed. Feature extractors resample or refuse
based on what they are told, and telling one 16 kHz about audio that is actually 44.1 kHz produces
a valid looking score from a signal shifted an octave and a half, which no amount of downstream
fusion can detect. audio.py is what guarantees the array actually arrives at the declared rate.

describe() returns the same keys as HFImageClassifier.describe() on purpose. /api/v1/models is one
endpoint and the frontend reads one shape from it, so a pathway that reported readiness and errors
under different names would be rendered as blank rows rather than as a degraded ensemble.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from ..config import ModelSpec
from .base import Detector, DetectorResult, LoadError
from .hf_image import describe_checkpoint, resolve_fake_indices

AUDIO_HEAD = "audio classification head"


class HFAudioClassifier(Detector):
    def __init__(self, spec: ModelSpec, device: str, dtype: str = "auto"):
        self.spec = spec
        self.key = spec.key
        self.kind = spec.kind
        self.device = device
        self.dtype = dtype
        self.repo_used: str | None = None
        self.label_mapping: str = ""
        self.error: str | None = None
        self._model = None
        self._processor = None
        self._fake_indices: list[int] = []

    @property
    def ready(self) -> bool:
        return self._model is not None and self.error is None

    def _torch_dtype(self):
        import torch

        if self.dtype == "auto":
            return torch.float16 if self.device == "cuda" else torch.float32
        return getattr(torch, self.dtype, torch.float32)

    def load(self) -> None:
        # Ahead of the torch import for the same reason as in the image wrapper: there is nothing
        # to load, and the failure stays reachable on a machine with no torch installed.
        if self.spec.is_local and not Path(self.spec.repo).is_dir():
            self.error = f"{self.spec.repo} is not a directory"
            raise LoadError(
                f"No candidate loaded for {self.key}. {self.error}. A local checkpoint needs a "
                f"directory containing config.json, the weights, and preprocessor_config.json."
            )

        import torch  # noqa: F401  imported here so a missing install fails as a load error
        from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

        candidates = (self.spec.repo, *self.spec.fallbacks)
        failures: list[str] = []

        for repo in candidates:
            try:
                processor = AutoFeatureExtractor.from_pretrained(repo)
                model = AutoModelForAudioClassification.from_pretrained(
                    repo, torch_dtype=self._torch_dtype()
                )
                id2label = {int(k): v for k, v in model.config.id2label.items()}

                if self.spec.fake_index is not None:
                    fake_indices = [self.spec.fake_index]
                    mapping = f"forced fake_index={self.spec.fake_index} over {id2label}"
                else:
                    fake_indices, mapping = resolve_fake_indices(id2label)

                model.eval()
                model.to(self.device)

                self._processor = processor
                self._model = model
                self._fake_indices = fake_indices
                self.label_mapping = mapping
                self.repo_used = repo
                self.error = None
                return
            except Exception as exc:
                failures.append(f"{repo}: {type(exc).__name__}: {exc}")

        self.error = " | ".join(failures)
        hint = describe_checkpoint(self.spec.repo, AUDIO_HEAD) if self.spec.is_local else ""
        raise LoadError(f"No candidate loaded for {self.key}. {self.error}{hint}")

    @property
    def expected_sample_rate(self) -> int | None:
        """What the loaded feature extractor says it wants, or None if it does not say.

        Reported rather than enforced. A mismatch between this and the rate audio.py resamples to
        is a configuration problem worth surfacing, but refusing to run would take out a pathway
        over something the extractor itself will usually handle.
        """
        rate = getattr(self._processor, "sampling_rate", None)
        return int(rate) if isinstance(rate, (int, float)) else None

    def predict(self, samples: np.ndarray, sampling_rate: int = 16000) -> DetectorResult:
        if not self.ready:
            return DetectorResult(
                key=self.key,
                kind=self.kind,
                p_fake=None,
                error=self.error or "not loaded",
            )

        import torch

        started = time.perf_counter()
        try:
            inputs = self._processor(samples, sampling_rate=sampling_rate, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            if self.device == "cuda" and self._torch_dtype() == torch.float16:
                inputs = {
                    k: (v.half() if v.dtype == torch.float32 else v) for k, v in inputs.items()
                }

            with torch.inference_mode():
                logits = self._model(**inputs).logits.float()
            probs = torch.softmax(logits, dim=-1)[0]
            p_fake = float(sum(probs[i].item() for i in self._fake_indices))
        except Exception as exc:
            return DetectorResult(
                key=self.key,
                kind=self.kind,
                p_fake=None,
                latency_ms=(time.perf_counter() - started) * 1000,
                error=f"{type(exc).__name__}: {exc}",
            )

        return DetectorResult(
            key=self.key,
            kind=self.kind,
            p_fake=min(max(p_fake, 0.0), 1.0),
            latency_ms=(time.perf_counter() - started) * 1000,
            detail=self.spec.notes,
        )

    def describe(self) -> dict:
        return {
            "key": self.key,
            "kind": self.kind,
            "configured_repo": self.spec.repo,
            "loaded_repo": self.repo_used,
            "ready": self.ready,
            "label_mapping": self.label_mapping,
            "weight": self.spec.weight,
            "notes": self.spec.notes,
            "error": self.error,
        }
