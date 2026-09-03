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

from ..config import FAKE_LABEL_TOKENS, REAL_LABEL_TOKENS, ModelSpec
import json
import re
from .base import Detector, DetectorResult, LoadError

MIN_PREFIX = 3
"""Shortest token allowed to match a vocabulary entry by prefix rather than exactly.

Two characters is too short to be evidence. "ai" as a substring appears inside "painting",
"chair" and "brain", so a label like "real_painting" would resolve as generated.
"""


def _label_tokens(label: str) -> list[str]:
    """Split a class label into lowercase tokens, treating camelCase boundaries as separators.

    Punctuation and underscores are the obvious separators. camelCase has to count too, because
    Hemgg/Deepfake-audio-detection labels its classes AIVoice and HumanVoice: without the split
    those are single tokens, "ai" is too short to match by prefix and too different to match
    exactly, and the checkpoint is refused as unresolvable. The alternative was setting fake_index
    to 0 on that spec, which is the index position assumption this project removed once already.

    Two boundaries are needed. The lower-to-upper one splits HumanVoice. The upper-to-upper-lower
    one splits AIVoice, where three capitals run together and the break belongs before the last of
    them. Verified against every checkpoint currently declared: no image label tokenises differently
    under this than under plain punctuation splitting.
    """
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", str(label))
    spaced = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", " ", spaced)
    return [t for t in re.split(r"[^a-z0-9]+", spaced.strip().lower()) if t]


def _matches(tokens: list[str], vocabulary: tuple[str, ...]) -> str | None:
    """Return the vocabulary entry a label token corresponds to, or None.

    Matching is exact, or by prefix in either direction once both sides are long enough. The
    outward direction catches inflections such as "fakes"; the inward direction catches the
    truncations checkpoints actually ship, for example Ateeqq/ai-vs-human-image-detector labelling
    its real class "hum". Substring matching anywhere in the token is deliberately not used.
    """
    for token in tokens:
        for word in vocabulary:
            if token == word:
                return word
            if len(token) >= MIN_PREFIX and len(word) >= MIN_PREFIX:
                if token.startswith(word) or word.startswith(token):
                    return word
    return None


def resolve_fake_indices(id2label: dict[int, str]) -> tuple[list[int], str]:
    """Return indices whose label denotes generated content, plus a human readable mapping.

    Handles more than two classes by summing every generated-ish class, which covers
    checkpoints that split fake into per generator classes.

    A label matching both vocabularies is refused rather than resolved by which side was checked
    first, since "which list did I search first" is not evidence about the class.
    """
    fake: list[int] = []
    real: list[int] = []
    unmatched: list[str] = []

    for idx, raw in id2label.items():
        tokens = _label_tokens(raw)
        as_fake = _matches(tokens, FAKE_LABEL_TOKENS)
        as_real = _matches(tokens, REAL_LABEL_TOKENS)

        if as_fake and as_real:
            raise LoadError(
                f"Label {raw!r} reads as both {as_fake!r} and {as_real!r}, so it is ambiguous. "
                f"Full mapping {dict(id2label)}."
            )
        if as_fake:
            fake.append(int(idx))
        elif as_real:
            real.append(int(idx))
        else:
            unmatched.append(f"{idx}={raw}")

    if not fake or not real:
        detail = f" Unrecognised: {', '.join(unmatched)}." if unmatched else ""
        raise LoadError(
            f"Could not tell which class means generated from labels {dict(id2label)}.{detail} "
            f"Set fake_index on the ModelSpec after verifying the order against known samples."
        )

    mapping = ", ".join(f"{i}={id2label[i]}" for i in sorted(id2label))
    return fake, mapping


def load_processor(repo: str):
    """Load a checkpoint's image processor, preferring the fast implementation.

    transformers routes the fast path through torchvision, so a missing torchvision raises
    ImportError for every checkpoint rather than degrading. The slow processor is PIL based and
    produces equivalent tensors, just slower, which beats refusing to load at all. Installing
    torchvision is still the right fix; this only stops one absent optional backend from taking
    the whole ensemble down.
    """
    from transformers import AutoImageProcessor

    try:
        return AutoImageProcessor.from_pretrained(repo)
    except ImportError:
        return AutoImageProcessor.from_pretrained(repo, use_fast=False)


def describe_checkpoint(repo: str, head: str = "image classification head") -> str:
    """Report what a local checkpoint claims to be, for use in failure messages.

    A private checkpoint that is not an image classifier fails inside from_pretrained with an error
    that rarely names the actual architecture. Reading config.json directly turns "could not load"
    into something actionable, and it costs nothing because the file is small and local.

    head is what the caller expected to find, so the audio wrapper can reuse this rather than keep
    its own near identical copy. Nothing else about the report differs by modality.
    """
    config = Path(repo) / "config.json"
    if not config.is_file():
        return ""
    try:
        payload = json.loads(config.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ""

    architectures = payload.get("architectures") or []
    model_type = payload.get("model_type") or "unknown"
    labels = payload.get("id2label") or {}
    described = ", ".join(str(a) for a in architectures) if architectures else "none declared"
    return (
        f" The checkpoint declares model_type={model_type}, architectures=[{described}], "
        f"{len(labels)} label(s). If that is not an {head}, it needs its own "
        f"detector wrapper rather than this one."
    )


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
