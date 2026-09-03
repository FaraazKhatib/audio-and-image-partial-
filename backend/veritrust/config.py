"""Central configuration. Stdlib only so it stays importable without torch or pydantic."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
REPO_DIR = BASE_DIR.parent
CALIBRATION_PATH = BASE_DIR / "calibration.json"
LOCAL_MODELS_PATH = Path(os.getenv("VT_LOCAL_MODELS") or (BASE_DIR / "models.local.json"))


def _env_str(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ[name])
    except (KeyError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ[name])
    except (KeyError, ValueError):
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


SYNTHETIC = "synthetic"
FACE = "face"
AUDIO = "audio"

MODEL_KINDS = (SYNTHETIC, FACE, AUDIO)
"""Kinds that are statistical estimates from a checkpoint, as opposed to read evidence.

fusion.py uses this to decide what may escalate a verdict and what counts toward the disagreement
spread. Provenance is deliberately absent: it overrides fusion outright when it has evidence, so
folding it into a spread between estimates would be comparing two different kinds of claim.
"""

VERDICT_AUTHENTIC = "likely_authentic"
VERDICT_UNCERTAIN = "uncertain"
VERDICT_AI = "likely_ai_generated"


@dataclass(frozen=True)
class ModelSpec:
    """One member of the detector ensemble.

    repo is a Hugging Face model id, loaded through AutoModelForImageClassification for the
    synthetic and face pathways and through AutoModelForAudioClassification for the audio one.
    Which class is used follows from `kind`, so a checkpoint declared under the wrong kind fails
    to load rather than being coerced into the wrong wrapper.
    fallbacks are tried in order if repo fails to resolve, so a single dead or renamed
    repo degrades the ensemble instead of breaking startup.

    weight is a prior, not a measured quantity. Run eval/calibrate.py against a labelled
    set to replace these with fitted values. Until then the fusion output is uncalibrated
    and every API response says so.
    """

    key: str
    repo: str
    kind: str
    weight: float = 1.0
    fallbacks: tuple[str, ...] = ()
    gradcam_target: str | None = None
    notes: str = ""
    is_local: bool = False
    """True when `repo` is a filesystem path rather than a Hub id.

    from_pretrained accepts either, so this does not change loading. It exists so failures can say
    "no such directory" instead of reporting a network lookup for a path, and so the prefetch script
    knows there is nothing to download.
    """
    fake_index: int | None = None
    """Escape hatch for checkpoints whose labels are generic, for example LABEL_0 and LABEL_1.

    Leave as None to force name based resolution. A model with unresolvable labels is marked
    unavailable rather than guessed at, because a silently inverted class order produces
    confident nonsense. Set this only after confirming the order against known samples.
    """


SYNTHETIC_MODELS: tuple[ModelSpec, ...] = (
    ModelSpec(
        key='vit_v2_deepfake',
        repo='dima806/deepfake_vs_real_image_detection',
        kind=SYNTHETIC,
        weight=1.0,
        fallbacks=(),
        gradcam_target='vit.layernorm',
        notes=(
            'ViT real-vs-generated detector. Its model card warns that its training data is about '
            'three years old; treat scores on current generators as uncalibrated until evaluated locally.'
        ),
    ),
)

FACE_MODELS: tuple[ModelSpec, ...] = (
    ModelSpec(
        key='efficientnet_ffpp',
        repo='Xicor9/efficientnet-b0-ffpp-c23',
        kind=FACE,
        weight=1.0,
        fallbacks=(),
        notes=(
            'FaceForensics++ C23 face-crop detector. It targets face manipulation rather than '
            'whole-image generation and is not calibrated for this deployment.'
        ),
    ),
)

# Seven checkpoints spanning three architecture families, on the same reasoning as the image
# ensemble: wav2vec2 fine-tunes, WavLM fine-tunes and an Audio Spectrogram Transformer read
# different things, so their mistakes are less likely to coincide than seven wav2vec2 variants would
# be. Every one was checked against the Hub API for existence, an id2label that resolves by name, an
# audio-classification pipeline tag and a preprocessor_config.json, because a community audio repo
# missing that last file fails inside AutoFeatureExtractor at load time.
#
# Note the class orders below disagree with each other. deepfake_wav2vec2 is 0=fake, 1=real while
# every other member is 0=real, 1=fake. That is not a hypothetical: it was read off the published
# config.json of each repo. Nothing here may assume index position; see resolve_fake_indices and
# keep fake_index unset.
#
# Every weight is 1.0 apart from ast_asvspoof. There is nothing measured to justify any other
# spread, and a guessed weight is a claim about relative accuracy that no one here has earned. The
# one exception is argued in that spec's own notes and is about architectural diversity rather than
# about accuracy. eval/calibrate.py is what replaces all of these with fitted coefficients.
#
# The authors' own reported figures are not repeated here. They were measured on the authors' own
# splits, mostly ASVspoof, and quoting them next to this code would read as this system's accuracy.
# Nothing about this pathway has been measured. See eval/ and calibration.json.
AUDIO_MODELS: tuple[ModelSpec, ...] = (
    ModelSpec(
        key="deepfake_wav2vec2",
        repo="MelodyMachine/Deepfake-audio-detection-V2",
        kind=AUDIO,
        weight=1.0,
        fallbacks=("motheecreator/Deepfake-audio-detection",),
        notes=(
            "wav2vec2 fine-tune, labels 0=fake 1=real, the one member whose class order is "
            "reversed relative to the rest. Its fallback is the checkpoint it was fine-tuned from, "
            "so they are one lineage and belong in one slot rather than two."
        ),
    ),
    ModelSpec(
        key="xlsr_deepfake",
        repo="Vansh180/deepfake-audio-wav2vec2",
        kind=AUDIO,
        weight=1.0,
        fallbacks=("Gustking/wav2vec2-large-xlsr-deepfake-audio-classification",),
        notes=(
            "XLS-R 53 fine-tune, labels 0=real 1=fake, resolved by name. Its fallback is a larger "
            "XLS-R trained on the same task, so an unavailable primary degrades to the same family "
            "rather than to a different architecture."
        ),
    ),
    ModelSpec(
        key="wavlm_itw",
        repo="abhishtagatya/wavlm-base-960h-itw-deepfake",
        kind=AUDIO,
        weight=1.0,
        notes=(
            "WavLM base, labels 0=bona-fide 1=spoof. Trained on in-the-wild audio rather than a "
            "studio spoofing corpus, which is closer to what a user actually uploads."
        ),
    ),
    ModelSpec(
        key="ast_asvspoof",
        repo="MattyB95/AST-ASVspoof5-Synthetic-Voice-Detection",
        kind=AUDIO,
        weight=1.2,
        notes=(
            "Audio Spectrogram Transformer, labels 0=Bonafide 1=Spoof. Reads a mel spectrogram as a "
            "patch grid, so it fails differently from every waveform model here. Weighted above the "
            "rest for that reason: an ensemble of one architecture agrees with itself, and this is "
            "the only member that can disagree for a structural reason rather than a data one."
        ),
    ),
    ModelSpec(
        key="deepfake_voice_detector",
        repo="garystafford/wav2vec2-deepfake-voice-detector",
        kind=AUDIO,
        weight=1.0,
        notes=(
            "wav2vec2 fine-tune aimed at TTS and voice cloning rather than at replay or splicing, "
            "labels 0=real 1=fake."
        ),
    ),
    ModelSpec(
        key="wav2vec2_base_itw",
        repo="abhishtagatya/wav2vec2-base-960h-itw-deepfake",
        kind=AUDIO,
        weight=1.0,
        notes=(
            "wav2vec2 base counterpart to wavlm_itw, same in-the-wild training data, labels "
            "0=bona-fide 1=spoof. Distinct checkpoint and distinct backbone, so it is a real second "
            "observation rather than the same weights under another name."
        ),
    ),
    ModelSpec(
        key="deepfake_audio_hemgg",
        repo="Hemgg/Deepfake-audio-detection",
        kind=AUDIO,
        weight=1.0,
        notes=(
            "wav2vec2 base fine-tune, labels {0: AIVoice, 1: HumanVoice}. Those resolve by name "
            "only because _label_tokens splits camelCase; before that it was the one checkpoint "
            "here that needed a fake_index, which is the index position assumption this project "
            "removed once already. No forced index is set, so if the labels are ever renamed it "
            "refuses to load rather than silently inverting."
        ),
    ),
)

BUILTIN_MODELS: tuple[ModelSpec, ...] = SYNTHETIC_MODELS + FACE_MODELS + AUDIO_MODELS


def _coerce_local_spec(entry: dict) -> ModelSpec:
    """Build a spec from one models.local.json entry, resolving a relative path against backend/."""
    missing = [f for f in ("key", "path") if not str(entry.get(f, "")).strip()]
    if missing:
        raise ValueError(f"missing or empty required field(s): {', '.join(missing)}")

    key = str(entry["key"]).strip()
    raw_path = str(entry["path"]).strip()
    kind = str(entry.get("kind", SYNTHETIC)).strip().lower()
    if kind not in MODEL_KINDS:
        allowed = ", ".join(repr(k) for k in MODEL_KINDS)
        raise ValueError(f"{key}: kind must be one of {allowed}, got {kind!r}")

    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = BASE_DIR / path

    fake_index = entry.get("fake_index")
    return ModelSpec(
        key=key,
        repo=str(path),
        kind=kind,
        weight=float(entry.get("weight", 1.0)),
        gradcam_target=entry.get("gradcam_target") or None,
        notes=str(entry.get("notes", "")),
        fake_index=None if fake_index is None else int(fake_index),
        is_local=True,
    )


def load_local_specs(path: Path = LOCAL_MODELS_PATH) -> tuple[tuple[ModelSpec, ...], list[str]]:
    """Read locally held checkpoints declared in JSON, returning the specs and any complaints.

    This exists so private or unreleased weights can join the ensemble without editing code and
    without inventing a Hub repo id for something that has no Hub presence. A malformed file
    degrades to no local models with the reason reported, on the same principle as a dead
    checkpoint: configuration problems must be visible but must not stop the service booting.

    The path is only recorded here. Whether it actually contains a loadable model is the
    registry's problem, and a missing directory is reported there like any other load failure.
    """
    if not path.is_file():
        return (), []

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return (), [f"{path.name} could not be read, so no local models were registered: {exc}"]

    entries = payload.get("models") if isinstance(payload, dict) else payload
    if not isinstance(entries, list):
        return (), [f"{path.name} must contain a list of models, or an object with a models list."]

    specs: list[ModelSpec] = []
    problems: list[str] = []
    seen: set[str] = set()
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            problems.append(f"{path.name} entry {index} is not an object, so it was skipped.")
            continue
        try:
            spec = _coerce_local_spec(entry)
        except (KeyError, TypeError, ValueError) as exc:
            problems.append(f"{path.name} entry {index} was skipped: {exc}")
            continue
        if spec.key in seen or any(spec.key == m.key for m in BUILTIN_MODELS):
            problems.append(f"{path.name} entry {index} reuses the key {spec.key!r}, so it was skipped.")
            continue
        seen.add(spec.key)
        specs.append(spec)

    return tuple(specs), problems


LOCAL_SPECS, LOCAL_SPEC_PROBLEMS = load_local_specs()

LOCAL_SYNTHETIC_MODELS: tuple[ModelSpec, ...] = tuple(s for s in LOCAL_SPECS if s.kind == SYNTHETIC)
LOCAL_FACE_MODELS: tuple[ModelSpec, ...] = tuple(s for s in LOCAL_SPECS if s.kind == FACE)
LOCAL_AUDIO_MODELS: tuple[ModelSpec, ...] = tuple(s for s in LOCAL_SPECS if s.kind == AUDIO)

ALL_SYNTHETIC_MODELS: tuple[ModelSpec, ...] = SYNTHETIC_MODELS + LOCAL_SYNTHETIC_MODELS
ALL_FACE_MODELS: tuple[ModelSpec, ...] = FACE_MODELS + LOCAL_FACE_MODELS
ALL_AUDIO_MODELS: tuple[ModelSpec, ...] = AUDIO_MODELS + LOCAL_AUDIO_MODELS

ALL_MODELS: tuple[ModelSpec, ...] = ALL_SYNTHETIC_MODELS + ALL_FACE_MODELS
"""The image ensemble only, deliberately excluding audio.

Everything in here is loaded by HFImageClassifier. Audio checkpoints need a different Auto class and
a feature extractor rather than an image processor, so putting them in this tuple would hand an
audio model to the image wrapper and produce a load failure that reads like a dead repo. The
prefetch and verification scripts iterate both tuples separately for the same reason.
"""

# Matched against label text by detectors/hf_image.py, never against index position.
#
# Deliberately contains no bare digits. "0" and "1" were listed here originally and were dead
# entries, skipped by a length filter in the matcher. Reviving them would resolve LABEL_0 and
# LABEL_1 as real and fake respectively, which is index position wearing a label's clothes, and
# inverts every prediction on any checkpoint that happens to order its classes the other way.
# A checkpoint with generic labels must fail and be given an explicit fake_index instead.
FAKE_LABEL_TOKENS = (
    "fake",
    "ai",
    "artificial",
    "generated",
    "synthetic",
    "deepfake",
    "spoof",
    "manipulated",
)

REAL_LABEL_TOKENS = (
    "real",
    "human",
    "authentic",
    "genuine",
    "natural",
    "pristine",
    "live",
    # Anti-spoofing terminology, which the audio checkpoints use instead of "real". Without this,
    # three of the seven audio models are refused outright: "bona-fide" tokenises to ["bona", "fide"]
    # and "Bonafide" to ["bonafide"], and neither matched anything above, so resolve_fake_indices
    # found no real class and marked the checkpoint unavailable. "bona" covers both, exactly and by
    # prefix, and collides with nothing in FAKE_LABEL_TOKENS. "spoof" is already handled there.
    "bona",
)


@dataclass
class Thresholds:
    """Score boundaries for the three band verdict.

    Anything between authentic_max and ai_min is reported as uncertain rather than
    forced into a binary. Widen the band to trade recall for trustworthiness.
    """

    authentic_max: float = field(default_factory=lambda: _env_float("VT_AUTHENTIC_MAX", 0.35))
    ai_min: float = field(default_factory=lambda: _env_float("VT_AI_MIN", 0.65))

    def verdict(self, score: float) -> str:
        if score >= self.ai_min:
            return VERDICT_AI
        if score <= self.authentic_max:
            return VERDICT_AUTHENTIC
        return VERDICT_UNCERTAIN


@dataclass
class FusionConfig:
    """Weights and limits applied in logit space during fusion.

    face_weight only participates when at least one face is found. provenance_weight is
    deliberately small because provenance acts through hard overrides when it has real
    evidence, and absence of metadata is close to meaningless on its own.

    signal_clamp bounds every model probability before its logit is taken. Logit space is
    unbounded, so an uncalibrated checkpoint reporting 0.001 contributes roughly -6.9 and can
    outvote the entire rest of the ensemble on its own. None of these models has earned a
    1 in 1000 claim, so the clamp caps any single member's pull at about 3.9. Provenance
    overrides are exempt because they short circuit fusion entirely.

    dissent_escalates encodes an asymmetry: these detectors recognise the generator
    fingerprints they were trained on and default to "real" on everything else, including
    recompressed images and newer generators. That makes a vote for "real" weak evidence and a
    vote for "AI" strong evidence, so averaging them symmetrically manufactures false
    negatives. When one member clears ai_min, the verdict is held at uncertain rather than
    allowed to settle on authentic. This trades false negatives for false positives on
    purpose. Set VT_DISSENT_ESCALATES=false to get plain averaging back.

    face_decides hands the whole verdict to the face pathway whenever its reading lands outside
    the uncertain band, in either direction, and is the one place this project lets a single
    uncalibrated model overrule the ensemble. It exists because on a face swap the whole image
    detectors are answering a different question, correctly reading camera pixels everywhere
    outside the swapped region, and averaging them against the face model buries the finding: a
    face read at 1.00 against three whole image readings of 0.02 fuses to 0.174. It was chosen
    deliberately on 2026-08-21 with the cost stated, which is that the authentic direction now
    lets one model close a case, so a generated face this model does not recognise is called real
    while any whole image detector that spotted it is overruled. It is disabled by default;
    set VT_FACE_DECIDES=true only after measuring the trade-off on a representative labelled
    face-swap set. The default returns to fusion plus the dissent floor.

    audio_weight applies to the audio pathway on the same terms as synthetic_weight, and there is
    deliberately no audio equivalent of face_decides. The face override is the single declared
    exception to the no-veto rule and adding a second one is a product decision of the same weight,
    not a detail of adding a modality. Audio detectors instead reach the verdict through fusion plus
    the dissent floor, so one model reading spoof holds the verdict at uncertain rather than settling
    it. A weighted quorum on escalation was tried while adding audio, requiring several dissenters
    before the floor applied, and it was removed: it inverts the asymmetry the floor exists for. A
    lone positive is the strong reading, silence is the weak one, and uncertain is an abstention
    rather than an accusation, so there is nothing for a quorum to protect against here.
    """

    synthetic_weight: float = field(default_factory=lambda: _env_float("VT_W_SYNTHETIC", 1.0))
    face_weight: float = field(default_factory=lambda: _env_float("VT_W_FACE", 1.2))
    audio_weight: float = field(default_factory=lambda: _env_float("VT_W_AUDIO", 1.0))
    provenance_weight: float = field(default_factory=lambda: _env_float("VT_W_PROVENANCE", 0.4))
    temperature: float = field(default_factory=lambda: _env_float("VT_TEMPERATURE", 1.0))
    signal_clamp: float = field(default_factory=lambda: _env_float("VT_SIGNAL_CLAMP", 0.02))
    dissent_escalates: bool = field(
        default_factory=lambda: _env_bool("VT_DISSENT_ESCALATES", True)
    )
    audio_dissent_min_weight: float = field(
        default_factory=lambda: _env_float("VT_AUDIO_DISSENT_MIN_WEIGHT", 0.2)
    )
    face_dissent_min_weight: float = field(
        default_factory=lambda: _env_float("VT_FACE_DISSENT_MIN_WEIGHT", 0.5)
    )
    # A face checkpoint is useful evidence, but it has not been evaluated or calibrated on this
    # deployment.  Do not let it silently replace the whole-image result by default; that made the
    # answer for ordinary portraits hinge on one FF++-trained checkpoint.  Operators who have
    # measured that trade-off on their own face-swap set can opt in explicitly.
    face_decides: bool = field(default_factory=lambda: _env_bool("VT_FACE_DECIDES", False))
    logit_spread_limit: float = field(
        default_factory=lambda: _env_float("VT_LOGIT_SPREAD_LIMIT", 2.2)
    )


@dataclass
class Settings:
    device: str = field(default_factory=lambda: _env_str("VT_DEVICE", "auto"))
    dtype: str = field(default_factory=lambda: _env_str("VT_DTYPE", "auto"))
    max_upload_bytes: int = field(default_factory=lambda: _env_int("VT_MAX_UPLOAD_BYTES", 20 * 1024 * 1024))
    max_pixels: int = field(default_factory=lambda: _env_int("VT_MAX_PIXELS", 50_000_000))
    max_edge: int = field(default_factory=lambda: _env_int("VT_MAX_EDGE", 2048))
    enable_heatmap: bool = field(default_factory=lambda: _env_bool("VT_ENABLE_HEATMAP", True))
    enable_faces: bool = field(default_factory=lambda: _env_bool("VT_ENABLE_FACES", True))
    face_detector: str = field(default_factory=lambda: _env_str("VT_FACE_DETECTOR", "auto"))
    yunet_path: str = field(
        default_factory=lambda: _env_str("VT_YUNET_PATH", str(BASE_DIR / "weights" / "face_detection_yunet_2023mar.onnx"))
    )
    face_score_threshold: float = field(default_factory=lambda: _env_float("VT_FACE_SCORE", 0.6))
    face_min_size: int = field(default_factory=lambda: _env_int("VT_FACE_MIN_SIZE", 64))
    face_margin: float = field(default_factory=lambda: _env_float("VT_FACE_MARGIN", 0.15))
    max_faces: int = field(default_factory=lambda: _env_int("VT_MAX_FACES", 8))
    # Audio is analysed in overlapping windows rather than as one clip, so a few seconds of cloned
    # speech spliced into a real recording is not averaged away. audio.py explains the window and
    # hop choice, the silence gate and why the per model aggregate is a high quantile and not a max.
    enable_audio: bool = field(default_factory=lambda: _env_bool("VT_ENABLE_AUDIO", True))
    max_audio_bytes: int = field(
        default_factory=lambda: _env_int("VT_MAX_AUDIO_BYTES", 40 * 1024 * 1024)
    )
    max_audio_seconds: float = field(
        default_factory=lambda: _env_float("VT_MAX_AUDIO_SECONDS", 600.0)
    )
    audio_sample_rate: int = field(default_factory=lambda: _env_int("VT_AUDIO_SAMPLE_RATE", 16000))
    audio_window_seconds: float = field(
        default_factory=lambda: _env_float("VT_AUDIO_WINDOW_SECONDS", 4.0)
    )
    audio_hop_seconds: float = field(default_factory=lambda: _env_float("VT_AUDIO_HOP_SECONDS", 2.0))
    audio_max_windows: int = field(default_factory=lambda: _env_int("VT_AUDIO_MAX_WINDOWS", 24))
    audio_min_rms: float = field(default_factory=lambda: _env_float("VT_AUDIO_MIN_RMS", 0.004))
    audio_quantile: float = field(default_factory=lambda: _env_float("VT_AUDIO_QUANTILE", 0.9))
    max_concurrency: int = field(default_factory=lambda: max(1, _env_int("VT_MAX_CONCURRENCY", 2)))
    # Current image models have not been calibrated on a labelled deployment set.  A low raw score
    # is therefore not enough evidence to affirm authenticity; callers can opt back into that
    # risky behaviour only after fitting and reviewing their own evaluation set.
    allow_uncalibrated_authentic: bool = field(
        default_factory=lambda: _env_bool("VT_ALLOW_UNCALIBRATED_AUTHENTIC", False)
    )
    hf_cache: str | None = field(default_factory=lambda: os.environ.get("HF_HOME"))
    allow_origins: tuple[str, ...] = ("*",)
    thresholds: Thresholds = field(default_factory=Thresholds)
    fusion: FusionConfig = field(default_factory=FusionConfig)

    def resolved_device(self) -> str:
        if self.device != "auto":
            return self.device
        try:
            import torch
        except ImportError:
            return "cpu"
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"


ALLOWED_MIME = {
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/bmp",
    "image/tiff",
}

# WAV, FLAC and Ogg decode through soundfile, which libsndfile backs. MP3 needs a recent
# libsndfile and MP4 family containers need PyAV, so both are listed here and reported as an
# unavailable decoder rather than an unsupported format when the backend is missing. m4a is on the
# list because it is what an iPhone voice memo actually is.
ALLOWED_AUDIO_MIME = {
    "audio/wav",
    "audio/flac",
    "audio/ogg",
    "audio/mpeg",
    "audio/mp4",
    "audio/aac",
    "audio/webm",
}

settings = Settings()
