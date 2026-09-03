"""Confirm every configured checkpoint still resolves, and that its labels are unambiguous.

Run this first. Hugging Face repos get renamed, gated or deleted, so the model registry in
config.py is a set of intentions rather than a guarantee. This reports the truth for your
machine and exits non zero if the ensemble came up empty.

    python -m scripts.verify_models

Both modalities are checked, image and audio, because label resolution is where these fail
quietly. The audio checkpoints genuinely disagree about class order, so one of them loading with
an inverted mapping would produce confident scores pointing the wrong way, and printing the
resolved mapping is the only way to see that before it reaches a verdict.

It also checks the YuNet file and the locally declared checkpoints, because both fail in ways that
look like success from the outside: an LFS pointer passes every existence check, and a mistyped path
in models.local.json silently produces an ensemble one member smaller than you think.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image

from veritrust.config import (
    ALL_AUDIO_MODELS,
    ALL_MODELS,
    LOCAL_MODELS_PATH,
    LOCAL_SPEC_PROBLEMS,
    LOCAL_SPECS,
    settings,
)
from veritrust.detectors.base import LoadError
from veritrust.detectors.hf_audio import HFAudioClassifier
from veritrust.detectors.hf_image import HFImageClassifier
from veritrust.detectors.torch_image import TorchImageClassifier, uses_raw_torch_checkpoint
from veritrust.faces import yunet_file_problem

TICK = "[ ok ]"
CROSS = "[fail]"
WARN = "[warn]"


def probe_image() -> Image.Image:
    """A trivial gradient. Only used to confirm a forward pass runs end to end."""
    import numpy as np

    grid = np.zeros((256, 256, 3), dtype=np.uint8)
    grid[..., 0] = np.linspace(0, 255, 256, dtype=np.uint8)[None, :]
    grid[..., 1] = np.linspace(0, 255, 256, dtype=np.uint8)[:, None]
    grid[..., 2] = 128
    return Image.fromarray(grid)


def probe_audio():
    """One window of a bare tone at the rate the pipeline resamples to.

    Deliberately not silence. Several of these checkpoints front a voice activity or energy gate,
    and a digitally silent buffer can return a degenerate score or divide by a zero norm, which
    would report as a broken checkpoint when the input was the problem. A tone is not speech
    either, but it exercises the feature extractor and the forward pass, which is all this proves.
    """
    import numpy as np

    rate = settings.audio_sample_rate
    t = np.arange(int(rate * settings.audio_window_seconds), dtype=np.float32) / rate
    return (0.1 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32), rate


def report_environment() -> str | None:
    """Explains a CPU fallback instead of just reporting it.

    A CUDA-less torch build on a machine with a GPU looks exactly like a machine without one, and
    the default PyPI torch wheel for Windows is CPU only, so this is an easy trap to fall into.
    """
    try:
        import torch
    except ImportError:
        print(f"{CROSS} torch is not installed. Install requirements.txt first.")
        return None

    print(f"python {sys.version.split()[0]}, torch {torch.__version__}")
    device = settings.resolved_device()
    print(f"Device: {device}")

    if device == "cpu":
        built_against = getattr(torch.version, "cuda", None)
        if built_against is None:
            print(
                f"  {WARN} this torch build has no CUDA support at all. If you have an NVIDIA "
                f"GPU, reinstall torch from the PyTorch CUDA index; see requirements.txt."
            )
        else:
            print(
                f"  {WARN} torch was built against CUDA {built_against} but no GPU is visible. "
                f"Usually a driver problem."
            )
        print("  CPU inference still works, it is just slower.")
    return device


def report_face_detector() -> None:
    """Report the YuNet file's real state, since existence alone says nothing about validity."""
    path = Path(settings.yunet_path)
    if not path.is_file():
        print(f"{WARN} YuNet not present at {path}")
        print("       Run scripts/download_models.py. Without it the face pathway uses Haar,")
        print("       which only finds roughly frontal faces.")
        return

    problem = yunet_file_problem(path)
    if problem is not None:
        print(f"{CROSS} YuNet file is unusable: {problem}")
        print("       Delete it and rerun scripts/download_models.py.")
        return
    print(f"{TICK} YuNet present at {path}, {path.stat().st_size / 1024:.0f} KB")


def report_local_specs() -> None:
    """Report what models.local.json declared and whether each path actually exists."""
    for problem in LOCAL_SPEC_PROBLEMS:
        print(f"{WARN} {problem}")

    if not LOCAL_SPECS:
        if LOCAL_MODELS_PATH.is_file():
            print(f"No local checkpoints declared in {LOCAL_MODELS_PATH}")
        else:
            print(f"No {LOCAL_MODELS_PATH.name} present, so only the built in checkpoints are used.")
        return

    print(f"{len(LOCAL_SPECS)} local checkpoint(s) declared in {LOCAL_MODELS_PATH}:")
    for spec in LOCAL_SPECS:
        state = TICK if Path(spec.repo).is_dir() else CROSS
        print(f"  {state} {spec.key} ({spec.kind}, weight {spec.weight}) -> {spec.repo}")


def check_pathway(specs, factory, run_probe, device, claimed) -> tuple[int, int]:
    """Load and forward-pass every spec in one pathway. Returns (usable, came from a fallback).

    Both modalities go through this because everything that can go wrong is shared: a renamed repo,
    a gated repo, labels that cannot be resolved by name, a checkpoint that loads but cannot run,
    and two keys resolving to the same weights. Only the wrapper class and the probe input differ,
    so those are arguments rather than a second copy of the loop.

    claimed is passed in rather than created here so the collision check spans the whole run while
    staying keyed by pathway, which is the rule Registry._build applies.
    """
    loaded = 0
    degraded = 0

    for spec in specs:
        print(f"{spec.key} ({spec.kind})")
        print(f"  configured: {spec.repo}{' [local]' if spec.is_local else ''}")
        # Serving uses a raw PyTorch wrapper for the FF++ face checkpoint.  Probing it through
        # AutoModelForImageClassification would report a false failure (and previously did), so
        # verification must select the same implementation as Registry._build.
        detector_factory = TorchImageClassifier if uses_raw_torch_checkpoint(spec) else factory
        detector = detector_factory(spec, device, settings.dtype)
        try:
            detector.load()
        except LoadError as exc:
            print(f"  {CROSS} {exc}\n")
            continue
        except Exception as exc:
            print(f"  {CROSS} {type(exc).__name__}: {exc}\n")
            continue

        if detector.repo_used != spec.repo:
            degraded += 1
            print(f"  {WARN} primary unavailable, fell back to {detector.repo_used}")
        print(f"  {TICK} loaded {detector.repo_used}")
        print(f"  labels: {detector.label_mapping}")
        fake_indices = getattr(detector, "_fake_indices", None)
        if fake_indices is not None:
            print(f"  treating as generated: index {fake_indices}")
        else:
            print("  treating as generated: raw checkpoint class mapping shown above")

        # Only the audio wrapper has this. A checkpoint wanting a rate other than the one audio.py
        # resamples to still runs, since the extractor handles it, but it is worth seeing.
        wanted = getattr(detector, "expected_sample_rate", None)
        if wanted is not None and wanted != settings.audio_sample_rate:
            print(
                f"  {WARN} extractor wants {wanted} Hz, pipeline delivers "
                f"{settings.audio_sample_rate} Hz"
            )

        # Mirrors Registry._build so the collision is visible here rather than only as a runtime
        # failure entry. Keyed by pathway, since the same weights on a face crop and a whole image
        # are two separate observations.
        slot = (spec.kind, str(detector.repo_used))
        if slot in claimed:
            print(f"  {WARN} same checkpoint as {claimed[slot]}, so serving will drop this one")
        else:
            claimed[slot] = spec.key

        result = run_probe(detector)
        if result.usable:
            loaded += 1
            print(f"  {TICK} forward pass ok, p_ai={result.p_fake:.4f} in {result.latency_ms:.0f} ms")
            print("  note: that score is on a meaningless synthetic input, it proves plumbing only")
        else:
            print(f"  {CROSS} forward pass failed: {result.error}")
        print()

    return loaded, degraded


def main() -> int:
    device = report_environment()
    if device is None:
        return 1
    print()
    report_face_detector()
    print()
    report_local_specs()

    audio_specs = ALL_AUDIO_MODELS if settings.enable_audio else ()
    if not settings.enable_audio:
        print(
            f"\n{WARN} VT_ENABLE_AUDIO is off, so the {len(ALL_AUDIO_MODELS)} audio checkpoint(s) "
            f"are not checked and will not load at serve time either."
        )

    total = len(ALL_MODELS) + len(audio_specs)
    print(f"\nChecking {total} configured checkpoint(s).\n")

    claimed: dict[tuple[str, str], str] = {}

    sample = probe_image()
    loaded, degraded = check_pathway(
        ALL_MODELS, HFImageClassifier, lambda d: d.predict(sample), device, claimed
    )

    if audio_specs:
        waveform, rate = probe_audio()
        audio_loaded, audio_degraded = check_pathway(
            audio_specs, HFAudioClassifier, lambda d: d.predict(waveform, rate), device, claimed
        )
        loaded += audio_loaded
        degraded += audio_degraded

    print(f"Usable checkpoints: {loaded} of {total}")
    if degraded:
        print(f"{degraded} checkpoint(s) came from a fallback. Update config.py if this persists.")
    if loaded == 0:
        print("\nNothing loaded. The ensemble cannot run. Check network access and HF availability.")
        return 1
    if loaded < total:
        print("\nRunning degraded. Fusion will use whatever loaded and the API will report it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
