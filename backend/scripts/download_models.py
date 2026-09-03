"""Prefetch checkpoints and the YuNet face detector so first request is not a cold download.

    python -m scripts.download_models

YuNet is fetched from the media endpoint, not raw.githubusercontent.com. opencv_zoo tracks the
ONNX file with Git LFS, and the raw host serves the LFS pointer text for those: a 131 byte file
beginning "version https://git-lfs.github.com/spec/v1" that every existence check happily accepts
and only fails much later inside cv2. The downloaded bytes are validated here for that reason.
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from veritrust.config import ALL_AUDIO_MODELS, ALL_MODELS, settings
from veritrust.faces import yunet_file_problem
from veritrust.detectors.torch_image import (
    raw_checkpoint_path,
    raw_checkpoint_url,
    uses_raw_torch_checkpoint,
)

YUNET_URL = (
    "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx"
)

YUNET_FALLBACK_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx"
)


def fetch_yunet() -> None:
    target = Path(settings.yunet_path)

    if target.is_file():
        problem = yunet_file_problem(target)
        if problem is None:
            size = target.stat().st_size / 1024
            print(f"[ ok ] YuNet already present at {target}, {size:.0f} KB")
            return
        print(f"[warn] Replacing the existing YuNet file: {problem}")

    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading YuNet to {target}")

    for url in (YUNET_URL, YUNET_FALLBACK_URL):
        temporary = target.with_suffix(target.suffix + ".part")
        try:
            urllib.request.urlretrieve(url, temporary)
        except Exception as exc:
            print(f"[warn] {url} failed: {type(exc).__name__}: {exc}")
            temporary.unlink(missing_ok=True)
            continue

        problem = yunet_file_problem(temporary)
        if problem is not None:
            print(f"[warn] {url} returned an unusable file: {problem}")
            temporary.unlink(missing_ok=True)
            continue

        temporary.replace(target)
        print(f"[ ok ] YuNet saved, {target.stat().st_size / 1024:.0f} KB")
        return

    print("[warn] Could not fetch a usable YuNet model.")
    print("       The face pathway will fall back to the bundled Haar cascade, which only")
    print("       reliably finds roughly frontal faces, so face swaps on turned or small faces")
    print("       will be missed entirely.")


def fetch_specs(specs, fetch_one) -> None:
    """Walk each spec's fallback chain until one candidate downloads.

    Stopping at the first success is deliberate and matches what the wrappers do at load time.
    Fetching the rest would pull weights that serving will never open, and on the image side
    broad_swin's fallback is sdxl_swin's primary, so a chain that kept going would download the
    same repo twice and then have it deduplicated anyway.
    """
    for spec in specs:
        if spec.is_local:
            if Path(spec.repo).is_dir():
                print(f"[skip] {spec.key}: local checkpoint at {spec.repo}, nothing to download")
            else:
                print(f"[skip] {spec.key}: local checkpoint not found at {spec.repo}")
            continue

        for repo in (spec.repo, *spec.fallbacks):
            try:
                print(f"Fetching {repo}")
                fetch_one(repo)
                print(f"[ ok ] {repo}")
                break
            except Exception as exc:
                print(f"[warn] {repo}: {type(exc).__name__}: {exc}")


def fetch_image_models() -> None:
    from transformers import AutoModelForImageClassification

    from veritrust.detectors.hf_image import load_processor

    def fetch_one(repo: str) -> None:
        load_processor(repo)
        AutoModelForImageClassification.from_pretrained(repo)

    # The FF++ face member is a raw .pth file already managed locally, not a Hub checkpoint in
    # Transformers format.  Asking AutoModelForImageClassification to prefetch it cannot warm the
    # serving path and obscures whether the actual file exists.
    hf_specs = tuple(spec for spec in ALL_MODELS if not uses_raw_torch_checkpoint(spec))
    raw_specs = tuple(spec for spec in ALL_MODELS if uses_raw_torch_checkpoint(spec))
    fetch_specs(hf_specs, fetch_one)
    for spec in raw_specs:
        target = raw_checkpoint_path(spec)
        if target.is_file():
            print(f"[ ok ] {spec.key}: raw PyTorch checkpoint already present at {target}")
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".part")
        url = raw_checkpoint_url(spec)
        try:
            print(f"Fetching {url}")
            urllib.request.urlretrieve(url, temporary)
            temporary.replace(target)
            print(f"[ ok ] {spec.key}: saved raw PyTorch checkpoint to {target}")
        except Exception as exc:
            temporary.unlink(missing_ok=True)
            print(f"[warn] {spec.key}: could not fetch raw checkpoint: {type(exc).__name__}: {exc}")


def fetch_audio_models() -> None:
    """Prefetch the audio pathway through the same auto classes the wrapper uses at load time.

    AutoFeatureExtractor rather than AutoImageProcessor, and the audio head rather than the image
    one. Going through the wrong pair would populate the cache with something from_pretrained then
    refuses on the serving path, which reads as a network problem on a machine that is fully
    warmed up.
    """
    if not settings.enable_audio:
        print(
            f"[skip] VT_ENABLE_AUDIO is off, so the {len(ALL_AUDIO_MODELS)} audio checkpoint(s) "
            f"are not fetched. They would not be loaded at serve time either."
        )
        return

    from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

    def fetch_one(repo: str) -> None:
        AutoFeatureExtractor.from_pretrained(repo)
        AutoModelForAudioClassification.from_pretrained(repo)

    fetch_specs(ALL_AUDIO_MODELS, fetch_one)


def main() -> int:
    fetch_yunet()
    print()
    fetch_image_models()
    print()
    fetch_audio_models()
    print("\nDone. Run scripts/verify_models.py to confirm label resolution.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
