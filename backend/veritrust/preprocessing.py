"""Image decoding and normalisation. Pure PIL and numpy, no torch.

Per model tensor preparation deliberately does not live here. Each detector uses its own
AutoImageProcessor so that resize, rescale and channel normalisation always match how that
checkpoint was trained. Hardcoding those steps here is what silently breaks a detector.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np

from .config import ALLOWED_MIME, Settings

Image.MAX_IMAGE_PIXELS = None

_MAGIC = (
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"BM", "image/bmp"),
    (b"II*\x00", "image/tiff"),
    (b"MM\x00*", "image/tiff"),
)


class ImageRejected(ValueError):
    """Raised when input is unusable or outside configured safety limits."""


@dataclass
class DecodedImage:
    image: Image.Image
    width: int
    height: int
    original_width: int
    original_height: int
    mime: str
    downscaled: bool
    raw: bytes

    @property
    def megapixels(self) -> float:
        return round(self.original_width * self.original_height / 1_000_000, 2)


def sniff_mime(data: bytes) -> str | None:
    """Identify format from magic bytes. Never trust a client supplied content type."""
    for magic, mime in _MAGIC:
        if data.startswith(magic):
            return mime
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


def decode_image(data: bytes, settings: Settings) -> DecodedImage:
    if not data:
        raise ImageRejected("Empty upload.")
    if len(data) > settings.max_upload_bytes:
        limit_mb = settings.max_upload_bytes / (1024 * 1024)
        raise ImageRejected(f"File exceeds the {limit_mb:.0f} MB limit.")

    mime = sniff_mime(data)
    if mime is None:
        raise ImageRejected("Unrecognised image format.")
    if mime not in ALLOWED_MIME:
        raise ImageRejected(f"Unsupported format: {mime}")

    try:
        probe = Image.open(io.BytesIO(data))
        probe.verify()
    except Exception as exc:
        raise ImageRejected("File is not a readable image.") from exc

    try:
        image = Image.open(io.BytesIO(data))
        original_width, original_height = image.size
        if original_width * original_height > settings.max_pixels:
            raise ImageRejected(
                f"Image is {original_width}x{original_height}, above the "
                f"{settings.max_pixels / 1_000_000:.0f} MP limit."
            )
        image = ImageOps.exif_transpose(image)
        image = image.convert("RGB")
    except ImageRejected:
        raise
    except Exception as exc:
        raise ImageRejected("Failed to decode image.") from exc

    downscaled = False
    if max(image.size) > settings.max_edge:
        image = _fit_within(image, settings.max_edge)
        downscaled = True

    return DecodedImage(
        image=image,
        width=image.width,
        height=image.height,
        original_width=original_width,
        original_height=original_height,
        mime=mime,
        downscaled=downscaled,
        raw=data,
    )


def _fit_within(image: Image.Image, max_edge: int) -> Image.Image:
    scale = max_edge / max(image.size)
    size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    return image.resize(size, Image.Resampling.LANCZOS)


def to_numpy(image: Image.Image) -> np.ndarray:
    """HWC uint8 array."""
    return np.asarray(image, dtype=np.uint8)


