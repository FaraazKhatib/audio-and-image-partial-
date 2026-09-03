"""Provenance and metadata evidence. Pure stdlib and PIL, no torch.

This is the highest precision signal available and it is entirely non statistical. A signed
C2PA manifest asserting trainedAlgorithmicMedia, or a Stable Diffusion parameters chunk, is
direct evidence rather than a guess.

The asymmetry matters and is encoded deliberately. Positive evidence of generation is close
to conclusive. Absence of metadata proves nothing, because screenshots, social platform
re-encodes and ordinary strip-on-upload all destroy metadata on genuine photos. So a bare
image returns no opinion instead of leaning fake.

Scanning is confined to metadata regions. A whole file byte scan would flag a photograph of
a Midjourney screenshot, or a file named midjourney.jpg, as generated.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass, field

from PIL import Image
from PIL.ExifTags import TAGS

GENERATOR_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"stable\s*diffusion", "Stable Diffusion"),
    (r"automatic1111|a1111", "AUTOMATIC1111"),
    (r"comfyui", "ComfyUI"),
    (r"midjourney", "Midjourney"),
    (r"dall[\s\-_]?e", "DALL-E"),
    (r"openai", "OpenAI"),
    (r"adobe\s*firefly|firefly", "Adobe Firefly"),
    (r"\bflux\b", "Flux"),
    (r"\bsdxl\b", "SDXL"),
    (r"novelai", "NovelAI"),
    (r"leonardo\.ai|leonardoai", "Leonardo AI"),
    (r"ideogram", "Ideogram"),
    (r"stability\s*ai", "Stability AI"),
    (r"\bimagen\b", "Imagen"),
    (r"invokeai", "InvokeAI"),
    (r"\bsora\b", "Sora"),
    (r"runway", "Runway"),
    (r"recraft", "Recraft"),
    (r"playground\s*ai", "Playground AI"),
)

AI_SOURCE_TYPE = re.compile(
    r"digitalSourceType[^>]{0,120}?(trainedAlgorithmicMedia|compositeWithTrainedAlgorithmicMedia"
    r"|algorithmicMedia)",
    re.IGNORECASE,
)

GENERATION_KEYS = (
    "parameters",
    "prompt",
    "workflow",
    "sd-metadata",
    "dream",
    "negative_prompt",
    "generation_data",
    "aiGenerated",
)

CAMERA_TAGS = ("Make", "Model", "DateTimeOriginal", "ExposureTime", "FNumber", "ISOSpeedRatings")

XMP_RE = re.compile(rb"<x:xmpmeta.{0,65536}?</x:xmpmeta>", re.DOTALL)


@dataclass
class ProvenanceResult:
    """p_fake of None means no opinion, which excludes this signal from fusion entirely."""

    p_fake: float | None = None
    override: bool = False
    c2pa_present: bool = False
    c2pa_ai_declared: bool = False
    exif_present: bool = False
    camera_signature: bool = False
    generators: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "p_fake": self.p_fake,
            "override": self.override,
            "c2pa_present": self.c2pa_present,
            "c2pa_ai_declared": self.c2pa_ai_declared,
            "exif_present": self.exif_present,
            "camera_signature": self.camera_signature,
            "generators": self.generators,
            "evidence": self.evidence,
        }


def _decode_exif(image: Image.Image) -> dict[str, str]:
    out: dict[str, str] = {}
    try:
        raw = image.getexif()
    except Exception:
        return out
    if not raw:
        return out
    for tag_id, value in raw.items():
        name = TAGS.get(tag_id, str(tag_id))
        if isinstance(value, bytes):
            value = value.decode("utf-8", "replace")
        out[name] = str(value)
    try:
        for ifd_id in (0x8769, 0x8825):
            ifd = raw.get_ifd(ifd_id)
            for tag_id, value in (ifd or {}).items():
                name = TAGS.get(tag_id, str(tag_id))
                if isinstance(value, bytes):
                    value = value.decode("utf-8", "replace")
                out.setdefault(name, str(value))
    except Exception:
        pass
    return out


def _extract_xmp(raw: bytes) -> str:
    match = XMP_RE.search(raw)
    return match.group(0).decode("utf-8", "replace") if match else ""


def _detect_c2pa(raw: bytes, info: dict) -> bool:
    """Detect a C2PA manifest container without validating its signature.

    JPEG carries it in APP11 JUMBF boxes, PNG in a caBX chunk. Presence alone is reported;
    cryptographic validation requires the c2pa package and is attempted separately.
    """
    if "c2pa" in info or "caBX" in info:
        return True
    head = raw[:4_000_000]
    if b"caBX" in head:
        return True
    if b"jumb" in head and (b"c2pa" in head or b"contentauth" in head.lower()):
        return True
    return b"urn:uuid:c2pa" in head or b"c2pa.assertions" in head


def _validate_c2pa(raw: bytes) -> tuple[bool, list[str]]:
    """Optional real validation when the c2pa package is installed."""
    try:
        import c2pa
    except ImportError:
        return False, []
    try:
        reader = c2pa.Reader(io.BytesIO(raw))
        manifest_json = reader.json()
    except Exception:
        try:
            manifest_json = c2pa.read_file(raw, None)
        except Exception:
            return False, []
    if not manifest_json:
        return False, []
    text = str(manifest_json)
    evidence = ["Signed C2PA manifest read successfully."]
    declared = bool(AI_SOURCE_TYPE.search(text))
    for pattern, label in GENERATOR_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            evidence.append(f"C2PA manifest names {label}.")
            declared = True
    return declared, evidence


def inspect(raw: bytes) -> ProvenanceResult:
    result = ProvenanceResult()
    try:
        image = Image.open(io.BytesIO(raw))
    except Exception:
        return result

    info = dict(getattr(image, "info", {}) or {})
    exif = _decode_exif(image)
    xmp = _extract_xmp(raw) or str(info.get("XML:com.adobe.xmp", "") or "")

    result.exif_present = bool(exif)
    result.camera_signature = sum(1 for tag in CAMERA_TAGS if exif.get(tag)) >= 3

    text_parts: list[str] = []
    for key, value in info.items():
        if isinstance(value, (str, bytes)) and key.lower() != "icc_profile":
            decoded = value.decode("utf-8", "replace") if isinstance(value, bytes) else value
            text_parts.append(f"{key}: {decoded[:4000]}")
    for key, value in exif.items():
        text_parts.append(f"{key}: {value[:2000]}")
    if xmp:
        text_parts.append(xmp[:16000])
    haystack = "\n".join(text_parts)

    for key in GENERATION_KEYS:
        for present in info:
            if present.lower() == key.lower() and info[present]:
                result.evidence.append(f"Embedded generation metadata key '{present}'.")
                break

    seen: set[str] = set()
    for pattern, label in GENERATOR_PATTERNS:
        if label in seen:
            continue
        if re.search(pattern, haystack, re.IGNORECASE):
            seen.add(label)
            result.generators.append(label)
            result.evidence.append(f"Metadata references {label}.")

    if AI_SOURCE_TYPE.search(haystack):
        result.c2pa_ai_declared = True
        result.evidence.append("IPTC digitalSourceType declares algorithmic generation.")

    result.c2pa_present = _detect_c2pa(raw, info)
    if result.c2pa_present:
        result.evidence.append("C2PA content credentials container present.")
        declared, extra = _validate_c2pa(raw)
        result.evidence.extend(extra)
        if declared:
            result.c2pa_ai_declared = True

    if exif.get("Software"):
        result.evidence.append(f"EXIF Software field: {exif['Software'][:120]}")

    if result.c2pa_ai_declared:
        result.p_fake = 0.99
        result.override = True
    elif result.generators or any("generation metadata key" in e for e in result.evidence):
        result.p_fake = 0.97
        result.override = True
    elif result.camera_signature:
        result.p_fake = 0.35
        result.evidence.append(
            "Camera EXIF signature present, which mildly favours a real capture. "
            "Metadata is trivially forged, so this is weak evidence."
        )
    else:
        result.evidence.append(
            "No usable provenance metadata. This is not evidence either way, since "
            "screenshots and most social uploads strip metadata from real photos too."
        )

    return result
