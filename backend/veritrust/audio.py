"""Audio decoding and windowing. Numpy only, no torch, no pydantic.

The audio analogue of preprocessing.py, and it follows the same rule: no per model feature
extraction happens here. Each audio checkpoint owns an AutoFeatureExtractor that knows the mel
filterbank, normalisation and padding it was trained with, and hand rolling any of that here is the
audio form of the divide by 255 bug this project already fixed once on the image side. What this
module produces is one mono float32 waveform at a known sample rate, plus the windows to score.

Four decisions here are load bearing.

Decoding is tiered, stdlib first. PCM WAV is read with the `wave` module and numpy, which needs no
third party decoder at all, so the most common upload works on a bare install and the whole path is
testable without soundfile present. soundfile handles WAV variants, FLAC and Ogg, and MP3 on a
recent libsndfile. PyAV handles the MP4 family, which matters because an iPhone voice memo is m4a
and libsndfile cannot read it. A missing backend is reported as a missing decoder rather than as an
unsupported file, because those need different fixes.

Resampling is anti-aliased, never plain decimation. These detectors read exactly the high frequency
texture that naive index-stride resampling folds back into the signal, so a bad resample manufactures
the artefacts they were trained to call spoofed. soxr does it properly, PyAV's resampler is the
fallback, and if neither is installed the file is refused with the reason rather than silently
aliased into a false positive.

Analysis is windowed. A partial spoof, meaning a real recording with a few seconds of cloned speech
spliced in, is the realistic attack, and scoring a five minute file as one clip averages those
seconds into nothing. Windows are 4 seconds with a 2 second hop, so every instant is covered twice
and a splice cannot fall in a gap. Long files are capped by sampling windows evenly across the whole
recording instead of taking the first N, since truncating to the opening seconds would miss anything
later and report full coverage.

Silent windows are skipped. Room tone contains no voice for a voice spoofing model to read, and
feeding it in produces an arbitrary reading that then joins the aggregate as though it were
evidence. If every window is silent the loudest is kept anyway, so the caller gets a reading plus a
note rather than an empty result it has to interpret.
"""

from __future__ import annotations

import io
import wave
from dataclasses import dataclass

import numpy as np

from .config import ALLOWED_AUDIO_MIME, Settings

MP3_MIME = "audio/mpeg"

# Magic bytes only. A client supplied content type is never trusted, exactly as in preprocessing.py.
_MAGIC = (
    (b"fLaC", "audio/flac"),
    (b"OggS", "audio/ogg"),
    (b"\x1a\x45\xdf\xa3", "audio/webm"),
    (b"ID3", MP3_MIME),
)

# ftyp brands that mean "this MP4 container holds audio we should try". Kept for reference and for
# the download-free tests; the sniffer accepts any ftyp box, see sniff_audio_mime.
_MP4_AUDIO_BRANDS = (b"M4A ", b"M4B ", b"mp42", b"mp41", b"isom", b"iso2", b"dash", b"qt  ")


class AudioRejected(ValueError):
    """Raised when input is unusable, unsupported, or outside configured safety limits."""


@dataclass
class DecodedAudio:
    """One mono waveform ready to hand to a feature extractor.

    `raw` is kept for the same reason DecodedImage keeps it: provenance must read the original
    container bytes, not a re-encode of the samples.
    """

    samples: np.ndarray
    sample_rate: int
    mime: str
    raw: bytes
    original_sample_rate: int
    original_channels: int
    decoder: str
    resampled: bool = False
    downmixed: bool = False
    truncated: bool = False
    original_duration: float = 0.0

    @property
    def duration(self) -> float:
        return round(len(self.samples) / max(self.sample_rate, 1), 3)


@dataclass
class AudioWindow:
    index: int
    start: float
    end: float
    rms: float
    samples: np.ndarray
    silent: bool = False


def sniff_audio_mime(data: bytes) -> str | None:
    """Identify a container from its header. Returns None when nothing matches."""
    if len(data) < 12:
        return None

    if data[:4] == b"RIFF" and data[8:12] == b"WAVE":
        return "audio/wav"
    # Any ftyp box is reported as audio/mp4 regardless of brand. The brand distinguishes M4A from
    # MP4 video, but a video container with an audio stream is a legitimate thing to analyse and
    # PyAV reports the absence of one, so filtering on brand here would refuse files that decode
    # perfectly while adding nothing a decoder attempt does not already establish.
    if data[4:8] == b"ftyp":
        return "audio/mp4"
    for magic, mime in _MAGIC:
        if data.startswith(magic):
            return mime

    # A bare MP3 with no ID3 tag starts at a frame header. Two bytes of sync plus a version and
    # layer that are not the reserved values, which is enough to tell it from arbitrary binary
    # without pulling in a parser.
    if data[0] == 0xFF and (data[1] & 0xE0) == 0xE0:
        version = (data[1] >> 3) & 0x03
        layer = (data[1] >> 1) & 0x03
        if version != 1 and layer != 0:
            return MP3_MIME
    return None


def _pcm_from_wave(data: bytes) -> tuple[np.ndarray, int, int] | None:
    """Decode integer PCM WAV with the stdlib. Returns (samples, sample_rate, channels) or None.

    None means "not plain PCM, try a real decoder", which covers ADPCM, mu-law, float WAV and the
    extensible header. Those all reach soundfile instead. Nothing here raises, so an odd WAV
    degrades to the next tier rather than failing the request.
    """
    try:
        with wave.open(io.BytesIO(data), "rb") as handle:
            channels = handle.getnchannels()
            width = handle.getsampwidth()
            rate = handle.getframerate()
            frames = handle.readframes(handle.getnframes())
    except Exception:
        return None

    if channels < 1 or rate < 1 or not frames:
        return None

    if width == 1:
        # 8 bit WAV is unsigned by definition, so it centres on 128 rather than 0.
        raw = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif width == 2:
        raw = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    elif width == 3:
        # 24 bit has no numpy dtype. Widen each sample to 4 bytes and sign extend by hand.
        packed = np.frombuffer(frames, dtype=np.uint8)
        usable = (len(packed) // 3) * 3
        triples = packed[:usable].reshape(-1, 3).astype(np.uint32)
        value = triples[:, 0] | (triples[:, 1] << 8) | (triples[:, 2] << 16)
        signed = np.where(value >= 0x800000, value.astype(np.int64) - 0x1000000, value)
        raw = signed.astype(np.float32) / 8388608.0
    elif width == 4:
        raw = np.frombuffer(frames, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        return None

    usable = (len(raw) // channels) * channels
    return raw[:usable].reshape(-1, channels), rate, channels


def _decode_with_soundfile(data: bytes) -> tuple[np.ndarray, int, int] | None:
    try:
        import soundfile
    except ImportError:
        return None
    try:
        block, rate = soundfile.read(io.BytesIO(data), dtype="float32", always_2d=True)
    except Exception:
        return None
    if block.size == 0:
        return None
    return block, int(rate), int(block.shape[1])


def _decode_with_av(data: bytes) -> tuple[np.ndarray, int, int] | None:
    """PyAV path, needed for the MP4 family. Bundled ffmpeg, so no system install is required."""
    try:
        import av
    except ImportError:
        return None

    try:
        with av.open(io.BytesIO(data)) as container:
            stream = next((s for s in container.streams if s.type == "audio"), None)
            if stream is None:
                return None
            rate = int(stream.rate or 0)
            channels = 0
            chunks: list[np.ndarray] = []
            for frame in container.decode(stream):
                block = frame.to_ndarray()
                # PyAV gives planar formats as (channels, samples) and packed as (1, samples *
                # channels). frame.layout is the authority on how many channels are in there.
                count = len(frame.layout.channels)
                channels = channels or count
                if block.ndim == 1:
                    block = block.reshape(1, -1)
                if block.shape[0] == count:
                    block = block.T
                else:
                    block = block.reshape(-1, count)
                chunks.append(np.asarray(block, dtype=np.float32))
                rate = rate or int(frame.sample_rate or 0)
    except Exception:
        return None

    if not chunks or not rate:
        return None
    stacked = np.concatenate(chunks, axis=0)
    if stacked.size == 0:
        return None

    # Integer sample formats arrive as integers. Scale by the width actually seen rather than
    # assuming 16 bit, since s32 would otherwise come out 65536 times too loud.
    peak = float(np.max(np.abs(stacked))) if stacked.size else 0.0
    if peak > 1.5:
        divisor = 2147483648.0 if peak > 32768.0 else 32768.0
        stacked = stacked / divisor
    return stacked, rate, channels or stacked.shape[1]


def resample(samples: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    """Anti-aliased resample of a mono float waveform.

    soxr first, then PyAV's swresample. Both apply a proper low pass filter. There is deliberately
    no numpy fallback: dropping or repeating samples aliases the high frequency content these
    detectors read, which fabricates the artefacts they treat as evidence of synthesis, and a
    fabricated positive is worse than a refusal that says what to install.
    """
    if source_rate == target_rate:
        return samples

    try:
        import soxr
    except ImportError:
        soxr = None

    if soxr is not None:
        # A soxr that is installed but fails is reported rather than allowed to fall through to
        # PyAV. Falling through would work, but it would hide a broken install behind a slower
        # path that produces slightly different samples, which is the sort of difference that only
        # shows up as an unreproducible score months later.
        try:
            return np.asarray(soxr.resample(samples, source_rate, target_rate), dtype=np.float32)
        except Exception as exc:
            raise AudioRejected(
                f"soxr could not resample this file from {source_rate} Hz to {target_rate} Hz: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

    try:
        import av
        from av.audio.resampler import AudioResampler
    except ImportError as exc:
        raise AudioRejected(
            f"This file is {source_rate} Hz and has to be resampled to {target_rate} Hz, but no "
            f"resampler is installed. Install soxr (pip install soxr). Resampling without one "
            f"would alias the high frequency detail these detectors read and could turn a real "
            f"recording into a false positive."
        ) from exc

    try:
        layout = "mono"
        frame = av.AudioFrame.from_ndarray(
            samples.reshape(1, -1).astype(np.float32), format="flt", layout=layout
        )
        frame.sample_rate = source_rate
        resampler = AudioResampler(format="flt", layout=layout, rate=target_rate)
        out = [f.to_ndarray().reshape(-1) for f in resampler.resample(frame)]
        flushed = resampler.resample(None)
        out.extend(f.to_ndarray().reshape(-1) for f in flushed)
    except Exception as exc:
        raise AudioRejected(f"Could not resample this file to {target_rate} Hz: {exc}") from exc

    if not out:
        raise AudioRejected(f"Resampling to {target_rate} Hz produced no samples.")
    return np.concatenate(out).astype(np.float32)


def decode_audio(data: bytes, settings: Settings) -> DecodedAudio:
    if not data:
        raise AudioRejected("Empty upload.")
    if len(data) > settings.max_audio_bytes:
        limit_mb = settings.max_audio_bytes / (1024 * 1024)
        raise AudioRejected(f"File exceeds the {limit_mb:.0f} MB limit.")

    mime = sniff_audio_mime(data)
    if mime is None:
        raise AudioRejected("Unrecognised audio format. WAV, FLAC, Ogg, MP3, M4A or WebM.")
    if mime not in ALLOWED_AUDIO_MIME:
        raise AudioRejected(f"Unsupported format: {mime}")

    decoded = None
    decoder = ""
    for name, attempt in (
        ("wave", _pcm_from_wave),
        ("soundfile", _decode_with_soundfile),
        ("av", _decode_with_av),
    ):
        decoded = attempt(data)
        if decoded is not None:
            decoder = name
            break

    if decoded is None:
        raise AudioRejected(
            f"Could not decode this {mime} file. WAV, FLAC and Ogg need soundfile; MP3 needs a "
            f"recent libsndfile; M4A, AAC and WebM need PyAV (pip install av). The file may also "
            f"simply be truncated."
        )

    block, source_rate, channels = decoded
    if block.ndim == 1:
        block = block.reshape(-1, 1)
    if block.shape[0] == 0:
        raise AudioRejected("This file contains no audio samples.")

    original_duration = round(block.shape[0] / max(source_rate, 1), 3)

    # Averaging channels is the right downmix here. Taking one channel would discard half the
    # evidence on a stereo interview, and these detectors read a single stream regardless.
    downmixed = block.shape[1] > 1
    mono = block.mean(axis=1).astype(np.float32) if downmixed else block[:, 0].astype(np.float32)

    truncated = False
    limit = int(settings.max_audio_seconds * source_rate)
    if limit > 0 and len(mono) > limit:
        mono = mono[:limit]
        truncated = True

    mono = np.nan_to_num(mono, nan=0.0, posinf=0.0, neginf=0.0)

    target_rate = int(settings.audio_sample_rate)
    resampled = source_rate != target_rate
    mono = resample(mono, source_rate, target_rate)

    if len(mono) < target_rate // 10:
        raise AudioRejected(
            f"This clip is {len(mono) / max(target_rate, 1):.2f} s long. Under 0.1 s there is "
            f"nothing for a speech model to read."
        )

    return DecodedAudio(
        samples=mono,
        sample_rate=target_rate,
        mime=mime,
        raw=data,
        original_sample_rate=source_rate,
        original_channels=channels,
        decoder=decoder,
        resampled=resampled,
        downmixed=downmixed,
        truncated=truncated,
        original_duration=original_duration,
    )


def rms(samples: np.ndarray) -> float:
    if samples.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(samples.astype(np.float64)))))


def make_windows(decoded: DecodedAudio, settings: Settings) -> tuple[list[AudioWindow], list[str]]:
    """Split a waveform into the windows to score, plus notes about what was dropped.

    Returns windows already filtered by the silence gate. The notes exist because every reduction
    here changes what the verdict is based on, and a capped or gated run that reported nothing would
    look like full coverage.
    """
    notes: list[str] = []
    rate = max(decoded.sample_rate, 1)
    total = len(decoded.samples)

    span = max(1, int(settings.audio_window_seconds * rate))
    hop = max(1, int(settings.audio_hop_seconds * rate))

    starts: list[int] = []
    if total <= span:
        starts = [0]
    else:
        position = 0
        while position + span <= total:
            starts.append(position)
            position += hop
        # The tail is worth its own window. Without this, up to hop-1 samples at the end are never
        # scored, and the end of a file is a natural place to splice something in.
        if starts and starts[-1] + span < total:
            starts.append(total - span)

    cap = max(1, int(settings.audio_max_windows))
    if len(starts) > cap:
        # Evenly spaced across the whole recording, not the first cap windows. Truncating to the
        # opening would miss a splice at minute four and still report a full analysis.
        picked = np.linspace(0, len(starts) - 1, cap).round().astype(int)
        kept = [starts[i] for i in sorted(set(int(i) for i in picked))]
        notes.append(
            f"This recording is {decoded.duration:.0f} s, longer than {cap} windows covers at the "
            f"configured hop, so {len(kept)} window(s) were sampled evenly across the whole file "
            f"rather than taken from the start. Coverage is spread out, not continuous."
        )
        starts = kept

    windows: list[AudioWindow] = []
    for index, start in enumerate(starts):
        chunk = decoded.samples[start : start + span]
        level = rms(chunk)
        windows.append(
            AudioWindow(
                index=index,
                start=round(start / rate, 3),
                end=round(min(start + span, total) / rate, 3),
                rms=round(level, 6),
                samples=chunk,
                silent=level < settings.audio_min_rms,
            )
        )

    audible = [w for w in windows if not w.silent]
    if not audible and windows:
        loudest = max(windows, key=lambda w: w.rms)
        loudest.silent = False
        audible = [loudest]
        notes.append(
            "Every window in this file is near silent. The loudest was scored anyway so there is a "
            "reading to show, but there is very little for a speech model to work with here and the "
            "result should not be trusted."
        )
    elif len(audible) < len(windows):
        skipped = len(windows) - len(audible)
        notes.append(
            f"Skipped {skipped} near silent window(s). Room tone carries no voice, so a reading on "
            f"it would be noise entering the aggregate as though it were evidence."
        )

    return audible, notes


def quantile(values: list[float], q: float) -> float:
    """Linear interpolated quantile of a small list. Kept here so it is testable without numpy dtype
    noise and so the aggregation rule sits next to the windowing it applies to.

    The audio pathway aggregates each model's per window readings with this rather than a max. A max
    over N windows from an uncalibrated model takes the highest of N draws, so it climbs with clip
    length on its own: a long real recording would look more suspicious than a short one for no
    reason other than having more windows. A high quantile still responds to a short spliced segment
    while needing more than one window to agree.
    """
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    ordered = sorted(float(v) for v in values)
    q = min(max(q, 0.0), 1.0)
    position = q * (len(ordered) - 1)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)
