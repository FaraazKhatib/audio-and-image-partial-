"""Audio decoding, windowing and aggregation tests. No weights are downloaded.

Everything here runs on waveforms built in memory with the stdlib wave module, so the suite stays
runnable on a bare install. The parts that need a real checkpoint are not tested, because a test
that downloads weights is a test that fails when the network does.

The emphasis is on the paths that fail quietly. A resample that aliases, a silence gate that drops
every window, a cap that silently analyses only the opening seconds, an aggregation that climbs with
clip length, and a label mapping that loads inverted all produce a confident number rather than an
error, which is exactly the class of bug this project keeps finding.
"""

from __future__ import annotations

import io
import sys
import types
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from veritrust.audio import (
    AudioRejected,
    AudioWindow,
    DecodedAudio,
    _pcm_from_wave,
    decode_audio,
    make_windows,
    quantile,
    resample,
    rms,
    sniff_audio_mime,
)
from veritrust.config import ALLOWED_AUDIO_MIME, Settings
from veritrust.detectors.base import LoadError
from veritrust.detectors.hf_image import _label_tokens, resolve_fake_indices


def wav_bytes(
    samples: np.ndarray, rate: int = 16000, width: int = 2, channels: int = 1
) -> bytes:
    """Encode a float waveform in -1..1 as an integer PCM WAV.

    24 bit is written by packing the low three bytes of a 32 bit little endian value, which is what
    the format actually is and what _pcm_from_wave has to sign extend back.
    """
    flat = np.asarray(samples, dtype=np.float64).reshape(-1)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(width)
        handle.setframerate(rate)
        if width == 1:
            payload = np.clip(flat * 128.0 + 128.0, 0, 255).astype(np.uint8).tobytes()
        elif width == 2:
            payload = np.clip(flat * 32767.0, -32768, 32767).astype("<i2").tobytes()
        elif width == 3:
            scaled = np.clip(flat * 8388607.0, -8388608, 8388607).astype("<i4")
            payload = scaled.view(np.uint8).reshape(-1, 4)[:, :3].tobytes()
        elif width == 4:
            payload = np.clip(flat * 2147483647.0, -2147483648, 2147483647).astype("<i4").tobytes()
        else:
            raise ValueError(f"unsupported width {width}")
        handle.writeframes(payload)
    return buffer.getvalue()


def tone(seconds: float, rate: int = 16000, freq: float = 220.0, amp: float = 0.5) -> np.ndarray:
    t = np.arange(int(seconds * rate), dtype=np.float64) / rate
    return amp * np.sin(2 * np.pi * freq * t)


def decoded_from(samples: np.ndarray, rate: int = 16000, **kwargs) -> DecodedAudio:
    return DecodedAudio(
        samples=np.asarray(samples, dtype=np.float32),
        sample_rate=rate,
        mime="audio/wav",
        raw=b"",
        original_sample_rate=kwargs.pop("original_sample_rate", rate),
        original_channels=kwargs.pop("original_channels", 1),
        decoder=kwargs.pop("decoder", "wave"),
        **kwargs,
    )


# --------------------------------------------------------------------------------------------
# Container sniffing. The image boundary matters most: a JPEG reaching the audio path would be
# refused by a decoder rather than analysed, but it would be refused with the wrong message.
# --------------------------------------------------------------------------------------------


def test_wav_requires_the_wave_form_type_not_just_riff():
    assert sniff_audio_mime(wav_bytes(tone(0.2))) == "audio/wav"
    # WebP is also a RIFF container. Matching on RIFF alone would route every WebP image into the
    # audio pathway, which is the one collision this header check exists to prevent.
    webp = b"RIFF" + (1000).to_bytes(4, "little") + b"WEBPVP8 " + b"\x00" * 32
    assert sniff_audio_mime(webp) is None


def test_jpeg_is_not_mistaken_for_a_bare_mp3_frame():
    # A bare MP3 starts FF Ex. A JPEG starts FF D8 FF, and D8 & 0xE0 is 0xC0, not 0xE0, so the sync
    # test rejects it. This is the assertion that keeps the two magic byte families disjoint.
    jpeg = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x02\x00"
    assert sniff_audio_mime(jpeg) is None

    png = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
    assert sniff_audio_mime(png) is None


def test_recognised_containers_map_to_allowed_mime_types():
    cases = {
        b"fLaC\x00\x00\x00\x22" + b"\x00" * 16: "audio/flac",
        b"OggS\x00\x02\x00\x00" + b"\x00" * 16: "audio/ogg",
        b"\x1a\x45\xdf\xa3\x01\x00\x00\x00" + b"\x00" * 16: "audio/webm",
        b"ID3\x03\x00\x00\x00\x00\x00\x00" + b"\x00" * 16: "audio/mpeg",
        b"\xff\xfb\x90\x00" + b"\x00" * 16: "audio/mpeg",
    }
    for header, expected in cases.items():
        got = sniff_audio_mime(header)
        assert got == expected, f"{header[:4]!r} sniffed as {got}, expected {expected}"
        # decode_audio checks membership separately, so a sniffer that returned something outside
        # this set would produce "Unsupported format" for a file it had just recognised.
        assert expected in ALLOWED_AUDIO_MIME


def test_any_ftyp_brand_is_offered_to_the_decoder():
    # Brand filtering was deliberately dropped: a video container with an audio stream is worth
    # analysing, and PyAV reports the absence of one. Both an m4a brand and a video brand must pass.
    for brand in (b"M4A ", b"mp42", b"isom"):
        header = b"\x00\x00\x00\x20ftyp" + brand + b"\x00" * 16
        assert sniff_audio_mime(header) == "audio/mp4", brand


def test_short_and_reserved_headers_are_refused():
    assert sniff_audio_mime(b"") is None
    assert sniff_audio_mime(b"RIFF\x00\x00\x00") is None
    # Version 1 and layer 0 are the reserved MPEG values, so this is not a real frame header even
    # though the 11 sync bits are set.
    assert sniff_audio_mime(b"\xff\xe8\x00\x00" + b"\x00" * 16) is None


# --------------------------------------------------------------------------------------------
# Stdlib PCM decode. Every supported width, because the 24 bit path sign extends by hand and the
# 8 bit path is unsigned, and both would produce plausible garbage rather than an error if wrong.
# --------------------------------------------------------------------------------------------


def test_pcm_wav_decodes_at_every_supported_width():
    reference = tone(0.25, rate=8000)
    # 8 bit carries roughly 48 dB of range, so it needs a far looser tolerance than the rest. The
    # point is that the scaling and the zero point are right, not that the quantisation is fine.
    tolerances = {1: 0.02, 2: 1e-3, 3: 1e-5, 4: 1e-5}
    for width, tolerance in tolerances.items():
        decoded = _pcm_from_wave(wav_bytes(reference, rate=8000, width=width))
        assert decoded is not None, f"width {width} returned None"
        block, rate, channels = decoded
        assert rate == 8000 and channels == 1
        assert block.shape == (len(reference), 1)
        error = float(np.max(np.abs(block[:, 0] - reference)))
        assert error < tolerance, f"width {width} peak error {error}"


def test_stereo_wav_reports_two_channels_without_flattening():
    left = tone(0.2, rate=8000, freq=200.0)
    right = tone(0.2, rate=8000, freq=400.0)
    interleaved = np.stack([left, right], axis=1).reshape(-1)
    decoded = _pcm_from_wave(wav_bytes(interleaved, rate=8000, channels=2))
    assert decoded is not None
    block, rate, channels = decoded
    assert channels == 2
    assert block.shape == (len(left), 2)
    assert float(np.max(np.abs(block[:, 0] - left))) < 1e-3
    assert float(np.max(np.abs(block[:, 1] - right))) < 1e-3


def test_a_wav_the_stdlib_cannot_read_degrades_instead_of_raising():
    # None is the contract for "not plain PCM, try a real decoder". Raising here would take out the
    # whole request for a file soundfile or PyAV could have read.
    assert _pcm_from_wave(b"RIFF" + b"\x00" * 40) is None
    assert _pcm_from_wave(b"") is None
    truncated = wav_bytes(tone(0.2))[:20]
    assert _pcm_from_wave(truncated) is None


# --------------------------------------------------------------------------------------------
# decode_audio limits and reporting.
# --------------------------------------------------------------------------------------------


def test_empty_oversize_and_unrecognised_uploads_are_rejected_with_reasons():
    settings = Settings()

    for data, fragment in (
        (b"", "Empty"),
        (b"\x00" * 64, "Unrecognised"),
    ):
        try:
            decode_audio(data, settings)
        except AudioRejected as exc:
            assert fragment.lower() in str(exc).lower(), f"{data[:8]!r} said {exc}"
        else:
            raise AssertionError(f"{data[:8]!r} was accepted")

    tiny = Settings(max_audio_bytes=1024)
    try:
        decode_audio(wav_bytes(tone(1.0)), tiny)
    except AudioRejected as exc:
        assert "limit" in str(exc)
    else:
        raise AssertionError("oversize upload was accepted")


def test_a_clip_too_short_to_read_is_refused_rather_than_scored():
    # Under 0.1 s there is not one window of anything. Returning a reading on it would be a number
    # with nothing behind it, which is the failure mode this project treats as worse than an error.
    try:
        decode_audio(wav_bytes(tone(0.02)), Settings())
    except AudioRejected as exc:
        assert "0.1 s" in str(exc)
    else:
        raise AssertionError("a 20 ms clip was accepted")


def test_stereo_is_averaged_to_mono_and_says_so():
    rate = 16000
    left = tone(0.5, rate=rate, freq=200.0)
    right = np.zeros_like(left)
    interleaved = np.stack([left, right], axis=1).reshape(-1)

    decoded = decode_audio(wav_bytes(interleaved, rate=rate, channels=2), Settings())
    assert decoded.downmixed is True
    assert decoded.original_channels == 2
    assert decoded.samples.ndim == 1
    # Averaging, not channel selection. A spoof in one channel of a stereo interview would be
    # discarded entirely by taking the first channel.
    assert float(np.max(np.abs(decoded.samples - (left / 2.0).astype(np.float32)))) < 1e-3


def test_a_long_recording_is_cut_and_reports_its_original_duration():
    settings = Settings(max_audio_seconds=1.0)
    decoded = decode_audio(wav_bytes(tone(3.0, rate=8000), rate=8000), settings)
    assert decoded.truncated is True
    assert abs(decoded.original_duration - 3.0) < 0.01
    assert abs(decoded.duration - 1.0) < 0.01
    # The engine turns truncated into a note. Reporting only the analysed duration would present a
    # one second analysis of a three second file as complete.
    assert decoded.original_duration > decoded.duration


def test_resampling_to_the_configured_rate_is_reported():
    settings = Settings(audio_sample_rate=16000)
    decoded = decode_audio(wav_bytes(tone(0.5, rate=44100), rate=44100), settings)
    assert decoded.sample_rate == 16000
    assert decoded.original_sample_rate == 44100
    assert decoded.resampled is True
    assert abs(decoded.duration - 0.5) < 0.02
    assert decoded.decoder == "wave"


def test_non_finite_samples_are_scrubbed_before_they_reach_a_model():
    # NaN survives every arithmetic step downstream and FastAPI emits it as a bare NaN token, which
    # is not valid JSON, so the browser fails before any explanation reaches the reader.
    rate = 16000
    samples = tone(0.5, rate=rate)
    raw = wav_bytes(samples, rate=rate)
    decoded = decode_audio(raw, Settings(audio_sample_rate=rate))
    assert np.all(np.isfinite(decoded.samples))

    injected = decoded_from(np.array([0.1, np.nan, np.inf, -np.inf, 0.2], dtype=np.float32))
    # rms is what the silence gate uses, so a non finite sample there would make every window
    # compare as not silent and pass through.
    assert not np.isfinite(rms(injected.samples))


# --------------------------------------------------------------------------------------------
# Resampling. The refusal is the feature: an unfiltered fallback would fabricate the exact
# artefacts these detectors treat as evidence of synthesis.
# --------------------------------------------------------------------------------------------


def test_matching_rates_skip_resampling_entirely():
    samples = tone(0.1).astype(np.float32)
    assert resample(samples, 16000, 16000) is samples


def test_resample_refuses_when_no_resampler_is_installed():
    blocked = {"soxr": None, "av": None, "av.audio.resampler": None}
    saved = {name: sys.modules.get(name) for name in blocked}
    sys.modules.update(blocked)
    try:
        resample(tone(0.1).astype(np.float32), 44100, 16000)
    except AudioRejected as exc:
        message = str(exc)
        assert "soxr" in message, message
        # The message has to say what to install and why, because "could not resample" reads as a
        # corrupt file and sends the reader looking in the wrong place.
        assert "alias" in message and "false positive" in message, message
    else:
        raise AssertionError("resampling was allowed with no resampler installed")
    finally:
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def test_an_installed_but_broken_soxr_is_reported_not_worked_around():
    # Falling through to PyAV here would work and would hide a broken install behind a path that
    # produces slightly different samples, which surfaces months later as an unreproducible score.
    def explode(*_args, **_kwargs):
        raise RuntimeError("simulated soxr failure")

    fake = types.ModuleType("soxr")
    fake.resample = explode
    saved = sys.modules.get("soxr")
    sys.modules["soxr"] = fake
    try:
        resample(tone(0.1).astype(np.float32), 44100, 16000)
    except AudioRejected as exc:
        assert "soxr" in str(exc) and "simulated soxr failure" in str(exc)
    else:
        raise AssertionError("a failing soxr was silently worked around")
    finally:
        if saved is None:
            sys.modules.pop("soxr", None)
        else:
            sys.modules["soxr"] = saved


def test_resampling_preserves_a_tone_rather_than_aliasing_it():
    # A 300 Hz tone resampled 44100 to 16000 must still be 300 Hz. Plain index striding would fold
    # anything above the new Nyquist back down, and this is the cheapest check that a real low pass
    # ran: the dominant bin has to stay put.
    rate_in, rate_out, freq = 44100, 16000, 300.0
    out = resample(tone(1.0, rate=rate_in, freq=freq).astype(np.float32), rate_in, rate_out)
    spectrum = np.abs(np.fft.rfft(out.astype(np.float64)))
    peak_hz = float(np.fft.rfftfreq(len(out), 1.0 / rate_out)[int(np.argmax(spectrum))])
    assert abs(peak_hz - freq) < 5.0, f"dominant bin moved to {peak_hz} Hz"


# --------------------------------------------------------------------------------------------
# Windowing.
# --------------------------------------------------------------------------------------------


def test_a_clip_shorter_than_one_window_yields_exactly_one_window():
    decoded = decoded_from(tone(1.5, rate=1000), rate=1000)
    windows, notes = make_windows(decoded, Settings())
    assert len(windows) == 1
    assert windows[0].start == 0.0
    assert notes == []


def test_windows_overlap_so_a_splice_cannot_fall_in_a_gap():
    rate = 1000
    decoded = decoded_from(tone(8.0, rate=rate), rate=rate)
    settings = Settings(audio_window_seconds=4.0, audio_hop_seconds=2.0)
    windows, _ = make_windows(decoded, settings)

    assert [w.start for w in windows] == [0.0, 2.0, 4.0]
    # Every instant covered twice is the property that matters, so consecutive starts must be closer
    # together than one window is long.
    for earlier, later in zip(windows, windows[1:]):
        assert later.start < earlier.end


def test_the_tail_gets_its_own_window():
    # Without this, up to hop-1 samples at the end are never scored, and the end of a file is a
    # natural place to splice something in.
    rate = 1000
    decoded = decoded_from(tone(11.0, rate=rate), rate=rate)
    windows, _ = make_windows(decoded, Settings())
    assert len(windows) == 5
    assert windows[-1].end == 11.0


def test_the_window_cap_samples_the_whole_file_not_the_opening():
    rate = 1000
    decoded = decoded_from(tone(200.0, rate=rate), rate=rate)
    settings = Settings(audio_max_windows=24)
    windows, notes = make_windows(decoded, settings)

    assert len(windows) <= 24
    # First cap windows would end at 4 + 23*2 = 50 s. Reaching the end of a 200 s file is what
    # proves the sampling is spread, which is the difference between analysing the file and
    # analysing its first 50 seconds while reporting a full analysis.
    assert windows[-1].end >= 199.0
    assert any("evenly across the whole file" in note for note in notes)


def test_near_silent_windows_are_skipped_and_counted():
    rate = 1000
    loud = tone(4.0, rate=rate, amp=0.5)
    quiet = np.zeros(int(4.0 * rate), dtype=np.float64)
    decoded = decoded_from(np.concatenate([loud, quiet]), rate=rate)

    windows, notes = make_windows(decoded, Settings())
    assert len(windows) == 2
    assert all(not w.silent for w in windows)
    assert any("near silent window" in note for note in notes)


def test_an_entirely_silent_file_still_returns_a_reading_plus_a_warning():
    # An empty result would leave the caller to interpret nothing. One window and an explicit
    # warning is more useful and does not pretend the reading means anything.
    rate = 1000
    samples = np.zeros(int(9.0 * rate), dtype=np.float64)
    samples[5000:5100] = 0.001
    decoded = decoded_from(samples, rate=rate)

    windows, notes = make_windows(decoded, Settings())
    assert len(windows) == 1
    assert windows[0].silent is False
    assert any("near silent" in note and "not be trusted" in note for note in notes)


def test_window_bounds_never_run_past_the_waveform():
    rate = 1000
    total = 7.3
    decoded = decoded_from(tone(total, rate=rate), rate=rate)
    windows, _ = make_windows(decoded, Settings())
    span = int(4.0 * rate)
    for window in windows:
        assert window.end <= total + 1e-6
        # A short final chunk would be padded by the feature extractor, but a chunk of the wrong
        # length here means the start offsets are wrong, not that the file ended.
        assert len(window.samples) == span


# --------------------------------------------------------------------------------------------
# Aggregation. A max over N windows climbs with clip length on its own, which would make a long
# real recording look more suspicious than a short one for no reason.
# --------------------------------------------------------------------------------------------


def test_quantile_interpolates_and_stays_below_the_max():
    values = [0.10, 0.20, 0.30, 0.90, 0.95]
    got = quantile(values, 0.9)
    assert abs(got - 0.93) < 1e-9, got
    assert got < max(values)


def test_quantile_edge_cases():
    assert quantile([], 0.9) == 0.0
    assert quantile([0.42], 0.9) == 0.42
    assert quantile([0.1, 0.9], 0.0) == 0.1
    assert quantile([0.1, 0.9], 1.0) == 0.9
    # Out of range q is clamped rather than raising, so a bad VT_AUDIO_QUANTILE degrades to the
    # nearest sane aggregation instead of taking out the pathway.
    assert quantile([0.1, 0.9], 5.0) == 0.9
    assert quantile([0.1, 0.9], -1.0) == 0.1


def test_the_aggregate_does_not_climb_with_clip_length_on_a_steady_signal():
    # Same distribution of per window readings, more windows. A max would drift upward here; the
    # quantile must not, because clip length is not evidence.
    rng = np.random.default_rng(1234)
    draws = rng.uniform(0.0, 0.4, size=400).tolist()
    short = quantile(draws[:8], 0.9)
    long = quantile(draws, 0.9)
    assert abs(long - short) < 0.15, (short, long)
    assert max(draws) > long


def test_a_single_loud_window_still_moves_a_high_quantile():
    # The other half of the tradeoff. A spliced segment covers few windows, so the aggregation has
    # to respond to a small number of high readings or partial spoofs are averaged away.
    calm = [0.02] * 8
    assert quantile(calm, 0.9) < 0.05
    assert quantile(calm + [0.99, 0.98], 0.9) > quantile(calm, 0.9)


# --------------------------------------------------------------------------------------------
# Label resolution for the audio checkpoints. These disagree with each other about class order, so
# resolving by index would silently invert one of them.
# --------------------------------------------------------------------------------------------


def test_camel_case_labels_resolve_by_name():
    # Hemgg/Deepfake-audio-detection ships AIVoice and HumanVoice. Without the camelCase split
    # these are single tokens, "ai" never matches, and the checkpoint is refused as unresolvable.
    # The alternative was fake_index=0, which is the index position assumption removed once already.
    assert _label_tokens("AIVoice") == ["ai", "voice"]
    assert _label_tokens("HumanVoice") == ["human", "voice"]
    fake, mapping = resolve_fake_indices({0: "AIVoice", 1: "HumanVoice"})
    assert fake == [0]
    assert "AIVoice" in mapping


def test_camel_case_splitting_does_not_change_plain_labels():
    # The image pathway shares this function, so the split has to be inert on labels that were
    # already resolving. These are the forms the declared image checkpoints actually ship.
    for label, expected in (
        ("real", ["real"]),
        ("hum", ["hum"]),
        ("fake", ["fake"]),
        ("artificial", ["artificial"]),
        ("ai_generated", ["ai", "generated"]),
        ("real_painting", ["real", "painting"]),
        ("bonafide", ["bonafide"]),
        ("spoof", ["spoof"]),
    ):
        assert _label_tokens(label) == expected, label


def test_the_audio_checkpoints_disagree_about_class_order():
    # This is why name resolution is load bearing rather than defensive. Both mappings below are
    # real, and an index based reading of one of them is inverted.
    fake_first, _ = resolve_fake_indices({0: "fake", 1: "real"})
    real_first, _ = resolve_fake_indices({0: "real", 1: "fake"})
    assert fake_first == [0]
    assert real_first == [1]


def test_asvspoof_vocabulary_resolves():
    # The ASVspoof convention is bonafide against spoof, neither of which is the word real or fake.
    # "bona" is in the real vocabulary and matches bonafide by prefix.
    fake, _ = resolve_fake_indices({0: "spoof", 1: "bonafide"})
    assert fake == [0]


def test_unresolvable_audio_labels_are_refused_rather_than_guessed():
    for mapping in ({0: "LABEL_0", 1: "LABEL_1"}, {0: "class_a", 1: "class_b"}):
        try:
            resolve_fake_indices(mapping)
        except LoadError as exc:
            assert "fake_index" in str(exc)
        else:
            raise AssertionError(f"{mapping} was resolved without evidence")


# --------------------------------------------------------------------------------------------
# Invariants the rest of the project depends on.
# --------------------------------------------------------------------------------------------


def test_audio_module_imports_without_torch():
    # Same rule as config, preprocessing, provenance and fusion. Decoding and windowing have to be
    # testable on a machine with no torch and no GPU, and the checks that run before the torch
    # import in the detector wrappers have to stay reachable.
    import importlib

    saved_torch = sys.modules.get("torch")
    saved_audio = sys.modules.pop("veritrust.audio", None)
    sys.modules["torch"] = None
    try:
        module = importlib.import_module("veritrust.audio")
        assert module.sniff_audio_mime(wav_bytes(tone(0.2))) == "audio/wav"
    finally:
        if saved_torch is None:
            sys.modules.pop("torch", None)
        else:
            sys.modules["torch"] = saved_torch
        if saved_audio is not None:
            sys.modules["veritrust.audio"] = saved_audio


def test_the_audio_wrapper_refuses_a_missing_local_directory_before_importing_torch():
    from veritrust.config import AUDIO, ModelSpec
    from veritrust.detectors.hf_audio import HFAudioClassifier

    spec = ModelSpec(
        key="missing_local_audio",
        repo=str(Path(__file__).resolve().parent / "no_such_checkpoint_dir"),
        kind=AUDIO,
        weight=1.0,
        # is_local is an explicit field rather than something inferred from the path, because a Hub
        # id and a relative path are not distinguishable by inspection. Without it this spec is
        # treated as a Hub repo and the directory check is skipped entirely.
        is_local=True,
    )
    detector = HFAudioClassifier(spec, "cpu")

    saved = sys.modules.get("torch")
    sys.modules["torch"] = None
    try:
        detector.load()
    except LoadError as exc:
        assert "is not a directory" in str(exc)
        assert "preprocessor_config.json" in str(exc)
    else:
        raise AssertionError("a missing local audio checkpoint loaded")
    finally:
        if saved is None:
            sys.modules.pop("torch", None)
        else:
            sys.modules["torch"] = saved

    assert detector.ready is False


def test_the_audio_wrapper_describes_itself_like_the_image_one():
    # /api/v1/models is one endpoint rendering one shape. A pathway reporting readiness and errors
    # under different keys would render as blank rows rather than as a degraded ensemble.
    from veritrust.config import AUDIO, SYNTHETIC, ModelSpec
    from veritrust.detectors.hf_audio import HFAudioClassifier
    from veritrust.detectors.hf_image import HFImageClassifier

    audio = HFAudioClassifier(
        ModelSpec(key="a", repo="stub/a", kind=AUDIO, weight=1.0, notes="n"), "cpu"
    )
    image = HFImageClassifier(
        ModelSpec(key="i", repo="stub/i", kind=SYNTHETIC, weight=1.0, notes="n"), "cpu"
    )
    assert set(audio.describe()) == set(image.describe())


def test_an_unloaded_audio_detector_reports_a_fault_not_a_score():
    from veritrust.config import AUDIO, ModelSpec
    from veritrust.detectors.hf_audio import HFAudioClassifier

    detector = HFAudioClassifier(ModelSpec(key="a", repo="stub/a", kind=AUDIO), "cpu")
    result = detector.predict(np.zeros(16000, dtype=np.float32), 16000)
    # p_fake None rather than 0.5. A neutral reading from a broken checkpoint would drag every
    # verdict toward the middle while looking like a working member of the ensemble.
    assert result.p_fake is None
    assert result.usable is False
    assert result.error


def test_expected_sample_rate_reads_the_extractor_rather_than_assuming():
    from veritrust.config import AUDIO, ModelSpec
    from veritrust.detectors.hf_audio import HFAudioClassifier

    detector = HFAudioClassifier(ModelSpec(key="a", repo="stub/a", kind=AUDIO), "cpu")
    assert detector.expected_sample_rate is None

    detector._processor = types.SimpleNamespace(sampling_rate=8000)
    assert detector.expected_sample_rate == 8000

    # An extractor that does not declare a rate must report None rather than a guessed 16000, since
    # verify_models compares this against the pipeline rate and would print a false mismatch.
    detector._processor = types.SimpleNamespace()
    assert detector.expected_sample_rate is None


def test_decoded_audio_and_window_shapes_are_what_the_engine_reads():
    # The engine copies these straight into image_info, and MediaOut drops anything it does not
    # declare with a 200 and no warning.
    decoded = decoded_from(tone(1.0, rate=16000))
    for name in (
        "duration",
        "original_duration",
        "sample_rate",
        "original_sample_rate",
        "original_channels",
        "downmixed",
        "truncated",
        "decoder",
        "mime",
    ):
        assert hasattr(decoded, name), name

    window = AudioWindow(index=0, start=0.0, end=4.0, rms=0.1, samples=np.zeros(4))
    for name in ("index", "start", "end", "rms", "samples", "silent"):
        assert hasattr(window, name), name


def _run_all() -> int:
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    passed, failed = 0, []
    for name, fn in tests:
        try:
            fn()
            passed += 1
            print(f"  ok    {name}")
        except Exception as exc:
            failed.append(name)
            print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")
    print(f"\n{passed} passed, {len(failed)} failed, {len(tests)} total")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
