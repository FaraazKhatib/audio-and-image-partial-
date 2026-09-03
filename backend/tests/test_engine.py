"""Engine orchestration tests using stub detectors in place of real checkpoints.

This covers the wiring that is otherwise only exercised on a machine with the weights present:
signal assembly, per detector failure isolation, face aggregation, override propagation, registry
deduplication and the response shape. The stubs return fixed scores, so nothing here says anything
about accuracy.
"""

from __future__ import annotations

import ast
import io
import json
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image
from PIL.PngImagePlugin import PngInfo

from veritrust.config import AUDIO as AUDIO_KIND
from veritrust.config import FACE, SYNTHETIC, ModelSpec, Settings
from veritrust.detectors import registry as registry_module
from veritrust.detectors.base import DetectorResult, LoadError
from veritrust.detectors.hf_image import HFImageClassifier, describe_checkpoint
from veritrust.detectors.registry import Registry
from veritrust.detectors.torch_image import TorchImageClassifier
from veritrust.engine import Engine
from veritrust.faces import Face, FaceDetection


@dataclass
class StubDetector:
    """Stands in for HFImageClassifier. Same surface the engine actually touches."""

    key: str
    kind: str
    score: float | None = 0.5
    error: str | None = None
    weight: float = 1.0
    calls: int = 0

    @property
    def spec(self) -> ModelSpec:
        return ModelSpec(key=self.key, repo=f"stub/{self.key}", kind=self.kind, weight=self.weight)

    def predict(self, *args, **kwargs) -> DetectorResult:
        self.calls += 1
        return DetectorResult(
            key=self.key,
            kind=self.kind,
            p_fake=self.score,
            detail="stub",
            error=self.error,
            latency_ms=0.0,
        )


class StubFaces:
    def __init__(self, faces=(), backend="stub", available=True, note=""):
        self.faces = list(faces)
        self.backend = backend
        self.available = available
        self.note = note

    def load(self) -> None:
        return None

    def detect(self, image) -> FaceDetection:
        return FaceDetection(self.faces, self.backend, self.available, self.note)


def png(width=256, height=256, meta=None) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (140, 110, 80)).save(buffer, format="PNG", pnginfo=meta)
    return buffer.getvalue()


def build_engine(synthetic=(), face=(), faces_stub=None) -> Engine:
    """Bypasses load() so no checkpoint is ever fetched."""
    engine = Engine(Settings())
    engine.registry.synthetic = list(synthetic)
    engine.registry.face = list(face)
    engine.face_detector = faces_stub or StubFaces()
    return engine


def analyse(engine: Engine, data: bytes):
    return engine.analyze(data, want_heatmap=False)


def test_ffpp_checkpoint_uses_its_published_fake_class_index():
    """Xicor9's model card is 0=real, 1=fake; an inversion looks valid but reverses every result."""
    detector = TorchImageClassifier(
        ModelSpec(key="ffpp", repo="Xicor9/efficientnet-b0-ffpp-c23", kind=FACE)
    )
    assert detector.label_mapping == {"0": "real", "1": "fake"}
    assert detector._fake_index == 1


def test_two_agreeing_detectors_produce_ai_verdict():
    engine = build_engine(
        synthetic=[StubDetector("a", SYNTHETIC, 0.93), StubDetector("b", SYNTHETIC, 0.88)]
    )
    result = analyse(engine, png())
    assert result.verdict == "likely_ai_generated"
    assert [s["name"] for s in result.signals] == ["a", "b"]
    assert result.errors == []
    assert result.heatmap is None
    assert "heatmap" not in result.timing_ms


def test_uncalibrated_low_image_readings_abstain_by_default():
    engine = build_engine(
        synthetic=[StubDetector("a", SYNTHETIC, 0.04), StubDetector("b", SYNTHETIC, 0.09)]
    )
    result = analyse(engine, png())
    assert result.verdict == "uncertain"
    assert result.score_ai < 0.35, "the diagnostic score is retained for later calibration"
    assert result.confidence == 0.0
    assert any("Held at uncertain" in note for note in result.notes)


def test_failed_detector_is_isolated_not_fatal():
    broken = StubDetector("broken", SYNTHETIC, None, error="CUDA out of memory")
    working = StubDetector("working", SYNTHETIC, 0.91)
    result = analyse(engine := build_engine(synthetic=[broken, working]), png())

    assert [s["name"] for s in result.signals] == ["working"]
    assert result.errors == [{"detector": "broken", "error": "CUDA out of memory"}]
    assert result.verdict == "likely_ai_generated"
    assert engine.registry.synthetic[0].calls == 1


def test_no_detectors_abstains_and_says_why():
    result = analyse(build_engine(), png())
    assert result.verdict == "uncertain"
    assert result.signals == []
    assert any("No detection model loaded" in note for note in result.notes)


def test_provenance_override_beats_confident_models():
    meta = PngInfo()
    meta.add_text("parameters", "cinematic portrait, Steps: 28, Sampler: Euler a")
    engine = build_engine(
        synthetic=[StubDetector("a", SYNTHETIC, 0.02), StubDetector("b", SYNTHETIC, 0.05)]
    )
    result = analyse(engine, png(meta=meta))

    assert result.overridden_by == "provenance"
    assert result.verdict == "likely_ai_generated"
    assert result.score_ai > 0.9
    names = [s["name"] for s in result.signals]
    assert "provenance" in names and "a" in names, "model signals stay visible under an override"


def test_absent_metadata_adds_no_signal():
    engine = build_engine(synthetic=[StubDetector("a", SYNTHETIC, 0.5)])
    result = analyse(engine, png())
    assert "provenance" not in [s["name"] for s in result.signals]
    assert result.provenance["p_fake"] is None


def test_face_pathway_takes_the_worst_face():
    faces = StubFaces([Face((20, 20, 90, 90), 0.99), Face((140, 140, 200, 200), 0.95)])
    engine = build_engine(
        synthetic=[StubDetector("whole", SYNTHETIC, 0.30)],
        face=[StubDetector("face_vit", FACE, 0.87, weight=1.2)],
        faces_stub=faces,
    )
    result = analyse(engine, png())

    assert len(result.faces) == 2
    assert [f["index"] for f in result.faces] == [0, 1]
    assert all(f["p_ai"] == 0.87 for f in result.faces)

    face_signal = next(s for s in result.signals if s["name"] == "face_pathway")
    assert face_signal["p_ai"] == 0.87
    assert "Highest of 2 face crop(s)" in face_signal["detail"]
    assert engine.registry.face[0].calls == 2, "one call per detected face"


def test_decisive_face_crop_overrides_the_whole_image_verdict():
    """End to end wiring for the face override, not just the fusion arithmetic.

    The engine has to emit the face signal with kind FACE and the field has to survive into the
    response, otherwise the setting is unreachable from a real request. The box the UI paints and the
    verdict come from the same reading here, which is the behaviour that was asked for. Synthetic
    signals must be above authentic_max so the corroboration guard does not fire.
    """
    faces = StubFaces([Face((20, 20, 90, 90), 0.99)])
    engine = build_engine(
        synthetic=[StubDetector("a", SYNTHETIC, 0.50), StubDetector("b", SYNTHETIC, 0.45)],
        face=[StubDetector("face_vit", FACE, 0.96, weight=1.0)],
        faces_stub=faces,
    )
    engine.settings.fusion.face_decides = True
    result = analyse(engine, png())

    assert result.overridden_by == "face_pathway"
    assert result.verdict == "likely_ai_generated"
    assert result.score_ai == 0.96
    assert result.faces[0]["p_ai"] == 0.96, "the box and the verdict read the same number"
    assert result.calibrated is False


def test_multiple_face_detectors_are_averaged_per_face():
    faces = StubFaces([Face((20, 20, 90, 90), 0.99)])
    engine = build_engine(
        synthetic=[StubDetector("whole", SYNTHETIC, 0.5)],
        face=[StubDetector("f1", FACE, 0.9), StubDetector("f2", FACE, 0.5)],
        faces_stub=faces,
    )
    result = analyse(engine, png())
    assert result.faces[0]["p_ai"] == 0.7


def test_faces_present_but_no_face_model_loaded():
    """Dropping the pathway silently reads exactly like an image with no faces in it."""
    faces = StubFaces([Face((20, 20, 90, 90), 0.99)])
    engine = build_engine(synthetic=[StubDetector("whole", SYNTHETIC, 0.8)], faces_stub=faces)
    result = analyse(engine, png())
    assert result.faces == []
    assert "face_pathway" not in [s["name"] for s in result.signals]
    assert any("no face model is loaded" in note for note in result.notes)
    assert not any("No faces found" in note for note in result.notes)


def test_no_faces_found_is_stated():
    engine = build_engine(synthetic=[StubDetector("whole", SYNTHETIC, 0.8)])
    result = analyse(engine, png())
    assert any("No faces found" in note for note in result.notes)


def test_unavailable_face_backend_reports_itself():
    """Silence here would be indistinguishable from a clean result."""
    faces = StubFaces([], backend="none", available=False, note="OpenCV missing.")
    engine = build_engine(synthetic=[StubDetector("whole", SYNTHETIC, 0.8)], faces_stub=faces)
    result = analyse(engine, png())
    assert "OpenCV missing." in result.notes
    assert not any("No faces found" in note for note in result.notes)


def test_downscale_is_disclosed():
    settings = Settings()
    settings.max_edge = 200
    engine = Engine(settings)
    engine.registry.synthetic = [StubDetector("a", SYNTHETIC, 0.8)]
    engine.face_detector = StubFaces()
    result = engine.analyze(png(900, 600), want_heatmap=False)

    assert result.image_info["downscaled"] is True
    assert result.image_info["original_width"] == 900
    assert any("downscaled" in note for note in result.notes)


def test_response_dict_contract():
    engine = build_engine(
        synthetic=[StubDetector("a", SYNTHETIC, 0.8)],
        face=[StubDetector("f", FACE, 0.6)],
        faces_stub=StubFaces([Face((20, 20, 90, 90), 0.9)]),
    )
    payload = analyse(engine, png()).as_dict()

    expected = {
        "verdict", "score_ai", "confidence", "confidence_meaning", "calibrated", "signals",
        "notes", "provenance", "faces", "heatmap", "image", "timing_ms", "overridden_by",
        "escalated_by", "logit_spread", "spread_exceeds_limit", "errors",
    }
    assert expected == set(payload)
    assert payload["calibrated"] is False
    assert 0.0 <= payload["score_ai"] <= 1.0
    assert "not a probability" in payload["confidence_meaning"]

    for signal in payload["signals"]:
        assert {"name", "p_ai", "weight", "kind", "clamped", "counted"} <= set(signal)
    for face in payload["faces"]:
        assert {"x", "y", "w", "h", "score", "index", "p_ai"} <= set(face)
    for key in ("decode", "provenance", "synthetic", "faces", "fusion"):
        assert key in payload["timing_ms"]


def test_every_response_key_is_declared_in_the_schema():
    """FastAPI filters the payload through AnalyzeResponse, so an undeclared key never reaches the
    browser. It fails quietly: the request still returns 200 and the field is simply gone, which
    looks like a frontend bug rather than a missing declaration.

    schemas.py is parsed rather than imported because importing it needs pydantic, and these tests
    deliberately run without it.
    """
    source = (Path(__file__).resolve().parent.parent / "veritrust" / "schemas.py").read_text(
        encoding="utf-8"
    )
    model = next(
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.ClassDef) and node.name == "AnalyzeResponse"
    )
    declared = {
        node.target.id
        for node in model.body
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
    }

    engine = build_engine(synthetic=[StubDetector("a", SYNTHETIC, 0.8)])
    payload = analyse(engine, png()).as_dict()

    undeclared = set(payload) - declared
    assert not undeclared, f"emitted but not in AnalyzeResponse, so dropped: {sorted(undeclared)}"


def test_status_reports_without_any_model_loaded():
    status = build_engine().status()
    assert status["ensemble_size"] == 0
    assert status["calibrated"] is False
    assert status["thresholds"]["ai_min"] > status["thresholds"]["authentic_max"]
    assert "device" in status


class StubLoader:
    """Stands in for HFImageClassifier so registry wiring can be tested without weights.

    `resolves` maps a spec key to the repo its load would settle on, or None to fail.
    """

    resolves: dict[str, str | None] = {}

    def __init__(self, spec, device, dtype):
        self.spec = spec
        self.repo_used = None

    def load(self):
        target = self.resolves.get(self.spec.key)
        if target is None:
            raise LoadError(f"{self.spec.key} unavailable")
        self.repo_used = target


def build_registry(resolves, specs):
    StubLoader.resolves = resolves
    original = registry_module.HFImageClassifier
    registry_module.HFImageClassifier = StubLoader
    try:
        reg = Registry(Settings())
        return reg, reg._build(specs)
    finally:
        registry_module.HFImageClassifier = original


def spec(key: str, repo: str, fallbacks=()) -> ModelSpec:
    return ModelSpec(key=key, repo=repo, kind=SYNTHETIC, fallbacks=fallbacks)


def test_two_specs_resolving_to_one_checkpoint_load_once():
    """config.py gives broad_swin a fallback that is sdxl_swin's primary repo.

    If the primary is unavailable both keys resolve to the same weights, and without this check
    one opinion would be counted twice at double weight while reporting perfect agreement.
    """
    specs = (spec("a", "repo/one"), spec("b", "repo/two", ("repo/one",)))
    reg, loaded = build_registry({"a": "repo/one", "b": "repo/one"}, specs)

    assert [d.spec.key for d in loaded] == ["a"]
    assert len(reg.failures) == 1
    assert reg.failures[0]["key"] == "b"
    assert "already loaded as a" in reg.failures[0]["error"]


def test_distinct_checkpoints_both_load():
    specs = (spec("a", "repo/one"), spec("b", "repo/two"))
    reg, loaded = build_registry({"a": "repo/one", "b": "repo/two"}, specs)

    assert [d.spec.key for d in loaded] == ["a", "b"]
    assert reg.failures == []


def test_dedup_does_not_mask_a_plain_load_failure():
    specs = (spec("a", "repo/one"), spec("b", "repo/two"))
    reg, loaded = build_registry({"a": "repo/one", "b": None}, specs)

    assert [d.spec.key for d in loaded] == ["a"]
    assert reg.failures[0]["key"] == "b"
    assert "unavailable" in reg.failures[0]["error"]


def test_face_pathway_may_share_a_checkpoint_with_the_whole_image_pathway():
    """Dedup is per pathway on purpose: the same model on a face crop is a separate observation."""
    reg, synthetic = build_registry({"a": "repo/shared"}, (spec("a", "repo/shared"),))
    StubLoader.resolves = {"f": "repo/shared"}
    original = registry_module.HFImageClassifier
    registry_module.HFImageClassifier = StubLoader
    try:
        face = reg._build((ModelSpec(key="f", repo="repo/shared", kind=FACE),))
    finally:
        registry_module.HFImageClassifier = original

    assert [d.spec.key for d in synthetic] == ["a"]
    assert [d.spec.key for d in face] == ["f"]


def local_spec(key: str, path: str, kind: str = SYNTHETIC) -> ModelSpec:
    return ModelSpec(key=key, repo=path, kind=kind, is_local=True)


def test_two_local_specs_pointing_at_one_directory_load_once():
    """Nothing stops the same checkpoint being declared twice under different names."""
    specs = (local_spec("a", "/models/detect"), local_spec("b", "/models/detect"))
    reg, loaded = build_registry({"a": "/models/detect", "b": "/models/detect"}, specs)

    assert [d.spec.key for d in loaded] == ["a"]
    assert len(reg.failures) == 1 and reg.failures[0]["key"] == "b"


def test_local_spec_config_problems_reach_the_models_endpoint():
    """A typo in models.local.json must be visible, not swallowed at import time."""
    saved = registry_module.LOCAL_SPEC_PROBLEMS
    registry_module.LOCAL_SPEC_PROBLEMS = ["models.local.json entry 0 was skipped: bad kind"]
    try:
        reg = Registry(Settings())
    finally:
        registry_module.LOCAL_SPEC_PROBLEMS = saved

    assert len(reg.failures) == 1
    assert reg.failures[0]["kind"] == "config"
    assert "bad kind" in reg.failures[0]["error"]
    assert reg.status()["failures"] == reg.failures


def test_missing_local_directory_fails_before_touching_torch():
    """The message has to name the path, not report a Hub lookup for something that is a path.

    torch is blocked for the duration so the assertion holds whether or not it is installed. This
    runs the real HFImageClassifier, which is only possible because the check sits ahead of the
    torch import.
    """
    detector = HFImageClassifier(local_spec("detect_world", "/no/such/checkpoint"), "cpu", "auto")
    saved = sys.modules.get("torch", "absent")
    sys.modules["torch"] = None  # makes `import torch` raise ImportError
    try:
        detector.load()
    except LoadError as exc:
        message = str(exc)
    except ImportError as exc:
        raise AssertionError(f"the path check must run before importing torch: {exc}") from exc
    else:
        raise AssertionError("a nonexistent local directory must not load")
    finally:
        if saved == "absent":
            del sys.modules["torch"]
        else:
            sys.modules["torch"] = saved

    assert "/no/such/checkpoint" in message
    assert "is not a directory" in message
    assert "config.json" in message, "the message should say what a checkpoint directory needs"
    assert not detector.ready


def test_describe_checkpoint_names_the_architecture():
    """A private checkpoint that is not an image classifier should say so in its failure."""
    with tempfile.TemporaryDirectory() as folder:
        Path(folder, "config.json").write_text(
            json.dumps(
                {
                    "model_type": "qwen2_vl",
                    "architectures": ["Qwen2VLForConditionalGeneration"],
                    "id2label": {},
                }
            ),
            encoding="utf-8",
        )
        described = describe_checkpoint(folder)

    assert "qwen2_vl" in described
    assert "Qwen2VLForConditionalGeneration" in described
    assert "image classification head" in described


def test_describe_checkpoint_is_silent_when_it_cannot_help():
    with tempfile.TemporaryDirectory() as folder:
        assert describe_checkpoint(folder) == "", "no config.json means nothing to report"
        Path(folder, "config.json").write_text("{ not json", encoding="utf-8")
        assert describe_checkpoint(folder) == "", "a broken config must not raise during a failure"


def build_audio_engine(detectors=(), settings=None) -> Engine:
    """An engine with only the audio pathway populated. Bypasses load(), so nothing is fetched."""
    engine = Engine(settings or Settings())
    engine.registry.synthetic = []
    engine.registry.face = []
    engine.registry.audio = list(detectors)
    engine.face_detector = StubFaces()
    return engine


def speech_wav(seconds: float = 12.0, rate: int = 16000) -> bytes:
    """Real WAV bytes rather than a mocked decoder.

    The previous version of this test patched decode_audio and make_windows, which meant it asserted
    that a mock had been called and exercised none of the routing, decoding or windowing it looked
    like it covered. audio.py imports no torch and decodes PCM with the stdlib, so there is no
    reason to mock any of it.
    """
    from test_audio import tone, wav_bytes

    return wav_bytes(tone(seconds, rate=rate), rate=rate)


def test_audio_bytes_route_to_the_audio_pathway_on_their_header_alone():
    from veritrust.config import AUDIO

    engine = build_audio_engine([StubDetector("audio_model", AUDIO, 0.8)])
    result = analyse(engine, speech_wav())

    assert result.verdict == "likely_ai_generated"
    assert [s["kind"] for s in result.signals] == [AUDIO]
    assert result.signals[0]["name"] == "audio_model"
    # The shared fields an image result would fill have to be present and empty, not absent, or the
    # frontend has to test which modality it got before reading anything.
    assert result.faces == [] and result.heatmap is None
    assert result.provenance["exif_present"] is False
    assert result.image_info["mime"] == "audio/wav"
    assert result.image_info["windows_scored"] > 1, "12 s must produce more than one window"


def test_an_image_still_routes_to_the_image_pathway():
    # The other half of the routing contract. A PNG must not reach the audio path, and the audio
    # sniffer is the only thing standing between them.
    engine = build_engine(synthetic=[StubDetector("a", SYNTHETIC, 0.9)])
    engine.registry.audio = [StubDetector("audio_model", AUDIO_KIND, 0.1)]
    result = analyse(engine, png())

    assert [s["name"] for s in result.signals] == ["a"]
    assert result.image_info["mime"] == "image/png"
    assert "windows_scored" not in result.image_info


def test_windowing_and_fusion_notes_both_reach_the_response():
    # Both sources were being lost independently. make_windows returns the notes about coverage, and
    # _analyze_audio started from that list, but it dropped fuse()'s own notes entirely, so the
    # uncalibrated warning and the discarded reading notice never reached the reader on this path.
    from veritrust.config import AUDIO

    engine = build_audio_engine(
        [StubDetector("good", AUDIO, 0.9), StubDetector("broken", AUDIO, float("nan"))]
    )
    result = analyse(engine, speech_wav())

    assert any("uncalibrated" in note for note in result.notes), "fusion notes were dropped"
    assert any("broken" in note for note in result.notes), "the faulty reading was not named"
    # NaN is not valid JSON, so it has to be gone rather than merely unused.
    json.dumps(result.as_dict())


def test_a_long_recording_reports_that_it_was_cut():
    from veritrust.config import AUDIO

    engine = build_audio_engine(
        [StubDetector("a", AUDIO, 0.9)], settings=Settings(max_audio_seconds=2.0)
    )
    result = analyse(engine, speech_wav(seconds=8.0))

    assert result.image_info["truncated"] is True
    assert result.image_info["original_duration"] > result.image_info["duration"]
    assert any("was not examined" in note for note in result.notes)


def test_a_broken_audio_checkpoint_reports_once_not_once_per_window():
    # A checkpoint fails the same way on every window. One row per window presented a single dead
    # checkpoint as two dozen separate faults.
    from veritrust.config import AUDIO

    broken = StubDetector("bad", AUDIO, None, error="CUDA out of memory")
    engine = build_audio_engine([StubDetector("a", AUDIO, 0.9), broken])
    result = analyse(engine, speech_wav())

    assert len(result.errors) == 1, result.errors
    assert result.errors[0]["detector"] == "bad"
    assert broken.calls == result.image_info["windows_scored"], "every window should still be tried"
    assert [s["name"] for s in result.signals] == ["a"]


def test_audio_upload_with_no_model_loaded_says_so():
    engine = build_audio_engine([])
    result = analyse(engine, speech_wav())

    assert result.verdict == "uncertain"
    assert any("No audio detection model loaded" in note for note in result.notes)


def test_audio_disabled_by_configuration_is_disclosed():
    # A degradation path that is otherwise invisible: the response looks like a normal abstention.
    engine = build_audio_engine([], settings=Settings(enable_audio=False))
    result = analyse(engine, speech_wav())

    assert any("switched off by configuration" in note for note in result.notes)
    assert any("VT_ENABLE_AUDIO" in note for note in result.notes)


def test_a_lone_audio_dissenter_floors_the_verdict_without_moving_the_score():
    # Same rule as the image pathway, and it has to hold here too: these checkpoints fall back to
    # real on anything unfamiliar, so a positive is strong evidence and silence is weak.
    from veritrust.config import AUDIO

    engine = build_audio_engine(
        [
            StubDetector("a", AUDIO, 0.02),
            StubDetector("b", AUDIO, 0.03),
            StubDetector("loud", AUDIO, 0.97),
        ]
    )
    result = analyse(engine, speech_wav())

    assert result.verdict == "uncertain"
    assert result.escalated_by == "loud"
    # The score is what the eval harness ranks and what calibration is fitted against, so the floor
    # must not touch it.
    assert result.score_ai < 0.35, result.score_ai
    assert result.confidence == 0.0


def test_audio_metadata_is_reported_as_unread_rather_than_as_clean():
    engine = build_audio_engine([StubDetector("a", AUDIO_KIND, 0.9)])
    result = analyse(engine, speech_wav())

    # An empty ProvenanceResult and a missing one look identical to the frontend, so the note is the
    # only thing distinguishing "checked, found nothing" from "not checked".
    assert result.provenance["c2pa_present"] is False
    assert any("Container metadata was not examined" in note for note in result.notes)


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
