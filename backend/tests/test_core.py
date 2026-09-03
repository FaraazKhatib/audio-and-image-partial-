"""Tests for the torch free core: preprocessing, provenance, fusion and label resolution.

These modules are deliberately dependency light so they can be tested without downloading a
single checkpoint. Runs under pytest, or directly with `python tests/test_core.py`.
"""

from __future__ import annotations

import io
import json
import math
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from veritrust import provenance
from veritrust.config import (
    BUILTIN_MODELS,
    FACE,
    FAKE_LABEL_TOKENS,
    REAL_LABEL_TOKENS,
    SYNTHETIC,
    VERDICT_AI,
    VERDICT_AUTHENTIC,
    VERDICT_UNCERTAIN,
    FusionConfig,
    ModelSpec,
    Settings,
    Thresholds,
    load_local_specs,
)
from veritrust.detectors.base import LoadError
from veritrust.detectors.hf_image import resolve_fake_indices
from veritrust.faces import yunet_file_problem
from veritrust.fusion import Calibration, Signal, clamp_probability, fuse, logit, sigmoid
from veritrust.preprocessing import (
    ImageRejected,
    crop_with_margin,
    decode_image,
    sniff_mime,
)

PROVENANCE = "provenance"


def make_image(width=64, height=64, colour=(120, 90, 60)) -> Image.Image:
    return Image.new("RGB", (width, height), colour)


def to_bytes(image: Image.Image, fmt="PNG", **kwargs) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format=fmt, **kwargs)
    return buffer.getvalue()


def fresh_settings(**overrides) -> Settings:
    settings = Settings()
    for key, value in overrides.items():
        setattr(settings, key, value)
    return settings


# ---------- preprocessing ----------


def test_sniff_mime_by_magic_bytes():
    assert sniff_mime(to_bytes(make_image(), "PNG")) == "image/png"
    assert sniff_mime(to_bytes(make_image(), "JPEG")) == "image/jpeg"
    assert sniff_mime(to_bytes(make_image(), "WEBP")) == "image/webp"
    assert sniff_mime(b"not an image at all") is None


def test_decode_rejects_bad_input():
    settings = fresh_settings()
    for payload, reason in [(b"", "empty"), (b"\x00\x01\x02\x03", "garbage")]:
        try:
            decode_image(payload, settings)
        except ImageRejected:
            continue
        raise AssertionError(f"{reason} input should have been rejected")


def test_decode_rejects_oversize_file():
    settings = fresh_settings(max_upload_bytes=10)
    try:
        decode_image(to_bytes(make_image()), settings)
    except ImageRejected as exc:
        assert "limit" in str(exc).lower()
        return
    raise AssertionError("oversize file should have been rejected")


def test_decode_rejects_pixel_bomb():
    settings = fresh_settings(max_pixels=100)
    try:
        decode_image(to_bytes(make_image(64, 64)), settings)
    except ImageRejected as exc:
        assert "MP limit" in str(exc)
        return
    raise AssertionError("pixel count above the limit should have been rejected")


def test_decode_downscales_large_edge():
    settings = fresh_settings(max_edge=128)
    decoded = decode_image(to_bytes(make_image(600, 300)), settings)
    assert decoded.downscaled is True
    assert max(decoded.width, decoded.height) == 128
    assert decoded.original_width == 600 and decoded.original_height == 300
    assert abs(decoded.width / decoded.height - 2.0) < 0.05


def test_decode_leaves_small_image_alone():
    decoded = decode_image(to_bytes(make_image(200, 100)), fresh_settings())
    assert decoded.downscaled is False
    assert (decoded.width, decoded.height) == (200, 100)
    assert decoded.image.mode == "RGB"


def test_decode_applies_exif_orientation():
    """Orientation 6 means rotate 90 degrees. Ignoring it feeds a sideways face to the model."""
    image = make_image(100, 50)
    exif = image.getexif()
    exif[0x0112] = 6
    data = to_bytes(image, "JPEG", exif=exif)
    decoded = decode_image(data, fresh_settings())
    assert (decoded.width, decoded.height) == (50, 100), "orientation tag was not applied"


def test_crop_with_margin_expands_and_clamps():
    image = make_image(100, 100)
    crop = crop_with_margin(image, (40, 40, 60, 60), 0.5)
    assert crop.size == (40, 40)

    edge = crop_with_margin(image, (0, 0, 20, 20), 0.5)
    assert edge.size == (30, 30), "margin should clamp at the image boundary"


# ---------- provenance ----------


def test_provenance_flags_stable_diffusion_parameters_chunk():
    meta = PngInfo()
    meta.add_text("parameters", "a portrait, Steps: 30, Sampler: DPM++ 2M, Model: sd_xl_base_1.0")
    data = to_bytes(make_image(), "PNG", pnginfo=meta)

    result = provenance.inspect(data)
    assert result.p_fake is not None and result.p_fake > 0.9
    assert result.override is True
    assert any("generation metadata" in line for line in result.evidence)


def test_provenance_flags_generator_named_in_exif_software():
    image = make_image()
    exif = image.getexif()
    exif[0x0131] = "Midjourney v6"
    result = provenance.inspect(to_bytes(image, "JPEG", exif=exif))
    assert "Midjourney" in result.generators
    assert result.override is True


def test_provenance_detects_iptc_digital_source_type():
    meta = PngInfo()
    meta.add_text(
        "XML:com.adobe.xmp",
        '<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF><rdf:Description '
        'Iptc4xmpExt:digitalSourceType="http://cv.iptc.org/newscodes/digitalsourcetype/'
        'trainedAlgorithmicMedia"/></rdf:RDF></x:xmpmeta>',
    )
    result = provenance.inspect(to_bytes(make_image(), "PNG", pnginfo=meta))
    assert result.c2pa_ai_declared is True
    assert result.p_fake == 0.99


def test_provenance_camera_exif_leans_real_but_weakly():
    image = make_image()
    exif = image.getexif()
    exif[0x010F] = "Canon"
    exif[0x0110] = "EOS R6"
    exif[0x9003] = "2026:03:14 11:02:41"
    result = provenance.inspect(to_bytes(image, "JPEG", exif=exif))
    assert result.camera_signature is True
    assert result.p_fake == 0.35
    assert result.override is False
    assert any("trivially forged" in line for line in result.evidence)


def test_provenance_has_no_opinion_without_metadata():
    """The asymmetry that matters. A stripped file must not read as suspicious."""
    result = provenance.inspect(to_bytes(make_image(), "PNG"))
    assert result.p_fake is None
    assert result.override is False
    assert any("not evidence either way" in line for line in result.evidence)


def test_provenance_ignores_bytes_outside_metadata_regions():
    """A whole file byte scan would flag this. Only parsed metadata should count."""
    data = to_bytes(make_image(), "PNG") + b"midjourney stable diffusion parameters"
    result = provenance.inspect(data)
    assert result.generators == []
    assert result.p_fake is None


# ---------- label resolution ----------


def test_resolve_labels_both_orders():
    fake, _ = resolve_fake_indices({0: "fake", 1: "real"})
    assert fake == [0]
    fake, _ = resolve_fake_indices({0: "real", 1: "fake"})
    assert fake == [1], "index position must not be assumed"


def test_resolve_labels_alternate_vocabulary():
    fake, _ = resolve_fake_indices({0: "human", 1: "artificial"})
    assert fake == [1]
    fake, _ = resolve_fake_indices({0: "AI_GENERATED", 1: "authentic"})
    assert fake == [0]


def test_resolve_labels_multiclass_sums_generated_classes():
    fake, _ = resolve_fake_indices({0: "real photo", 1: "fake sdxl", 2: "deepfake swap"})
    assert sorted(fake) == [1, 2]


def test_resolve_labels_refuses_to_guess_generic_labels():
    """The old code assumed index 0 was fake. Refusing beats guessing."""
    for labels in (
        {0: "LABEL_0", 1: "LABEL_1"},
        {0: "class_a", 1: "class_b"},
        {0: "0", 1: "1"},
    ):
        try:
            resolve_fake_indices(labels)
        except LoadError:
            continue
        raise AssertionError(f"{labels} should not resolve")


def test_resolve_labels_from_the_four_configured_checkpoints():
    """The exact id2label maps the configured repos report, captured from a real load on 2026-08-21.

    These are the only label sets known to be real rather than invented, so they are the ones worth
    pinning. A repo changing its labels should break here rather than at request time.
    """
    observed = {
        "sdxl_swin": ({0: "artificial", 1: "human"}, [0]),
        "broad_swin": ({0: "artificial", 1: "real"}, [0]),
        "siglip_ai_human": ({0: "ai", 1: "hum"}, [0]),
        "face_vit": ({0: "Real", 1: "Fake"}, [1]),
    }
    for key, (labels, expected) in observed.items():
        fake, mapping = resolve_fake_indices(labels)
        assert fake == expected, f"{key}: {labels} resolved to {fake}, expected {expected}"
        assert mapping, "the mapping is printed by verify_models.py and must not be empty"


def test_resolve_labels_matches_a_truncated_label():
    """Ateeqq/ai-vs-human-image-detector labels its real class "hum", not "human".

    Exact token matching refused that checkpoint outright, which is how one of four detectors went
    missing while the ensemble reported itself merely degraded.
    """
    fake, _ = resolve_fake_indices({0: "ai", 1: "hum"})
    assert fake == [0]
    fake, _ = resolve_fake_indices({0: "synth", 1: "nat"})
    assert fake == [0], "prefixes should work on either side of the comparison"


def test_resolve_labels_ignores_a_vocabulary_word_buried_in_another_word():
    """"ai" appears inside "painting". Substring matching anywhere would invert this pair."""
    fake, _ = resolve_fake_indices({0: "real_painting", 1: "ai_painting"})
    assert fake == [1]


def test_resolve_labels_refuses_a_label_reading_both_ways():
    """A negation the matcher cannot parse must not be resolved by list search order."""
    for labels in ({0: "ai-generated", 1: "not-ai"}, {0: "real", 1: "real or fake"}):
        try:
            resolve_fake_indices(labels)
        except LoadError:
            continue
        raise AssertionError(f"{labels} should not resolve")


def test_resolve_labels_names_what_it_did_not_recognise():
    """The failure has to be actionable enough to write a fake_index from."""
    try:
        resolve_fake_indices({0: "class_a", 1: "class_b"})
    except LoadError as exc:
        message = str(exc)
    else:
        raise AssertionError("should not resolve")

    assert "class_a" in message and "class_b" in message
    assert "fake_index" in message


def test_label_vocabularies_contain_no_bare_digits():
    """A digit in the vocabulary is index position in disguise.

    "0" and "1" sat in these tuples as dead entries, live only because the matcher filtered short
    tokens. Reviving them would resolve LABEL_0 as real and LABEL_1 as fake with nothing behind it.
    """
    for vocabulary in (FAKE_LABEL_TOKENS, REAL_LABEL_TOKENS):
        assert not [t for t in vocabulary if t.isdigit()]


def test_forced_fake_index_bypasses_label_resolution():
    """The escape hatch has to work for checkpoints whose labels genuinely cannot be read."""
    spec = ModelSpec(key="k", repo="r", kind=SYNTHETIC, fake_index=1)
    assert spec.fake_index == 1, "fake_index must survive on a frozen spec"


# ---------- fusion ----------


def test_logit_sigmoid_roundtrip():
    for probability in (0.01, 0.25, 0.5, 0.75, 0.99):
        assert abs(sigmoid(logit(probability)) - probability) < 1e-9


def test_logit_clamps_extremes():
    assert logit(0.0) < -10 and logit(1.0) > 10


def make_signal(name, p, kind=SYNTHETIC, weight=1.0, override=False):
    return Signal(name=name, p_fake=p, weight=weight, kind=kind, override=override)


def test_fuse_agreeing_signals_sharpen():
    thresholds, cfg = Thresholds(), FusionConfig()
    result = fuse([make_signal("a", 0.8), make_signal("b", 0.85)], thresholds, cfg)
    assert result.verdict == VERDICT_AI
    assert result.score > 0.8
    assert result.calibrated is False


def test_fuse_lands_in_uncertain_band():
    thresholds, cfg = Thresholds(), FusionConfig()
    result = fuse([make_signal("a", 0.5), make_signal("b", 0.5)], thresholds, cfg)
    assert result.verdict == VERDICT_UNCERTAIN
    assert result.confidence == 0.0


def test_fuse_authentic_side():
    result = fuse([make_signal("a", 0.05), make_signal("b", 0.1)], Thresholds(), FusionConfig())
    assert result.verdict == VERDICT_AUTHENTIC
    assert result.confidence > 0


def test_fuse_disagreement_reduces_confidence():
    thresholds, cfg = Thresholds(), FusionConfig()
    agree = fuse([make_signal("a", 0.9), make_signal("b", 0.92)], thresholds, cfg)
    disagree = fuse([make_signal("a", 0.99), make_signal("b", 0.42)], thresholds, cfg)
    assert disagree.confidence < agree.confidence
    assert any("disagree" in note for note in disagree.notes)


def test_clamp_probability_bounds():
    assert clamp_probability(0.0, 0.02) == 0.02
    assert clamp_probability(1.0, 0.02) == 0.98
    assert clamp_probability(0.5, 0.02) == 0.5
    assert clamp_probability(0.001, 0.0) == 0.001


def test_clamp_is_reported_per_signal():
    result = fuse([make_signal("loud", 0.001), make_signal("quiet", 0.3)], Thresholds(), FusionConfig())
    by_name = {row["name"]: row for row in result.signals}
    assert by_name["loud"]["clamped"] is True
    assert by_name["quiet"]["clamped"] is False
    assert by_name["loud"]["p_ai"] == 0.001, "the raw reading must still be reported"
    assert any("Capped" in note for note in result.notes)


def test_one_overconfident_member_cannot_veto_the_ensemble():
    """The regression this clamp exists for.

    Observed in the field: two whole image models read 0.219 and 0.001 on a generated image and
    fusion returned 0.0165 with a near maximal margin. Logit space is unbounded, so the 0.001
    contributed -6.9 and no realistic opposition could offset it. Adding a face model that
    correctly reads 0.85 has to be able to move the result.

    Worth noting what this measures: the clamp alone lifts 0.19 to 0.33, which is still inside
    the authentic band. The clamp reopens the vote, the escalation rule is what actually stops
    the wrong verdict, and neither is sufficient alone.

    face_decides is off throughout, because a face reading of 0.85 now takes the verdict outright
    and would settle this before either mechanism ran. That path is covered separately. These two
    are what remain load bearing when the face reading is undecided or the setting is turned off.
    """
    thresholds = Thresholds()
    signals = [
        make_signal("sdxl_swin", 0.219),
        make_signal("broad_swin", 0.001),
        make_signal("siglip", 0.60, weight=0.8),
        make_signal("face_pathway", 0.85, kind=FACE),
    ]

    unclamped = fuse(
        signals,
        thresholds,
        FusionConfig(signal_clamp=0.0, dissent_escalates=False, face_decides=False),
    )
    clamped = fuse(signals, thresholds, FusionConfig(dissent_escalates=False, face_decides=False))
    both = fuse(signals, thresholds, FusionConfig(face_decides=False))

    assert unclamped.verdict == VERDICT_AUTHENTIC, "documents the old broken behaviour"
    assert clamped.score > unclamped.score * 1.5
    assert both.verdict == VERDICT_UNCERTAIN
    assert both.escalated_by == "face_pathway"


def test_lone_dissenter_escalates_to_uncertain():
    """The floor with face_decides off, which is the only state where it applies to a face."""
    thresholds, cfg = Thresholds(), FusionConfig(face_decides=False)
    signals = [
        make_signal("a", 0.03),
        make_signal("b", 0.05),
        make_signal("face_pathway", 0.88, kind=FACE),
    ]
    result = fuse(signals, thresholds, cfg)
    assert result.verdict == VERDICT_UNCERTAIN
    assert result.escalated_by == "face_pathway"
    assert result.confidence == 0.0
    assert any("Held at uncertain" in note for note in result.notes)


def test_escalation_leaves_the_score_untouched():
    """The score is what the eval harness ranks and calibrates on, so it must not be patched."""
    thresholds = Thresholds()
    signals = [make_signal("a", 0.03), make_signal("b", 0.88)]
    escalating = fuse(signals, thresholds, FusionConfig())
    plain = fuse(signals, thresholds, FusionConfig(dissent_escalates=False))
    assert escalating.score == plain.score
    assert escalating.verdict != plain.verdict


def test_escalation_can_be_turned_off():
    signals = [make_signal("a", 0.02), make_signal("b", 0.9)]
    result = fuse(signals, Thresholds(), FusionConfig(dissent_escalates=False))
    assert result.escalated_by is None


def test_escalation_is_a_floor_not_a_promotion():
    """A single dissenter can stop an authentic call. It must not manufacture an AI call."""
    signals = [make_signal("a", 0.02), make_signal("b", 0.99)]
    result = fuse(signals, Thresholds(), FusionConfig())
    assert result.verdict == VERDICT_UNCERTAIN

    agreeing = fuse([make_signal("a", 0.9), make_signal("b", 0.95)], Thresholds(), FusionConfig())
    assert agreeing.verdict == VERDICT_AI
    assert agreeing.escalated_by is None


def test_provenance_override_is_not_clamped():
    """Provenance is read evidence, not an estimate, so the clamp must not soften it."""
    signals = [make_signal(PROVENANCE, 0.99, kind=PROVENANCE, override=True)]
    result = fuse(signals, Thresholds(), FusionConfig())
    assert result.score == 0.99


def test_disagreement_is_measured_in_odds_not_probability():
    """0.219 against 0.001 is 0.22 apart in probability and a factor of 280 apart in raw odds.

    The old probability space check used a 0.35 tolerance, so this pair passed as agreement and the
    result was reported with a near maximal margin. Spread is measured after clamping, so the figure
    the note actually prints for this pair is 14, not 280. Both members still lean the same way, so
    the correct outcome is a reduced margin rather than an abstention.
    """
    cfg = FusionConfig(dissent_escalates=False)
    split = fuse([make_signal("a", 0.219), make_signal("b", 0.001)], Thresholds(), cfg)
    tight = fuse([make_signal("a", 0.03), make_signal("b", 0.02)], Thresholds(), cfg)

    assert abs(0.219 - 0.001) < 0.35, "would have passed the old probability space check"
    assert any("disagree" in note for note in split.notes)
    assert not any("disagree" in note for note in tight.notes)
    assert split.confidence < tight.confidence


def test_spread_is_reported_not_only_narrated():
    """The frontend needs the figure, and pattern matching the note text for it is not a contract.

    Confidence reaches zero two unrelated ways, from sitting inside the band and from members
    contradicting each other, and the UI has to tell those apart to avoid printing "0%" at the reader
    when one member is in fact certain.
    """
    cfg = FusionConfig(dissent_escalates=False)
    split = fuse([make_signal("a", 0.219), make_signal("b", 0.001)], Thresholds(), cfg)
    tight = fuse([make_signal("a", 0.03), make_signal("b", 0.02)], Thresholds(), cfg)

    assert split.spread_exceeds_limit is True
    assert tight.spread_exceeds_limit is False
    assert split.logit_spread > tight.logit_spread
    assert round(math.exp(split.logit_spread)) == 14, "the post clamp figure, not the raw 280"
    assert f"{math.exp(split.logit_spread):.0f}" in " ".join(split.notes)


def test_reported_spread_is_the_post_clamp_figure():
    """Clamping is what makes 280 into 14, and the note and the field must not disagree."""
    signals = [make_signal("a", 0.219), make_signal("b", 0.001)]
    loose = fuse(signals, Thresholds(), FusionConfig(signal_clamp=0.0005, dissent_escalates=False))
    clamped = fuse(signals, Thresholds(), FusionConfig(signal_clamp=0.02, dissent_escalates=False))
    assert loose.logit_spread > clamped.logit_spread


def test_paths_that_never_compute_a_spread_report_zero():
    """An override and an empty ensemble have no spread, and 0 is honest there rather than stale."""
    override = fuse(
        [make_signal(PROVENANCE, 0.99, kind=PROVENANCE, override=True)],
        Thresholds(),
        FusionConfig(),
    )
    empty = fuse([], Thresholds(), FusionConfig())
    for result in (override, empty):
        assert result.logit_spread == 0.0
        assert result.spread_exceeds_limit is False


def test_spread_fields_reach_the_payload():
    """as_dict is the API contract, so a field added to the dataclass alone would be invisible."""
    result = fuse(
        [make_signal("a", 0.219), make_signal("b", 0.001)],
        Thresholds(),
        FusionConfig(dissent_escalates=False),
    )
    payload = result.as_dict()
    assert payload["spread_exceeds_limit"] is True
    assert payload["logit_spread"] > 0


def test_face_swap_dilution_figures_quoted_in_the_readme():
    """README states what a third whole image detector does to a face swap score. Pin the numbers.

    Two wrong figures were published here before, because the claim was arithmetic done by hand and
    nothing recomputed it. It uses the shipped spec weights rather than literals so that retuning a
    weight fails this test instead of quietly making the README wrong again.

    These are the figures fusion produces with face_decides off, which is why that setting exists.
    The last assertion pins that the default now falls through to fusion plus escalation when the
    whole image signals all read authentic, since the corroboration guard prevents the face pathway
    from overriding when synthetics strongly disagree.
    """
    specs = {spec.key: spec for spec in BUILTIN_MODELS}
    thresholds = Thresholds()
    cfg = FusionConfig(face_decides=False)
    face = make_signal("face_pathway", 1.00, kind=FACE, weight=specs["face_vit"].weight)
    whole = [
        make_signal(key, 0.02, weight=specs[key].weight)
        for key in ("sdxl_swin", "broad_swin", "siglip_ai_human")
    ]

    two = fuse([face] + whole[:2], thresholds, cfg)
    three = fuse([face] + whole, thresholds, cfg)

    assert round(two.score, 3) == 0.274
    assert round(three.score, 3) == 0.174
    assert three.score < two.score, "a third whole image vote moves a face swap toward authentic"
    for result in (two, three):
        assert result.verdict == VERDICT_UNCERTAIN, "escalation holds the verdict up as the score sinks"
        assert result.escalated_by == "face_pathway"
        assert round(math.exp(result.logit_spread)) == 2401

    # With face_decides on, the corroboration guard fires because the whole image signals all read
    # below authentic_max, so fusion plus escalation runs rather than face override.
    decided = fuse([face] + whole, thresholds, FusionConfig(face_decides=True))
    assert decided.overridden_by is None, "corroboration guard prevents face override here"
    assert decided.escalated_by == "face_pathway"
    assert decided.verdict == VERDICT_UNCERTAIN


def test_decisive_face_takes_the_verdict_toward_ai():
    """The case this setting was added for: a swapped face against whole image reads that are not
    strongly authentic. The corroboration guard only blocks when a synthetic detector reads below
    authentic_max, so uncertain whole image readings let the face pathway decide.

    Fusion would dilute the score here. The score reported is the face reading itself, because a
    needle sitting deep in the authentic zone under a headline saying AI generated is a
    contradiction, and the fused value is not the number that produced the verdict.
    """
    face = make_signal("face_pathway", 0.97, kind=FACE)
    whole = [make_signal("a", 0.50), make_signal("b", 0.45), make_signal("c", 0.50, weight=0.8)]
    result = fuse([face] + whole, Thresholds(), FusionConfig(face_decides=True))

    assert result.verdict == VERDICT_AI
    assert result.score == 0.97
    assert result.overridden_by == "face_pathway"
    assert result.escalated_by is None, "an override does not also escalate"
    assert result.calibrated is False
    assert any("Verdict taken from face_pathway alone" in note for note in result.notes)
    assert any("Overruled 3 whole image reading(s)" in note for note in result.notes)


def test_decisive_face_takes_the_verdict_toward_real_and_admits_the_cost():
    """The direction the user accepted knowingly, so it must be tested and stated, not implied.

    A green face closes the case even over a whole image detector past ai_min. That is the one
    situation where this build calls an image real over a detector that flagged it, and the note has
    to say so rather than leaving the reader to notice the overruled row on their own.
    """
    face = make_signal("face_pathway", 0.04, kind=FACE)
    loud = make_signal("sdxl_swin", 0.97)
    result = fuse([face, loud], Thresholds(), FusionConfig(face_decides=True))

    assert result.verdict == VERDICT_AUTHENTIC
    assert result.score == 0.04
    assert result.overridden_by == "face_pathway"
    assert any("cleared the 0.65 AI threshold and was overruled" in note for note in result.notes)
    assert any("sdxl_swin at 0.97" in note for note in result.notes)


def test_face_reading_inside_the_band_does_not_decide():
    """An undecided face has nothing to hand down, so it falls through to ordinary fusion."""
    signals = [make_signal("face_pathway", 0.5, kind=FACE), make_signal("a", 0.02)]
    result = fuse(signals, Thresholds(), FusionConfig())
    assert result.overridden_by is None
    assert result.score < 0.35


def test_provenance_outranks_a_decisive_face():
    """Read evidence beats an estimate, however confident the estimate is."""
    signals = [
        make_signal("face_pathway", 0.02, kind=FACE),
        make_signal(PROVENANCE, 0.99, kind=PROVENANCE, override=True),
    ]
    result = fuse(signals, Thresholds(), FusionConfig())
    assert result.overridden_by == PROVENANCE
    assert result.verdict == VERDICT_AI


def test_muted_face_weight_cannot_decide():
    """A zero weight means ignore this detector, so handing it the whole verdict inverts the order."""
    signals = [make_signal("a", 0.02), make_signal("face_pathway", 0.99, weight=0.0, kind=FACE)]
    result = fuse(signals, Thresholds(), FusionConfig())
    assert result.overridden_by is None
    assert result.verdict == VERDICT_AUTHENTIC


def test_face_decides_is_disabled_by_default():
    """A raw, uncalibrated face model must not own ordinary portrait verdicts by default."""
    signals = [
        make_signal("a", 0.02),
        make_signal("b", 0.02),
        make_signal("face_pathway", 0.99, kind=FACE),
    ]
    result = fuse(signals, Thresholds(), FusionConfig())
    assert result.overridden_by is None
    assert result.escalated_by == "face_pathway"
    assert result.verdict == VERDICT_UNCERTAIN
    assert round(result.score, 3) == 0.274, "the fused score the floor is holding up"


def test_a_deciding_face_never_reports_uncertain():
    """The invariant that keeps the headline coherent with overridden_by.

    _decisive_face and Thresholds.verdict must test the same comparisons against the same number. If
    they ever drift, the response says a single check decided this while the verdict says nobody
    could tell, and the frontend has no way to render that honestly.
    """
    thresholds = Thresholds()
    for reading in (0.0, 0.01, 0.2, 0.35, 0.65, 0.8, 1.0):
        result = fuse(
            [make_signal("face_pathway", reading, kind=FACE), make_signal("a", 0.5)],
            thresholds,
            FusionConfig(face_decides=True),
        )
        assert result.overridden_by == "face_pathway", f"{reading} should be decisive"
        assert result.verdict != VERDICT_UNCERTAIN, f"{reading} decided but reported uncertain"
        assert result.score == reading

    for reading in (0.36, 0.5, 0.64):
        result = fuse(
            [make_signal("face_pathway", reading, kind=FACE), make_signal("a", 0.5)],
            thresholds,
            FusionConfig(),
        )
        assert result.overridden_by is None, f"{reading} is inside the band and must fuse"


def test_face_override_reports_the_spread_it_overruled():
    """Confidence is band distance, so it cannot show the disagreement. The fields have to.

    Companion synthetic signal must be above authentic_max so the corroboration guard does not fire.
    """
    signals = [make_signal("face_pathway", 1.0, kind=FACE), make_signal("a", 0.50)]
    result = fuse(signals, Thresholds(), FusionConfig(face_decides=True))
    assert result.spread_exceeds_limit is True
    assert result.logit_spread > 0
    assert any("this overruled disagree by a factor of" in note for note in result.notes)

    agreeing = fuse(
        [make_signal("face_pathway", 0.97, kind=FACE), make_signal("a", 0.96)],
        Thresholds(),
        FusionConfig(),
    )
    assert agreeing.spread_exceeds_limit is False


def test_face_override_stays_uncalibrated_even_with_a_fitted_calibration():
    """calibrate.py drops records with overridden_by set, and this path must land in that filter.

    The score here is one model's raw reading rather than the fitted expression, so a record marked
    calibrated would teach the next fit a relationship that never ran. Companion synthetic signal
    above authentic_max so the corroboration guard does not fire.
    """
    calibration = Calibration(
        bias=0.0, weights={"a": 2.0, "face_pathway": 2.0}, fitted=True, source="unit"
    )
    signals = [make_signal("a", 0.50), make_signal("face_pathway", 0.9, kind=FACE)]
    result = fuse(signals, Thresholds(), FusionConfig(face_decides=True), calibration)
    assert result.calibrated is False
    assert result.overridden_by == "face_pathway"
    assert result.score == 0.9
    assert result.as_dict()["overridden_by"] == "face_pathway"


def test_only_the_deciding_face_is_marked_as_counted():
    """The overruled rows are still shown, and showing them as contributing would be a lie.

    Companion synthetic signal above authentic_max so the corroboration guard does not fire.
    """
    signals = [make_signal("a", 0.50), make_signal("face_pathway", 0.9, kind=FACE)]
    result = fuse(signals, Thresholds(), FusionConfig(face_decides=True))
    by_name = {row["name"]: row for row in result.signals}
    assert by_name["face_pathway"]["counted"] is True
    assert by_name["a"]["counted"] is False
    assert by_name["a"]["p_ai"] == 0.50, "the overruled reading is still reported as it was read"


def test_face_override_vetoed_by_authentic_synthetic():
    """The corroboration guard: a face reading AI plus a synthetic reading authentic falls through.

    This is the false positive case. The face model is confident, but a whole image detector strongly
    reads authentic (below authentic_max), which is evidence the face model is wrong rather than
    evidence of a swap. A real face swap does not alter pixels outside the face region, so a whole
    image detector has no reason to read authentic on one.

    The guard blocks the face override so fusion runs normally. The fused score may still be high
    when other signals also read high, but the important property is that the face pathway no longer
    decides alone.
    """
    face = make_signal("face_pathway", 0.997, kind=FACE)
    authentic_whole = make_signal("siglip_ai_human", 0.001)
    uncertain_whole = make_signal("sdxl_swin", 0.59)
    flagging_whole = make_signal("broad_swin", 0.956)
    signals = [face, authentic_whole, uncertain_whole, flagging_whole]

    result = fuse(signals, Thresholds(), FusionConfig())
    assert result.overridden_by is None, "corroboration guard must block face override"

    # When only the face and one authentic whole image detector are present, the fused score
    # should not land in AI territory because the ensemble is split.
    small = fuse(
        [face, make_signal("a", 0.001), make_signal("b", 0.05)],
        Thresholds(),
        FusionConfig(),
    )
    assert small.overridden_by is None, "corroboration guard fires with authentic synthetics"
    assert small.score < 0.65, "face plus two authentic whole image reads should not fuse to AI"


def test_face_override_authentic_direction_ignores_corroboration():
    """The authentic direction keeps no corroboration requirement.

    A low face reading on a flagged whole image is evidence of the same kind, not a contradiction,
    so the guard only applies in the AI direction.
    """
    face = make_signal("face_pathway", 0.04, kind=FACE)
    flagging_whole = make_signal("a", 0.02)
    result = fuse([face, flagging_whole], Thresholds(), FusionConfig(face_decides=True))
    assert result.overridden_by == "face_pathway"
    assert result.verdict == VERDICT_AUTHENTIC


def test_fuse_provenance_override_wins_outright():
    thresholds, cfg = Thresholds(), FusionConfig()
    signals = [
        make_signal("model_a", 0.02),
        make_signal("model_b", 0.03),
        make_signal(PROVENANCE, 0.99, kind=PROVENANCE, override=True),
    ]
    result = fuse(signals, thresholds, cfg)
    assert result.verdict == VERDICT_AI
    assert result.score == 0.99
    assert result.overridden_by == PROVENANCE


def test_fuse_with_no_signals_abstains():
    result = fuse([], Thresholds(), FusionConfig())
    assert result.verdict == VERDICT_UNCERTAIN
    assert result.score == 0.5
    assert result.confidence == 0.0


def test_fuse_marks_uncalibrated_by_default():
    result = fuse([make_signal("a", 0.7)], Thresholds(), FusionConfig())
    assert result.calibrated is False
    assert any("uncalibrated" in note for note in result.notes)


def test_fuse_uses_calibration_when_present():
    calibration = Calibration(bias=0.0, weights={"a": 2.0}, fitted=True, source="unit test")
    result = fuse([make_signal("a", 0.7)], Thresholds(), FusionConfig(), calibration)
    assert result.calibrated is True
    assert abs(result.score - sigmoid(2.0 * logit(0.7))) < 1e-9
    assert any("unit test" in note for note in result.notes)


def test_calibrated_score_is_the_fitted_expression_exactly():
    """Serving must evaluate sum(c_i * l_i) + b, the same function eval/calibrate.py fits.

    The previous implementation reused the weighted mean form built for priors, which renormalises
    by the weights present. That is a different function whenever a signal is missing or a
    coefficient is not positive, and both are ordinary cases.
    """
    calibration = Calibration(
        bias=-0.4, weights={"a": 1.3, "b": 0.6, "face_pathway": 2.1}, fitted=True, source="unit"
    )
    signals = [
        make_signal("a", 0.31),
        make_signal("b", 0.44),
        make_signal("face_pathway", 0.52, kind=FACE),
    ]
    expected = sigmoid(1.3 * logit(0.31) + 0.6 * logit(0.44) + 2.1 * logit(0.52) - 0.4)
    result = fuse(signals, Thresholds(), FusionConfig(), calibration)
    assert abs(result.score - expected) < 1e-9


def test_absent_signal_is_imputed_exactly_as_the_fit_imputed_it():
    """calibrate.py fills a missing signal with logit(0.5) = 0, so omitting it must add nothing.

    This is the common case, not an edge one: most images have no faces and no metadata. The
    weighted mean form failed here by renormalising over the signals that happened to be present,
    which silently inflated every remaining coefficient.
    """
    calibration = Calibration(
        bias=0.2, weights={"a": 1.5, "b": 1.5, "face_pathway": 1.5}, fitted=True, source="unit"
    )
    present_only = fuse([make_signal("a", 0.8)], Thresholds(), FusionConfig(), calibration)
    with_neutrals = fuse(
        [
            make_signal("a", 0.8),
            make_signal("b", 0.5),
            make_signal("face_pathway", 0.5, kind=FACE),
        ],
        Thresholds(),
        FusionConfig(),
        calibration,
    )
    assert abs(present_only.score - with_neutrals.score) < 1e-9
    assert abs(present_only.score - sigmoid(1.5 * logit(0.8) + 0.2)) < 1e-9


def test_negative_calibration_coefficient_is_applied_not_dropped():
    """A fitted anti correlation is a finding, not noise to be discarded.

    The weighted mean form skipped any coefficient at or below zero, so the fit and the service
    disagreed on the sign of the evidence.
    """
    calibration = Calibration(bias=0.0, weights={"a": -1.8}, fitted=True, source="unit")
    result = fuse([make_signal("a", 0.9)], Thresholds(), FusionConfig(), calibration)
    assert abs(result.score - sigmoid(-1.8 * logit(0.9))) < 1e-9
    assert result.score < 0.5


def test_signal_absent_from_the_fit_can_still_escalate():
    """A detector the fit never saw is exactly the case the escalation rule exists for.

    An earlier version gated escalation on having contributed to the score, so a newly loading
    checkpoint reading 0.97 was silently unable to stop an authentic verdict. Contributing nothing
    to the score is a reason to abstain, not a reason to ignore the reading.
    """
    calibration = Calibration(bias=0.0, weights={"a": 2.0}, fitted=True, source="unit")
    signals = [make_signal("a", 0.05), make_signal("brand_new_model", 0.97)]
    result = fuse(signals, Thresholds(), FusionConfig(), calibration)
    assert result.escalated_by == "brand_new_model"
    assert result.verdict == VERDICT_UNCERTAIN
    assert abs(result.score - sigmoid(2.0 * logit(0.05))) < 1e-9, "score still excludes it"
    assert any("contributed nothing to the score" in note for note in result.notes)


def test_muted_weight_still_means_muted_for_escalation():
    """A zero weight is an operator decision to ignore a detector, unlike an unfitted coefficient."""
    signals = [make_signal("a", 0.05), make_signal("muted", 0.99, weight=0.0, kind=FACE)]
    result = fuse(signals, Thresholds(), FusionConfig())
    assert result.escalated_by is None
    assert result.verdict == VERDICT_AUTHENTIC


def test_calibrated_path_applies_the_clamp():
    """Without a reading inside the clamp, a test proves nothing about whether the clamp ran."""
    calibration = Calibration(bias=0.1, weights={"a": 1.4}, fitted=True, source="unit")
    result = fuse([make_signal("a", 0.001)], Thresholds(), FusionConfig(), calibration)
    assert abs(result.score - sigmoid(1.4 * logit(0.02) + 0.1)) < 1e-9
    assert result.signals[0]["clamped"] is True


def test_fitted_clamp_beats_the_runtime_clamp():
    """Coefficients only describe the transform they were fitted on."""
    calibration = Calibration(bias=0.0, weights={"a": 1.4}, fitted=True, source="unit", clamp=0.02)
    result = fuse([make_signal("a", 0.001)], Thresholds(), FusionConfig(signal_clamp=0.3), calibration)
    assert abs(result.score - sigmoid(1.4 * logit(0.02))) < 1e-9
    assert any("fitted with (0.020)" in note for note in result.notes)


def test_calibration_without_a_recorded_clamp_says_so():
    calibration = Calibration(bias=0.0, weights={"a": 1.0}, fitted=True, source="unit")
    result = fuse([make_signal("a", 0.4)], Thresholds(), FusionConfig(), calibration)
    assert any("predates clamp recording" in note for note in result.notes)


def test_corrupt_calibration_file_degrades_instead_of_raising():
    """Calibration.load runs during application startup, so anything it raises aborts the boot."""
    path = Path(tempfile.gettempdir()) / "vt_bad_calibration.json"
    broken = [
        '{"bias": null, "weights": {"a": 0.5}}',
        '{"weights": [["a", 0.5]], "bias": 0.1}',
        '{"weights": {"a": "abc"}, "bias": 0.1}',
        '{"weights": 3, "bias": 0.1}',
        '{"weights": {"a": NaN}, "bias": 0.1}',
        '{"weights": {"a": 1.0}, "bias": Infinity}',
        '{"bias": 2.0, "weights": {}}',
        "not json at all",
    ]
    try:
        for text in broken:
            path.write_text(text, encoding="utf-8")
            assert Calibration.load(path).fitted is False, text
    finally:
        path.unlink()


def test_non_finite_reading_is_dropped_not_averaged():
    """A NaN score serialises as a bare NaN token, which is not valid JSON and breaks the client."""
    signals = [make_signal("a", 0.8), make_signal("broken", float("nan"))]
    result = fuse(signals, Thresholds(), FusionConfig())
    assert math.isfinite(result.score)
    assert abs(result.score - 0.8) < 1e-9
    assert any("non numeric reading from broken" in note for note in result.notes)
    assert [s["name"] for s in result.signals] == ["a"]


def test_only_non_finite_readings_abstains():
    result = fuse([make_signal("broken", float("inf"))], Thresholds(), FusionConfig())
    assert result.score == 0.5
    assert result.verdict == VERDICT_UNCERTAIN


def test_non_finite_temperature_abstains_rather_than_emitting_nan():
    signals = [make_signal("a", 0.8)]
    result = fuse(signals, Thresholds(), FusionConfig(temperature=float("nan")))
    assert result.score == 0.5
    assert result.verdict == VERDICT_UNCERTAIN
    assert any("non numeric score" in note for note in result.notes)


def test_signal_absent_from_the_fit_is_excluded_rather_than_guessed():
    """A runtime signal the fit never saw has no coefficient, so it gets no vote.

    Previously it fell back to its prior weight divided by the fitted temperature, which mixed two
    incompatible scales and amplified an unfitted signal severalfold.
    """
    calibration = Calibration(bias=0.0, weights={"a": 1.0}, fitted=True, source="unit")
    signals = [make_signal("a", 0.4), make_signal("newcomer", 0.95)]
    result = fuse(signals, Thresholds(), FusionConfig(), calibration)
    assert abs(result.score - sigmoid(logit(0.4))) < 1e-9
    assert any("newcomer is not in the fitted calibration" in note for note in result.notes)
    newcomer = next(s for s in result.signals if s["name"] == "newcomer")
    assert newcomer["clamped"] is False, "a signal that did not vote was not clamped either"


def test_calibration_matching_no_runtime_signal_abstains():
    calibration = Calibration(bias=3.0, weights={"retired_model": 1.0}, fitted=True, source="unit")
    result = fuse([make_signal("a", 0.9)], Thresholds(), FusionConfig(), calibration)
    assert result.score == 0.5
    assert result.verdict == VERDICT_UNCERTAIN
    assert any("no verdict can be given" in note for note in result.notes)


def test_calibration_without_weights_is_not_treated_as_fitted():
    """Otherwise every score collapses to sigmoid(bias), the same number for every image."""
    path = Path(tempfile.gettempdir()) / "vt_empty_calibration.json"
    path.write_text(json.dumps({"bias": 2.0, "weights": {}}), encoding="utf-8")
    try:
        assert Calibration.load(path).fitted is False
    finally:
        path.unlink()


def test_zero_spread_limit_does_not_crash():
    """0 is the obvious thing to try for an off switch, and the other two knobs accept it."""
    signals = [make_signal("a", 0.2), make_signal("b", 0.8)]
    for limit in (0.0, -1.0):
        result = fuse(signals, Thresholds(), FusionConfig(logit_spread_limit=limit))
        assert result.confidence == 0.0
        assert 0.0 <= result.score <= 1.0


def test_zero_weight_signal_is_not_narrated_as_a_vote():
    """A signal with no weight cast no vote, so it cannot be described as having outvoted anyone."""
    signals = [make_signal("a", 0.5), make_signal("muted", 0.0005, weight=0.0)]
    result = fuse(signals, Thresholds(), FusionConfig())
    assert not any("Capped" in note for note in result.notes)
    muted = next(s for s in result.signals if s["name"] == "muted")
    assert muted["clamped"] is False
    assert muted["p_ai"] == 0.0005, "the raw reading is still reported"


def test_high_clamp_disclosed_when_it_makes_the_bands_unreachable():
    """A clamp above 0.3 leaves every model reading inside the uncertain band by construction."""
    signals = [make_signal("a", 0.02), make_signal("b", 0.99)]
    result = fuse(signals, Thresholds(), FusionConfig(signal_clamp=0.4))
    assert abs(result.score - 0.5) < 1e-9
    assert any("clamp in force is 0.40" in note for note in result.notes)


def test_high_clamp_note_quotes_the_effective_value_not_the_requested_one():
    """clamp_probability caps at 0.49, so quoting the raw setting would misreport what ran."""
    signals = [make_signal("a", 0.02), make_signal("b", 0.99)]
    result = fuse(signals, Thresholds(), FusionConfig(signal_clamp=5.0))
    assert any("clamp in force is 0.49" in note for note in result.notes)


def test_fuse_face_pathway_is_flagged_in_notes():
    signals = [make_signal("model_a", 0.55), make_signal("face_pathway", 0.9, kind=FACE)]
    result = fuse(signals, Thresholds(), FusionConfig())
    assert any("face replacement" in note for note in result.notes)


def test_thresholds_band_edges():
    thresholds = Thresholds()
    thresholds.authentic_max = 0.3
    thresholds.ai_min = 0.7
    assert thresholds.verdict(0.3) == VERDICT_AUTHENTIC
    assert thresholds.verdict(0.7) == VERDICT_AI
    assert thresholds.verdict(0.5) == VERDICT_UNCERTAIN


def test_numpy_is_available_for_explain_colormap():
    from veritrust.explain import _colorize

    coloured = _colorize(np.linspace(0, 1, 16).reshape(4, 4).astype(np.float32))
    assert coloured.shape == (4, 4, 3)
    assert coloured.dtype == np.uint8
    assert coloured[0, 0].sum() < coloured[3, 3].sum(), "colormap should brighten with intensity"


def test_yunet_lfs_pointer_is_rejected():
    """The exact failure that shipped: raw.githubusercontent.com serves pointer text for LFS files."""
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "face_detection_yunet_2023mar.onnx"
        path.write_bytes(
            b"version https://git-lfs.github.com/spec/v1\n"
            b"oid sha256:8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4\n"
            b"size 232589\n"
        )
        problem = yunet_file_problem(path)
        assert problem is not None, "an LFS pointer must not be accepted as a model"
        assert "LFS pointer" in problem
        assert "media.githubusercontent.com" in problem, "the message must say how to fix it"


def test_yunet_truncated_file_is_rejected():
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "yunet.onnx"
        path.write_bytes(b"\x08\x07" + b"\x00" * 4096)
        problem = yunet_file_problem(path)
        assert problem is not None
        assert "too small" in problem


def test_yunet_plausible_file_is_accepted():
    """Validation is a size and header check, not a real ONNX parse, so this only needs to be big."""
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "yunet.onnx"
        path.write_bytes(b"\x08\x07onnx" + b"\x00" * 300_000)
        assert yunet_file_problem(path) is None


def test_yunet_missing_file_reports_rather_than_raises():
    problem = yunet_file_problem(Path(tempfile.gettempdir()) / "no_such_yunet_model.onnx")
    assert problem is not None
    assert "cannot be read" in problem


def write_local_models(folder: str, payload, name="models.local.json") -> Path:
    path = Path(folder) / name
    path.write_text(payload if isinstance(payload, str) else json.dumps(payload), encoding="utf-8")
    return path


def test_local_specs_are_parsed_into_model_specs():
    with tempfile.TemporaryDirectory() as folder:
        path = write_local_models(
            folder,
            {
                "models": [
                    {"key": "detect_world", "path": "weights/detect-world", "kind": "synthetic"},
                    {
                        "key": "detect_3b_omni",
                        "path": "/opt/checkpoints/detect-3b",
                        "kind": "face",
                        "weight": 1.5,
                        "fake_index": 1,
                    },
                ]
            },
        )
        specs, problems = load_local_specs(path)

    assert problems == []
    assert [s.key for s in specs] == ["detect_world", "detect_3b_omni"]
    assert all(s.is_local for s in specs), "local specs must be flagged so failures name a path"
    assert specs[0].kind == SYNTHETIC and specs[1].kind == FACE
    assert specs[1].weight == 1.5 and specs[1].fake_index == 1
    assert Path(specs[0].repo).is_absolute(), "a relative path must resolve against backend/"
    assert specs[1].repo == str(Path("/opt/checkpoints/detect-3b"))


def test_local_specs_accept_a_bare_list():
    with tempfile.TemporaryDirectory() as folder:
        path = write_local_models(folder, [{"key": "detect_2b", "path": "weights/detect-2b"}])
        specs, problems = load_local_specs(path)

    assert problems == []
    assert [s.key for s in specs] == ["detect_2b"]
    assert specs[0].kind == SYNTHETIC, "kind should default to the whole image pathway"


def test_missing_local_models_file_is_silent():
    """Absence is the normal case, so it must not produce a complaint on every boot."""
    with tempfile.TemporaryDirectory() as folder:
        specs, problems = load_local_specs(Path(folder) / "models.local.json")
    assert specs == () and problems == []


def test_malformed_local_models_file_degrades_with_a_reason():
    with tempfile.TemporaryDirectory() as folder:
        path = write_local_models(folder, "{ this is not json")
        specs, problems = load_local_specs(path)

    assert specs == ()
    assert len(problems) == 1 and "could not be read" in problems[0]


def test_local_models_file_of_the_wrong_shape_degrades():
    with tempfile.TemporaryDirectory() as folder:
        path = write_local_models(folder, {"models": {"detect_world": "weights/x"}})
        specs, problems = load_local_specs(path)

    assert specs == ()
    assert len(problems) == 1 and "list of models" in problems[0]


def test_one_bad_local_entry_does_not_discard_the_good_ones():
    with tempfile.TemporaryDirectory() as folder:
        path = write_local_models(
            folder,
            {
                "models": [
                    {"key": "good", "path": "weights/good"},
                    {"key": "bad_kind", "path": "weights/bad", "kind": "text"},
                    {"path": "weights/nameless"},
                    "not an object",
                ]
            },
        )
        specs, problems = load_local_specs(path)

    assert [s.key for s in specs] == ["good"]
    assert len(problems) == 3
    assert any("kind must be" in p for p in problems)
    assert any("required field" in p for p in problems)
    assert any("not an object" in p for p in problems)


def test_local_spec_cannot_shadow_a_builtin_key():
    """Two specs under one key would silently replace a checkpoint rather than add to the ensemble."""
    with tempfile.TemporaryDirectory() as folder:
        path = write_local_models(folder, {"models": [{"key": "face_vit", "path": "weights/x"}]})
        specs, problems = load_local_specs(path)

    assert specs == ()
    assert len(problems) == 1 and "reuses the key" in problems[0]


def test_duplicate_local_keys_keep_the_first():
    with tempfile.TemporaryDirectory() as folder:
        path = write_local_models(
            folder,
            {"models": [{"key": "dup", "path": "weights/a"}, {"key": "dup", "path": "weights/b"}]},
        )
        specs, problems = load_local_specs(path)

    assert len(specs) == 1 and specs[0].repo.endswith("a")
    assert len(problems) == 1 and "reuses the key" in problems[0]


def test_empty_local_key_or_path_is_refused():
    with tempfile.TemporaryDirectory() as folder:
        path = write_local_models(
            folder, {"models": [{"key": "  ", "path": "weights/a"}, {"key": "b", "path": "   "}]}
        )
        specs, problems = load_local_specs(path)

    assert specs == ()
    assert len(problems) == 2 and all("required field" in p for p in problems)


def test_example_local_models_file_parses():
    """The template ships as documentation, so a typo in it would mislead rather than help."""
    example = Path(__file__).resolve().parent.parent / "models.local.example.json"
    assert example.is_file(), "the example file is referenced by the README"
    specs, problems = load_local_specs(example)
    assert problems == [], f"the shipped example must parse cleanly: {problems}"
    assert len(specs) == 3
    assert {s.kind for s in specs} == {SYNTHETIC, FACE}


def _run_all() -> int:
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_")]
    passed, failed = 0, []
    for name, fn in tests:
        try:
            fn()
            passed += 1
            print(f"  ok    {name}")
        except Exception as exc:
            failed.append((name, exc))
            print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")
    print(f"\n{passed} passed, {len(failed)} failed, {len(tests)} total")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
