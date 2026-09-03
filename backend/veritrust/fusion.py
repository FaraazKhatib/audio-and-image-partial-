"""Signal fusion. Pure stdlib, no torch, so it is unit testable in isolation.

Fusion happens in logit space, which preserves the strength of agreement between members better
than averaging probabilities does. It carries one sharp edge that has to be blunted explicitly:
logit space is unbounded, so a single member reporting 0.001 contributes about -6.9 and can
outvote every other signal by itself. An earlier version of this file claimed the opposite in its
docstring and had no guard, which produced a confident "authentic" verdict on a generated image
because one uncalibrated checkpoint was certain and wrong. Averaging probabilities would have
been bounded and would have landed in the uncertain band.

Three rules address that, and they are the reason this is not a plain weighted mean:

Every model probability is clamped to signal_clamp before its logit is taken. None of these
checkpoints has earned a 1 in 1000 assertion, so no single one gets to make it.

A vote for "AI" outweighs a vote for "real" by design. These detectors recognise the generator
fingerprints they were trained on and fall back to "real" on anything unfamiliar, which includes
recompressed images and newer generators, so silence from a member is weak evidence while a
positive is strong. When one member clears ai_min the verdict is floored at uncertain instead of
being allowed to average down to authentic. The score itself is left alone, because it is what
the eval harness ranks and calibrates on, and the reason is reported as escalated_by.

One exception overrides all of that, and it is the only place a single uncalibrated model decides
anything here. When the face pathway reads outside the uncertain band, face_decides hands it the
verdict and the score outright, in either direction, because on a swapped face the whole image
detectors are answering a different question and their correct reading of "photograph" drags the
fused score to 0.174 against a face read at 1.00. It reports through overridden_by like a
provenance override, and unlike the escalation rule it is symmetric, so a low face reading closes a
case as authentic even over a whole image detector that flagged it. That direction is a known cost
of the setting rather than an oversight; it is opt-in, so see FusionConfig and set
VT_FACE_DECIDES=true only after evaluation.

Disagreement is measured in logit space too. On the probability scale 0.219 against 0.001 looks
like a 0.22 gap, comfortably inside any sane tolerance, while as raw odds those two readings are a
factor of about 280 apart. Spread is computed after clamping, so the figure that pair actually
reports is a factor of 14, which still clears the default limit of 2.2 logits comfortably.
Measuring spread on the probability scale is nearly blind exactly where these models like to sit,
which is hard against the ends.

When a calibration file is present the fitted logistic regression is evaluated directly instead of
being folded into the weight and temperature form used for priors. See _score_calibrated for why
the two are not interchangeable.

Two honesty rules also live here. Output carries calibrated=False until eval/calibrate.py has
fitted against labelled data, because an uncalibrated ensemble score is an ordering and not a
probability. And disagreement reduces reported confidence rather than being averaged away.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

from .config import (
    AUDIO,
    FACE,
    SYNTHETIC,
    VERDICT_AUTHENTIC,
    VERDICT_UNCERTAIN,
    FusionConfig,
    Thresholds,
)

EPS = 1e-6

# Emitted from both the fused path and the face override path. It is a statement about what kind of
# manipulation the readings suggest, which is independent of how the verdict was arrived at, and it
# matters more on the override path rather than less. Kept as one constant so the two cannot drift.
FACE_DRIVEN_NOTE = (
    "The face pathway is the main driver here, which points at face replacement or "
    "reenactment rather than a fully generated image."
)


@dataclass
class Signal:
    name: str
    p_fake: float
    weight: float
    kind: str
    detail: str = ""
    override: bool = False

    def as_dict(self, effective: float | None = None, counted: bool = True) -> dict:
        """`counted` is false when this signal contributed nothing to the score.

        A muted weight or a calibration that never saw this signal both produce that state, and it
        is not visible from the numbers alone, so the UI needs to be told rather than left to infer.
        """
        used = self.p_fake if effective is None else effective
        return {
            "name": self.name,
            "p_ai": round(self.p_fake, 4),
            "weight": round(self.weight, 3),
            "kind": self.kind,
            "detail": self.detail,
            "clamped": abs(used - self.p_fake) > 1e-9,
            "counted": counted,
        }


@dataclass
class Calibration:
    """Coefficients and intercept of the logistic regression eval/calibrate.py fits.

    `clamp` records the signal_clamp the fit was performed with, because the coefficients are only
    valid for the feature transform that produced them. Serving honours the fitted clamp over the
    runtime setting rather than silently evaluating a different function. Files written before that
    key existed leave it None and fall back to the runtime value, with a note saying so.

    There is no temperature here on purpose. Earlier versions carried one, derived as 1/sum(c_i)
    so the fit could be squeezed into the weighted mean form used for priors. That derivation was
    unsound and the field is gone rather than left present and ignored.
    """

    bias: float = 0.0
    weights: dict[str, float] = field(default_factory=dict)
    fitted: bool = False
    source: str = ""
    clamp: float | None = None

    @classmethod
    def load(cls, path: Path) -> "Calibration":
        """A corrupt calibration file degrades to uncalibrated, it does not take the service down.

        This is called during application startup, so anything raised here aborts the boot. Every
        coercion therefore happens inside the guard, and non finite values are refused outright
        because a NaN coefficient would poison every score with a value JSON cannot even represent.
        """
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            weights = {str(k): float(v) for k, v in (payload.get("weights") or {}).items()}
            bias = float(payload.get("bias", 0.0))
            clamp = payload.get("signal_clamp")
            clamp = None if clamp is None else float(clamp)
        except (OSError, json.JSONDecodeError, TypeError, ValueError, AttributeError):
            return cls()

        finite = [v for v in weights.values() if math.isfinite(v)]
        if not weights or len(finite) != len(weights) or not math.isfinite(bias):
            return cls()
        if clamp is not None and not math.isfinite(clamp):
            clamp = None

        return cls(
            bias=bias,
            weights=weights,
            fitted=True,
            source=str(payload.get("fitted_on", "unknown dataset")),
            clamp=clamp,
        )


@dataclass
class FusionOutput:
    """Result of one fusion pass.

    The spread is reported rather than kept internal because a confidence of zero has two unrelated
    causes. A score inside the uncertainty band has no margin to report and is zero by definition,
    while a score outside it can still be driven to zero by members contradicting each other. Both
    render as "0%", which reads as the system knowing nothing in the exact case where one member is
    certain. Reading it back out of the notes was the alternative, and that breaks the moment the
    wording changes. Both fields stay at their defaults on the paths that never compute a spread,
    which are the override and the no usable signal returns.
    """

    score: float
    verdict: str
    confidence: float
    calibrated: bool
    signals: list[dict]
    overridden_by: str | None
    notes: list[str]
    escalated_by: str | None = None
    logit_spread: float = 0.0
    spread_exceeds_limit: bool = False

    def as_dict(self) -> dict:
        return {
            "score_ai": round(self.score, 4),
            "verdict": self.verdict,
            "confidence": round(self.confidence, 1),
            "calibrated": self.calibrated,
            "signals": self.signals,
            "overridden_by": self.overridden_by,
            "escalated_by": self.escalated_by,
            "logit_spread": round(self.logit_spread, 3),
            "spread_exceeds_limit": self.spread_exceeds_limit,
            "notes": self.notes,
        }


def logit(p: float) -> float:
    p = min(max(p, EPS), 1.0 - EPS)
    return math.log(p / (1.0 - p))


def clamp_probability(p: float, clamp: float) -> float:
    """Bound a model probability away from 0 and 1 before its logit is taken.

    Without this, one uncalibrated checkpoint can hold a veto over the whole ensemble.
    """
    clamp = min(max(clamp, 0.0), 0.49)
    return min(max(p, clamp), 1.0 - clamp)


def sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    exp_z = math.exp(z)
    return exp_z / (1.0 + exp_z)


def _kind_weight(kind: str, cfg: FusionConfig) -> float:
    if kind == SYNTHETIC:
        return cfg.synthetic_weight
    if kind == FACE:
        return cfg.face_weight
    return cfg.provenance_weight


def _margin_confidence(score: float, thresholds: Thresholds) -> float:
    """Distance from the uncertainty band, as a percentage.

    This reports how decisively the score sits outside the ambiguous zone. It is not a
    probability that the verdict is correct, and the API labels it accordingly.
    """
    if score >= thresholds.ai_min:
        span = max(EPS, 1.0 - thresholds.ai_min)
        return min(100.0, (score - thresholds.ai_min) / span * 100.0)
    if score <= thresholds.authentic_max:
        span = max(EPS, thresholds.authentic_max)
        return min(100.0, (thresholds.authentic_max - score) / span * 100.0)
    return 0.0


def _prior_weight(signal: Signal, cfg: FusionConfig) -> float:
    return signal.weight * _kind_weight(signal.kind, cfg)


def _decisive_face(
    signals: list[Signal], thresholds: Thresholds, cfg: FusionConfig
) -> list[Signal] | None:
    """The face pathway signal when face_decides lets it settle the verdict by itself.

    Decisive means outside the uncertain band in either direction, which is exactly the condition
    that gives the face box a colour in the UI. A reading inside the band gets no special treatment
    and falls through to ordinary fusion. The raw reading is tested rather than the clamped one,
    because this asks what the detector claimed, and clamping exists to limit what a claim is worth
    when averaged rather than to soften the claim itself.

    A muted face weight disables this, since a zero weight is an operator instruction to ignore the
    detector entirely and handing it the whole verdict would be the opposite of that.

    In the AI direction, a whole image detector that strongly reads authentic (below authentic_max)
    vetoes the override. A real face swap does not alter the pixels outside the face region, so a
    whole image detector has no reason to read authentic on one. When it does, the face model is
    more likely wrong than right, and falling through to fusion lets the ensemble weigh in rather
    than handing the verdict to a single uncalibrated model that the config itself describes as
    partly dataset specific. The authentic direction keeps no such requirement, because a low face
    reading on a flagged whole image is evidence of the same kind, not a contradiction.
    """
    if not cfg.face_decides:
        return None
    faces = [s for s in signals if s.kind == FACE and _prior_weight(s, cfg) > 0]
    if not faces:
        return None
        
    face_p = sum(s.p_fake for s in faces) / len(faces)
    
    if face_p <= thresholds.authentic_max:
        return faces
    if face_p >= thresholds.ai_min:
        return faces
    return None


def _face_verdict(
    faces: list[Signal],
    signals: list[Signal],
    thresholds: Thresholds,
    cfg: FusionConfig,
    notes: list[str],
) -> FusionOutput:
    """Build the result for a verdict taken from the face pathway alone.

    The score becomes the face reading rather than staying at the fused value. Reporting a fused
    0.174 next to a verdict of AI generated would put the needle deep in the authentic zone under a
    headline saying the opposite, and the fused number is not what produced this verdict. This is
    the same shape as a provenance override, so it reports through overridden_by and carries
    calibrated=False: eval/calibrate.py fits against the fused score and drops overridden records,
    which keeps this path out of a fit it never applies to.
    """
    face_ids = {id(f) for f in faces}
    face_name = faces[0].name if len(faces) == 1 else "face pathway"
    score = min(max(sum(f.p_fake for f in faces) / len(faces), 0.0), 1.0)
    others = [s for s in signals if id(s) not in face_ids and s.kind == SYNTHETIC]

    notes.append(
        f"Verdict taken from {face_name} alone, which averaged {score:.2f}. Outside the "
        f"{thresholds.authentic_max:.2f} to {thresholds.ai_min:.2f} band the face pathway decides "
        f"by itself, because on a swapped face the whole image detectors are reading real camera "
        f"pixels everywhere outside the swap and averaging them against it buries the finding. Set "
        f"VT_FACE_DECIDES=false to fuse the signals instead."
    )

    if score >= thresholds.ai_min:
        notes.append(FACE_DRIVEN_NOTE)

    if others:
        readings = ", ".join(f"{s.name} {s.p_fake:.2f}" for s in others)
        notes.append(
            f"Overruled {len(others)} whole image reading(s): {readings}. They are shown above and "
            f"contributed nothing to this verdict."
        )

    buried = [s for s in others if s.p_fake >= thresholds.ai_min]
    if score <= thresholds.authentic_max and buried:
        names = ", ".join(f"{s.name} at {s.p_fake:.2f}" for s in buried)
        notes.append(
            f"Note that {names} cleared the {thresholds.ai_min:.2f} AI threshold and was overruled "
            f"anyway. The face pathway is allowed to close a case in the authentic direction under "
            f"this setting, which is the configured behaviour and not a fault, but it is the one "
            f"situation where this build can call an image real over a detector that flagged it."
        )

    ml = [s for s in signals if s.kind in (SYNTHETIC, FACE, AUDIO) and _prior_weight(s, cfg) > 0]
    logits = [logit(clamp_probability(s.p_fake, cfg.signal_clamp)) for s in ml]
    spread = (max(logits) - min(logits)) if len(logits) > 1 else 0.0
    limit = max(cfg.logit_spread_limit, EPS)
    if spread > limit:
        notes.append(
            f"The readings this overruled disagree by a factor of {math.exp(spread):.0f} in odds "
            f"after clamping. Confidence is the distance from the band, so it does not reflect that."
        )

    # The provenance override omits this because provenance reads evidence rather than estimating.
    # Here the score is one uncalibrated model's raw output, so the caveat applies more strongly than
    # it does to a fused score, not less.
    notes.append(
        "This score is one uncalibrated model's raw reading, so treat it as a flag for review "
        "rather than a probability. Calibration does not apply to this path at all."
    )

    return FusionOutput(
        score=score,
        verdict=thresholds.verdict(score),
        confidence=_margin_confidence(score, thresholds),
        calibrated=False,
        signals=[s.as_dict(None, id(s) in face_ids) for s in signals],
        overridden_by=face_name,
        notes=notes,
        logit_spread=spread,
        spread_exceeds_limit=spread > limit,
    )


def _score_prior(rows: list[tuple[Signal, float]], cfg: FusionConfig) -> tuple[float, set[int]]:
    """Weighted mean of logits using the hand set priors, with no fitted bias.

    Returns the score and the ids of the signals that actually carried weight, so callers can
    avoid narrating a signal that had no vote.
    """
    voted: set[int] = set()
    weighted = 0.0
    total = 0.0
    for signal, effective in rows:
        weight = _prior_weight(signal, cfg)
        if weight <= 0:
            continue
        voted.add(id(signal))
        weighted += weight * logit(effective)
        total += weight
    if total <= 0:
        return 0.5, voted
    return sigmoid(weighted / total / max(cfg.temperature, EPS)), voted


def _score_calibrated(
    rows: list[tuple[Signal, float]], calibration: Calibration
) -> tuple[float, set[int], list[str]]:
    """Evaluate the exact function eval/calibrate.py fitted, rather than an approximation of it.

    The fit is a logistic regression over per signal logits: z = sum(c_i * l_i) + b, with absent
    signals imputed as logit(0.5) = 0. Applying that form directly is what makes serving identical
    to fitting. An earlier version folded the coefficients into the prior weight and temperature
    form via w_i = c_i and T = 1/sum(c_i). That identity only holds when every fitted signal is
    present and every coefficient is positive, because the weighted mean renormalises by the
    weights it can see while the regression renormalises by nothing. On the ordinary case of an
    image with no faces and no metadata the two disagreed outright, and a negative coefficient was
    dropped by the weight check instead of being applied.

    Omitting an absent signal here contributes exactly 0 to z, which is what the fit imputed for
    it, so the common case needs no special handling at all.
    """
    voted: set[int] = set()
    notes: list[str] = []
    z = calibration.bias
    for signal, effective in rows:
        coefficient = calibration.weights.get(signal.name)
        if coefficient is None:
            notes.append(
                f"{signal.name} is not in the fitted calibration, so it was excluded from the "
                f"score rather than given a guessed coefficient. Re-run eval/calibrate.py to "
                f"include it."
            )
            continue
        voted.add(id(signal))
        z += coefficient * logit(effective)
    return sigmoid(z), voted, notes


def fuse(
    signals: list[Signal],
    thresholds: Thresholds,
    cfg: FusionConfig,
    calibration: Calibration | None = None,
) -> FusionOutput:
    calibration = calibration or Calibration()
    notes: list[str] = []

    broken = [s.name for s in signals if not math.isfinite(s.p_fake)]
    if broken:
        # A NaN would propagate to the score and serialise as a bare NaN token, which is not valid
        # JSON and fails in the browser before any of this reasoning reaches the user.
        signals = [s for s in signals if math.isfinite(s.p_fake)]
        notes.append(
            f"Discarded a non numeric reading from {', '.join(broken)}. That detector is faulty "
            f"rather than undecided, so it was dropped instead of being counted as neutral."
        )

    if not signals:
        return FusionOutput(
            score=0.5,
            verdict=VERDICT_UNCERTAIN,
            confidence=0.0,
            calibrated=False,
            signals=[],
            overridden_by=None,
            notes=notes + ["No detector produced a usable result, so no verdict can be given."],
        )

    override = next((s for s in signals if s.override), None)
    if override is not None:
        score = min(max(override.p_fake, 0.0), 1.0)
        notes.append(
            f"Verdict set directly by {override.name}, which found explicit evidence rather "
            f"than a statistical estimate."
        )
        if score <= thresholds.authentic_max:
            notes.append(
                "This override points at authenticity, which no current provenance path produces. "
                "It bypasses the lone dissenter floor, so a model reading AI cannot raise it."
            )
        return FusionOutput(
            score=score,
            verdict=thresholds.verdict(score),
            confidence=_margin_confidence(score, thresholds),
            calibrated=False,
            signals=[s.as_dict(None, s is override) for s in signals],
            overridden_by=override.name,
            notes=notes,
        )

    # Checked after provenance and before any fusion. Provenance is read evidence and outranks an
    # estimate, however confident the estimate is.
    faces = _decisive_face(signals, thresholds, cfg)
    if faces is not None:
        return _face_verdict(faces, signals, thresholds, cfg, notes)

    if calibration.fitted and calibration.clamp is not None:
        clamp = calibration.clamp
        if abs(clamp - cfg.signal_clamp) > 1e-9:
            notes.append(
                f"Using the clamp this calibration was fitted with ({clamp:.3f}) rather than the "
                f"configured {cfg.signal_clamp:.3f}. The fitted coefficients only describe the "
                f"transform they were fitted on. Re-run eval/calibrate.py to change it."
            )
    else:
        clamp = cfg.signal_clamp
        if calibration.fitted:
            notes.append(
                "This calibration predates clamp recording, so the runtime clamp was applied and "
                "may not be the one it was fitted with. Re-running eval/calibrate.py removes the "
                "ambiguity."
            )

    rows = [(s, clamp_probability(s.p_fake, clamp)) for s in signals]

    if calibration.fitted:
        score, voted, notes_from_fit = _score_calibrated(rows, calibration)
        notes.extend(notes_from_fit)
    else:
        score, voted = _score_prior(rows, cfg)

    signal_payload = [
        s.as_dict(e, True) if id(s) in voted else s.as_dict(None, False) for s, e in rows
    ]

    if not voted or not math.isfinite(score):
        reason = (
            "No signal carried any weight, so no verdict can be given."
            if not voted
            else "Fusion produced a non numeric score, so no verdict can be given. Check the "
            "calibration file and the fusion environment variables."
        )
        return FusionOutput(
            score=0.5,
            verdict=VERDICT_UNCERTAIN,
            confidence=0.0,
            calibrated=False,
            signals=signal_payload,
            overridden_by=None,
            notes=notes + [reason],
        )

    confidence = _margin_confidence(score, thresholds)
    verdict = thresholds.verdict(score)

    capped = [s.name for s, e in rows if id(s) in voted and abs(e - s.p_fake) > 1e-9]
    if capped:
        notes.append(
            f"Capped an over precise reading from {', '.join(capped)} before fusing. An "
            f"uncalibrated model asserting near certainty would otherwise outvote the rest of "
            f"the ensemble on its own."
        )

    effective_clamp = min(max(clamp, 0.0), 0.49)
    if effective_clamp > 0.3:
        notes.append(
            f"The clamp in force is {effective_clamp:.2f}, high enough that the models cannot "
            f"reach either outer band on their own. Only provenance can produce a decisive verdict "
            f"at this setting."
        )

    # Escalation and disagreement deliberately use every signal the operator has not muted, not
    # just the ones that scored. A detector with no fitted coefficient still made an observation,
    # and a lone positive from an unfitted model is more reason to abstain rather than less.
    ml_rows = [
        (s, e)
        for s, e in rows
        if s.kind in (SYNTHETIC, FACE, AUDIO) and _prior_weight(s, cfg) > 0
    ]

    # ai_min is a threshold on the fused score, deliberately reused here against one raw reading.
    # They are different quantities: calibration rescales the fused score but never the per signal
    # ones, so this test stays as strict as the checkpoints themselves and does not shift when
    # calibration.json is rewritten. Raw rather than clamped, because what matters is what the
    # detector actually claimed.
    #
    # One dissenter is enough, deliberately. A weighted quorum was tried here while the audio
    # pathway was being added, on the theory that a lone overconfident member should not hold up
    # three that agree, and it was wrong in both directions: it silently disabled the floor on the
    # exact face swap case the rule was written for, and it treats uncertain as if it were an
    # accusation. Holding at uncertain says nobody could tell, which is the honest reading when one
    # detector is certain and the others recognise nothing.
    escalated_by = None
    if cfg.dissent_escalates and verdict == VERDICT_AUTHENTIC:
        dissenters = [s for s, _ in ml_rows if s.p_fake >= thresholds.ai_min]
        
        # Audio ensembles are large and prone to single-model hallucinations.
        # Require a quorum of audio weight to trigger an escalation from the audio pathway.
        audio_signals = [s for s, _ in ml_rows if s.kind == AUDIO]
        if audio_signals and dissenters:
            total_audio_weight = sum(_prior_weight(s, cfg) for s in audio_signals)
            dissent_audio_weight = sum(_prior_weight(s, cfg) for s in dissenters if s.kind == AUDIO)
            if dissent_audio_weight > 0 and dissent_audio_weight < total_audio_weight * cfg.audio_dissent_min_weight:
                dissenters = [s for s in dissenters if s.kind != AUDIO]
                notes.append(
                    f"Ignored an audio escalation. An audio model read AI, but the dissenting audio "
                    f"weight ({dissent_audio_weight:.1f}) was less than the required quorum "
                    f"({cfg.audio_dissent_min_weight*100:.0f}% of {total_audio_weight:.1f}). This "
                    f"protects against lone hallucinations on out-of-distribution real audio."
                )

        # Face ensembles are prone to single-model hallucinations on webcam images.
        # Require a quorum of face weight to trigger an escalation from the face pathway.
        face_signals = [s for s, _ in ml_rows if s.kind == FACE]
        if face_signals and dissenters:
            total_face_weight = sum(_prior_weight(s, cfg) for s in face_signals)
            dissent_face_weight = sum(_prior_weight(s, cfg) for s in dissenters if s.kind == FACE)
            if dissent_face_weight > 0 and dissent_face_weight < total_face_weight * cfg.face_dissent_min_weight:
                dissenters = [s for s in dissenters if s.kind != FACE]
                notes.append(
                    f"Ignored a face pathway escalation. A face model read AI, but the dissenting face "
                    f"weight ({dissent_face_weight:.1f}) was less than the required quorum "
                    f"({cfg.face_dissent_min_weight*100:.0f}% of {total_face_weight:.1f}). This "
                    f"protects against lone hallucinations."
                )

        if dissenters:
            loudest = max(dissenters, key=lambda s: s.p_fake)
            verdict = VERDICT_UNCERTAIN
            confidence = 0.0
            escalated_by = loudest.name
            aside = (
                ""
                if id(loudest) in voted
                else " It contributed nothing to the score itself, having no fitted coefficient, "
                "which is why the needle does not reflect it."
            )
            notes.append(
                f"The ensemble average lands in the authentic band, but {loudest.name} reads "
                f"{loudest.p_fake:.2f}, past the {thresholds.ai_min:.2f} AI threshold. "
                f"Held at uncertain rather than called authentic, because a vote for AI outweighs "
                f"a vote for real: these detectors fall back to real on anything unfamiliar, so "
                f"silence is weak evidence and a positive is strong.{aside}"
            )

    used_logits = [logit(e) for _, e in ml_rows]
    spread = (max(used_logits) - min(used_logits)) if len(used_logits) > 1 else 0.0
    limit = max(cfg.logit_spread_limit, EPS)

    if spread > limit:
        confidence *= max(0.0, 1.0 - (spread - limit) / limit)
        notes.append(
            f"Ensemble members disagree by a factor of {math.exp(spread):.0f} in odds. "
            f"Confidence has been reduced accordingly and this result deserves manual review."
        )

    if not calibration.fitted:
        notes.append(
            "Scores are uncalibrated. Treat the number as a ranking rather than a true "
            "probability until eval/calibrate.py has been run on labelled data."
        )
    else:
        notes.append(f"Calibrated on {calibration.source}.")

    face_signals = [s for s, _ in ml_rows if s.kind == FACE]
    if face_signals and sum(s.p_fake for s in face_signals) / len(face_signals) > 0.6:
        notes.append(FACE_DRIVEN_NOTE)

    return FusionOutput(
        score=score,
        verdict=verdict,
        confidence=confidence,
        calibrated=calibration.fitted,
        signals=signal_payload,
        overridden_by=None,
        notes=notes,
        escalated_by=escalated_by,
        logit_spread=spread,
        spread_exceeds_limit=spread > limit,
    )
