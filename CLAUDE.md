# CLAUDE.md

Guidance for working in this repository.

## What this is

Deepfake and AI generated media detection, for images and for audio. Classify an image as
photographic or generated, or an audio file as a real recording or synthetic speech, with a
confidence score either way. The earlier version of this project also did text based fake news
classification; that is dead and lives in `legacy/`. Do not revive it or import from it.

## Architecture

Signal sources fused in logit space, never a single model. On an image there are three:

- `detectors/` holds the whole image synthetic ensemble and the face crop model.
- `faces.py` routes to the face pathway only when faces are actually found.
- `provenance.py` reads metadata. Presence of AI provenance hard overrides the models; absence is
  explicitly treated as no evidence, not as evidence of authenticity.
- `fusion.py` combines them and applies the three band verdict.
- `engine.py` orchestrates and knows nothing about HTTP. `main.py` is the only file that does.

Audio is a fourth pathway through the same fusion, not a second system:

- `audio.py` decodes, downmixes, resamples to 16 kHz and plans overlapping windows. It holds the
  aggregation across windows. Sitting beside `preprocessing.py` rather than inside it is deliberate:
  the two share no code, and an audio concern in the image path is how a modality starts leaking.
- `detectors/hf_audio.py` wraps `AutoModelForAudioClassification` plus `AutoFeatureExtractor`, with
  the same name based label resolution as the image wrapper. It is a separate wrapper for the same
  reason `HFImageClassifier` is: the auto classes differ, and routing an audio checkpoint through the
  image wrapper produces a load failure that reads like a dead repo.
- `engine.py` chooses the modality from magic bytes, never from a client supplied MIME type or from
  which tab is open in the UI. The browser's `file.type` comes from the OS association table and is
  routinely empty for `.flac`, `.m4a` and `.opus`.
- Windows are scored per checkpoint and reduced to one reading per checkpoint before fusion, so the
  audio pathway presents to `fusion.py` exactly like the image ensemble does. Fusion knows nothing
  about windows.

Checkpoints come from two places. The built in specs in `config.py` are Hub ids. `models.local.json`
declares checkpoints held on disk, read by `load_local_specs`, so private or unreleased weights join
the ensemble without editing code and without inventing a Hub repo id for something that has no Hub
presence. `from_pretrained` accepts a path as readily as an id, so `is_local` exists only to make
failures accurate: it names a missing directory instead of reporting a network lookup for a path, and
it tells the prefetch script there is nothing to download. A malformed file, duplicate key or unknown
kind degrades to an ensemble without that model and reports why, exactly like a dead Hub repo. Never
add a checkpoint by guessing a repo id.

`ALL_MODELS` is the image ensemble only and deliberately excludes audio; `ALL_AUDIO_MODELS` is the
audio one. Anything that iterates checkpoints has to pick the right list and the right auto classes,
which is why `download_models.py` and `verify_models.py` each run their two pathways separately
rather than over one merged list.

Keep that separation. Preprocessing, model wrapping, fusion and inference orchestration stay in
separate modules. Do not consolidate into one script.

`config.py`, `preprocessing.py`, `provenance.py`, `fusion.py` and `audio.py` deliberately import **no
torch and no pydantic.** Torch imports live inside functions in `detectors/hf_image.py`,
`detectors/hf_audio.py` and `explain.py`. This is what makes the core testable without a GPU or a
model download, and for audio it is what lets the tests push real WAV bytes through the real decoder
instead of mocking it. Preserve it.

## Non negotiables

**Never invent accuracy, F1 or benchmark numbers.** The previous version claimed 99.6 percent
accuracy with nothing behind it. If something is untested, the code, docs and UI must say so. The
API reports `calibrated: false` until `eval/calibrate.py` has actually been run, and the frontend
surfaces that.

This binds hardest on audio, which is the newest pathway and the one with the least behind it. No
audio checkpoint has ever been loaded in this environment and the eval harness is image only, so
audio cannot currently be calibrated at all. Do not write a model count into UI copy either: the
number that actually reads a file is whatever resolved at startup, `/api/v1/models` reports that
honestly, and a hardcoded count goes stale the moment a checkpoint is added or fails to load.

**Never let a single uncalibrated model hold a veto, with one declared exception.** Fusion is in
logit space, which is unbounded, so a member reporting 0.001 contributes about -6.9 and outvotes
everything else. This already caused one confident false negative on a generated image.
`clamp_probability` bounds every model reading to `signal_clamp` before its logit is taken, and
`eval/calibrate.py` records the clamp it fitted with into `calibration.json` so serving can apply that
one in preference to the runtime setting. Coefficients only describe the transform that produced them,
so a mismatch is disclosed in the notes rather than resolved silently. Provenance overrides bypass
fusion and are not clamped, because they are read evidence rather than an estimate. The exception is
`face_decides`, below. It is the only one, and adding a second needs the same explicit decision.

**`face_decides` is a deliberate exception to the rule above.** When faces are found and the face
pathway reads outside the uncertain band, `_decisive_face` hands it the verdict and the score
outright, reporting through `overridden_by` like a provenance override. Aryan chose this on
2026-08-21 after the alternatives and the cost were laid out. Do not quietly narrow it, widen it or
revert it; it is a product decision, not an implementation detail.

The reason is that on a face swap the whole image detectors are answering a different question. They
correctly read camera pixels everywhere outside the swap, and averaging that against the face model
buries the finding: a face at 1.00 against three whole image readings of 0.02 fuses to 0.174. The
escalation rule saved the verdict there but left the needle deep in the authentic band under it.

Unlike escalation this rule is symmetric, so it also closes cases as authentic over a whole image
detector that flagged the image. That is the accepted cost and it must stay visible: the notes name
every overruled reading and state explicitly when one cleared `ai_min` and was overruled anyway.
Three limits keep it narrow, and each has a test. Provenance still wins, because read evidence
outranks an estimate. A reading inside the band decides nothing. A muted face weight cannot decide,
since a zero weight means ignore that detector. `VT_FACE_DECIDES=false` restores fusion plus the
dissent floor.

Two consequences that are easy to break. The score becomes the raw face reading rather than the fused
value, because a needle at 0.174 under a headline saying AI generated is the UI contradicting itself,
and `_decisive_face` must keep testing the same comparisons as `Thresholds.verdict` or a response can
claim one check decided while the verdict says nobody could tell. And the result carries
`calibrated=False`, which is what makes `eval/calibrate.py` drop the record; the fitted expression
never ran on this path, so training on it would teach a relationship that does not exist.

**A vote for AI outweighs a vote for real.** These detectors recognise the generators they were
trained on and fall back to "real" on anything unfamiliar, so silence is weak evidence and a
positive is strong. When one member clears `ai_min`, `dissent_escalates` floors the verdict at
`uncertain` and reports `escalated_by`. The score is deliberately left untouched, because that is
what the eval harness ranks and what calibration is fitted against. This trades false negatives
for false positives on purpose. Note that with `face_decides` on this only reaches a face reading
when that reading is inside the band, so tests for the floor on a face signal must set
`face_decides=False` or they are testing the override instead.

**There is no `audio_decides` and adding one needs the same explicit decision `face_decides` got.**
Audio reaches the verdict through fusion plus the dissent floor, like every pathway except the one
declared exception above. A weighted quorum on escalation was tried while adding audio, requiring
several dissenters rather than one, and reverted: it silently changed the escalation rule for images
too, and a new modality does not get to alter the existing one on its way in. If audio ever needs an
override, argue for it on its own and write down the cost, as was done on 2026-08-21 for faces.

**Audio is scored in overlapping windows, and they combine at a quantile, never at a max.** A max
over N windows only rises as N grows, so a longer file would score higher for being longer, which is
a duration detector dressed as a spoofing detector. `audio_quantile` interpolates, so it stays below
the loudest window and does not drift with clip length; both properties have tests. The overlap
exists because a splice landing on a window boundary is divided between two windows and weakened in
both, so do not "simplify" the hop to equal the window. Above `audio_max_windows` the sample is
spread across the whole file rather than taken from the front, since the first 48 seconds of a long
recording is not a sample of that recording.

**Audio resampling refuses rather than degrades.** These checkpoints expect 16 kHz, and naive
decimation folds high frequency content down into the exact band the spoofing detectors read, which
manufactures the artifacts they look for. `resample` rejects the request and names aliasing and false
positives instead of falling back to anything unfiltered. An installed but failing resampler is
reported, not worked around. Missing decoder and unsupported format are also reported as different
things, because they call for different fixes.

**One broken audio checkpoint reports one error, not one per window.** Windows multiply every
failure by their count, so a dead checkpoint on a 24 window file would fill `errors` with 24 copies
of the same line and make one fault look like a collapsed ensemble. Aggregate per checkpoint before
reporting. There is a test asserting exactly one row.

**Decide modality from the bytes, never from the client or the UI.** `file.type` in a browser comes
from the operating system's association table, not from the file, and is routinely empty for `.flac`,
`.m4a` and `.opus`. So the frontend sends anything it cannot classify rather than refusing it, and
the server sniffs magic bytes and returns a 400 naming the real problem. For the same reason the
verdict wording is taken from the response's `mime`, not from which tab is open: an audio file sent
from the image tab would otherwise get "photograph" in its headline. WAV sniffing must check the
`WAVE` form type, or a WebP registers as audio, and the bare MP3 frame sync must be tested after
JPEG, since `0xD8 & 0xE0 == 0xC0` makes every JPEG look like an MP3 frame header.

**Measure ensemble disagreement in logit space, never probability space.** On the probability
scale 0.219 against 0.001 is a 0.22 gap that any sane tolerance waves through, while as raw odds it
is a factor of 280. Spread is computed after clamping, so that pair reports a factor of 14, which
still clears the limit. These models sit hard against the ends of the range, which is exactly where
the probability scale goes blind. Quote the post clamp figure when describing what the code prints.

**Serve the calibration exactly as it was fitted.** `eval/calibrate.py` fits
`z = sum(c_i * logit(p_i)) + b` with absent signals imputed as zero, and `_score_calibrated`
evaluates that expression directly. Do not fold the coefficients back into the weight and
temperature form used for priors. That was tried, using `w_i = c_i` and `T = 1/sum(c_i)`, and it is
only valid when every fitted signal is present and every coefficient is positive: the weighted mean
renormalises by the weights it can see, the regression renormalises by nothing, and the weight check
silently dropped negative coefficients. Both failure cases are the common case, since most images
have no faces and no metadata. A runtime signal with no fitted coefficient is excluded and named in
the notes, never given a guessed weight.

**One checkpoint gets one vote per pathway.** Specs carry fallbacks, and `broad_swin`'s fallback is
`sdxl_swin`'s primary repo, so an unavailable primary makes two keys resolve to the same weights.
`Registry._build` drops the second and reports it, because otherwise one opinion is counted twice at
double weight while showing zero disagreement with itself, which inflates confidence exactly when the
ensemble has actually shrunk. The check is per pathway: a shared checkpoint reading a face crop and a
whole image is two real observations. Deduplication has to happen after `load`, since `repo_used` is
only known once the fallback chain has been walked.

**Distinguish contributing from being eligible to escalate.** A signal's `counted` flag is false when
its weight is muted or the loaded calibration has no coefficient for it. Escalation and the
disagreement spread instead use everything not muted by the operator, including signals with no
fitted coefficient. A zero configured weight is an instruction to ignore a detector; a missing
coefficient just means nobody has calibrated it yet, and an uncalibrated model reporting AI is a
reason to abstain rather than to look away. Conflating the two silently disabled escalation.

**A non numeric reading is a fault, not an abstention.** Drop it and name it in the notes. NaN
survives clamping, `min`, `max` and `sigmoid`, `round` keeps it, and FastAPI emits it as a bare `NaN`
token that is not valid JSON, so the browser fails before any explanation reaches the reader.
Counting it as a neutral 0.5 would be worse: a broken detector would silently drag every verdict
toward the middle.

**Never resolve class labels by index position.** Use `resolve_fake_indices` in
`detectors/hf_image.py`, which matches on label names from `model.config.id2label`. A checkpoint
with unresolvable labels is marked unavailable rather than guessed at. The old code assumed index
0 was fake with nothing to back it, which silently inverts predictions.

**Match label text by token, with prefixes, and never by substring.** Real checkpoints truncate:
`Ateeqq/ai-vs-human-image-detector` calls its real class `hum`, which exact matching refused, so one
of four detectors was silently absent while the ensemble reported itself merely degraded. So matching
allows a prefix in either direction once both sides reach `MIN_PREFIX`. It must not fall back to
plain substring search, because `ai` occurs inside `painting`, `chair` and `brain`, which resolves
`real_painting` as generated. Keep bare digits out of `FAKE_LABEL_TOKENS` and `REAL_LABEL_TOKENS`:
`0` and `1` sat there as dead entries that only a length filter kept inert, and live they would
resolve `LABEL_0` as real and `LABEL_1` as fake, which is index position wearing a label's clothes.
A label matching both vocabularies is refused rather than settled by which list was searched first.

The audio checkpoints made that vocabulary work harder. Three of the seven use the anti-spoofing terms
rather than "real": `bona-fide` tokenises to `["bona", "fide"]` and `Bonafide` to `["bonafide"]`, and
without `bona` in the vocabulary all three are refused outright since no real class resolves. One
checkpoint labels its classes `AIVoice` and `HumanVoice`, which resolve only because `_label_tokens`
splits camelCase; before that it was the single checkpoint here needing a forced `fake_index`, which is
the index position assumption this project already removed once. No forced index is set for it, so
renamed labels make it refuse to load rather than silently invert. Their class orders genuinely conflict
across the seven: one reports `0=fake 1=real` while others report `0=real 1=fake` and the `AIVoice`
member puts the fake class at index 0 again, which is exactly why none of this can be settled by
position.

**Never rescale pixels manually.** Use the checkpoint's own `AutoImageProcessor`. The old code
divided by 255 before feeding a model whose first layer already rescaled, halving the input range.
The audio equivalent is `AutoFeatureExtractor`: it owns normalisation and the expected sample rate,
so do not normalise a waveform by hand on the way in. `expected_sample_rate` reads what the extractor
declares so a mismatch with `audio_sample_rate` is reported by `verify_models.py` rather than
silently fed through.

**Validate model files, do not just check they exist.** `weights/face_detection_yunet_2023mar.onnx`
shipped as a 131 byte Git LFS pointer, because opencv_zoo tracks it with LFS and
`raw.githubusercontent.com` serves the pointer text for those. `is_file` was true, the download
script reported "already present" forever, and `cv2.FaceDetectorYN.create` failed deep inside OpenCV
so every request silently degraded to Haar. That is why the face pathway never appeared to find
anything. `yunet_file_problem` in `faces.py` is the single check, used by both the detector and the
download script so they cannot drift, and a present but invalid file is reported differently from a
missing one. Fetch from `media.githubusercontent.com`.

**A model that is not an image classifier needs its own wrapper.** `HFImageClassifier` is
`AutoModelForImageClassification` plus name based label resolution. A vision language model has no
`id2label` to resolve and emits text, which is not a probability and has nothing honest to contribute
to fusion. Do not coerce one by parsing its output for the word "fake". When a local checkpoint fails
to load, `describe_checkpoint` reports the `model_type` and `architectures` its `config.json` declares,
so this case identifies itself instead of surfacing as an opaque `from_pretrained` error.

**Failure is non fatal.** A dead or renamed checkpoint degrades the ensemble and is reported
through `/api/v1/models`. It must not take the service down or fail silently. The same applies to
`calibration.json`: `Calibration.load` runs inside the FastAPI lifespan, so anything it raises aborts
the boot, and every coercion in it therefore sits inside the guard. A corrupt or malformed file
degrades to uncalibrated. Non finite coefficients are refused there rather than allowed to reach a
score. `models.local.json` is read at import time and follows the same rule: every problem becomes a
reported string, never an exception. The audio decode chain is the same: `soundfile`, `av` and `soxr`
are each imported inside the function that needs them, so a missing one degrades to a named rejection
for that request rather than an `ImportError` at startup, and plain PCM WAV still works with none of
them installed via the standard library fallback.

**The UI must not contradict the verdict it is displaying.** Face boxes and per signal readings are
coloured by the same three bands the verdict uses, via `bandOf` in `app.js`, never by a 0.5 cut. A
0.5 cut painted a face reading 0.55 the same red as one reading 1.00 while the headline said
uncertain, so the colours read as "fake" and "real" for the whole image and the reader was right to
ask which to believe. The headline also has to distinguish an escalated uncertain from a flat one: a
set `escalated_by` means one check was decisive and got outvoted, and leaving that in the notes while
the headline says "not enough evidence either way" understates what the ensemble actually found.

The overlay text has to track `face_decides` rather than describe fusion generically. It used to say a
red face does not settle the whole image, which is now the opposite of what happens, so it branches on
`overridden_by === FACE_PATHWAY`. On that path the facts line says the face pathway decided alone and
prints no margin: the score is one raw model reading, so a face at 1.00 computes a 100 percent margin
and showing it would claim a certainty no checkpoint here has earned. `spread_exceeds_limit` also
means something different on that path, disagreement overruled rather than averaged, and the wording
distinguishes them.

**Report the reason confidence is zero, do not make the frontend infer it.** Zero has two unrelated
causes, a score inside the band having no margin by definition, and disagreement reducing the margin
to nothing, and rendering both as "0%" says the system knows nothing in the exact case where one
member is certain. `logit_spread` and `spread_exceeds_limit` are on the response for that reason.
Pattern matching the note text for the figure is not a contract.

**Every response key has to be declared in `AnalyzeResponse`.** FastAPI filters the payload through
it, so a field the engine emits but the schema omits is dropped with a 200 and no warning, which
presents as a frontend bug. `test_every_response_key_is_declared_in_the_schema` parses `schemas.py`
with `ast` instead of importing it, so the check still runs without pydantic installed.

## Style

Clean and concise. No unnecessary comments; comments explain why, not what. Module docstrings
explain design decisions and tradeoffs, which is worth the space. **No em dashes anywhere,**
including docstrings and commit messages.

Uncertainty gets surfaced, not hidden. The `uncertain` band is a feature. The frontend shows each
signal separately on a shared axis so disagreement is visible instead of averaged away.

## Before changing architecture

Explain the plan and confirm it first. This applies to swapping models, changing the fusion math,
altering the threshold scheme or restructuring modules.

## Tests

`python tests/run_tests.py` runs everything with no pytest dependency, currently 171 tests across
`test_core.py`, `test_engine.py` and `test_audio.py`. Tests use stub detectors, so they verify wiring
and never accuracy. Add tests for new fusion logic, new provenance patterns, new preprocessing
rejection paths and new configuration degradation paths. Do not write tests that require downloading
weights.

Do not mock what you can run. `audio.py` imports no torch and decodes PCM WAV with the standard
library, so the audio tests build real WAV bytes and push them through the real decoder, resampler and
window planner. The previous version of `test_audio_routing` patched out both `decode_audio` and
`make_windows`, which meant it asserted a mock had been called and exercised none of the routing,
decoding or windowing it appeared to cover; it could not have caught the bug where audio notes were
dropped before reaching the response. A test that only proves a mock was called is worse than no test,
because it reports coverage that does not exist.

Checks that must run before torch is imported belong ahead of the import, not just logically earlier,
so they stay reachable on a machine with no torch installed and can be tested. The local path check in
`HFImageClassifier.load` is there for that reason and the test blocks `sys.modules["torch"]` to prove
it. `HFAudioClassifier` has the same check with the same test. Note that `ModelSpec.is_local` is an
explicit field defaulting to False rather than something inferred from the path, so a test exercising
that branch has to set it: a Hub id and a relative path are not distinguishable by inspection.

One pre-existing failure on Windows, unrelated to any of the above.
`test_local_specs_are_parsed_into_model_specs` asserts a spec resolves to `/opt/checkpoints/detect-3b`,
which is not an absolute path on Windows because it has no drive letter, so `_coerce_local_spec`
correctly prepends `BASE_DIR` and the equality fails. The code is right and the test's fixture is
POSIX specific.
