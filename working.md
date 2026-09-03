# VeriTrust: how it works, end to end

This describes what the code actually does, in the order it does it, from the moment a file is
chosen in the browser to the moment a verdict is drawn on screen. It covers both modalities, images
and audio. It is written against the source as it stands on 2026-08-24 and every default quoted here
was read out of `config.py` rather than remembered.

One thing to hold onto while reading. Nothing in this system has been measured. No labelled
evaluation set has been run, so the score is an ordering of files against each other and not a
probability that any particular file is generated. The ensemble weights are hand set priors. The
thresholds are chosen, not fitted. Everything below describes machinery that works as designed;
whether the design produces correct answers is a separate question that only `backend/eval_data/`
can answer.

Audio deserves that warning twice over. It is the newest pathway, no audio checkpoint has been
loaded in the environment this was written in, and the eval harness reads image folders only, so
audio cannot currently be calibrated at all. Section 4.7 describes what runs; nothing in it has been
observed against real speech.

## 1. The shape of the system

There is one process. `uvicorn` runs a FastAPI app defined in `veritrust/main.py`, which serves the
JSON API under `/api/v1/` and also mounts the `frontend/` directory at `/` as static files. That
mount is why the frontend can use relative paths and needs no CORS in normal use, and it is the last
thing registered so it cannot shadow the API routes.

Inside that process there is exactly one `Engine` instance, created at import time and loaded during
the FastAPI lifespan startup. It holds the loaded model weights, the face detector and the
calibration. Requests do not create engines; they borrow the one that exists.

Analysis itself is synchronous, GPU bound Python. `main.py` pushes it into a worker thread with
`run_in_threadpool` so the event loop stays free, and guards it with an `asyncio.Semaphore` sized by
`max_concurrency`, default 2. Concurrent forward passes contend for the same VRAM, so without that
cap a burst of uploads turns into an out of memory error rather than a queue.

One upload endpoint serves both modalities. `engine.analyze` sniffs the bytes, and an audio container
header routes to `_analyze_audio` while everything else goes to `_analyze_image`. Both return the same
`Analysis` shape, so the frontend reads the same keys either way and the fields that do not apply are
simply absent rather than zero.

The division of labour between modules is deliberate and worth stating plainly, because it is the
thing most likely to erode. `main.py` is the only file that knows HTTP exists. `engine.py` owns the
pipeline and knows nothing about requests, responses or status codes, which is what lets the eval
scripts drive the identical pipeline from the command line. `fusion.py` knows nothing about images,
only about signals, which is exactly why adding audio needed no change to it at all.
`preprocessing.py`, `audio.py`, `provenance.py`, `faces.py` and the detectors each do one thing and
are unaware of each other.

## 2. The import rule that shapes the code

`config.py`, `preprocessing.py`, `audio.py`, `provenance.py` and `fusion.py` import no torch and no
pydantic. Torch is imported inside functions in `detectors/hf_image.py`, `detectors/hf_audio.py` and
`explain.py`, OpenCV inside functions in `faces.py`, and the three audio decode libraries inside the
functions in `audio.py` that need them, never at module top level.

This is not tidiness. It means the entire decision layer of the system, all the threshold logic, all
the fusion arithmetic, all the metadata reading, can be imported and tested on a machine with no
GPU, no torch and no downloaded weights. `tests/run_tests.py` runs 171 tests that way with no pytest
dependency. It also means a failure while loading a checkpoint cannot take down the parts of the
system that do not need it.

For audio it buys something extra. Because `audio.py` pulls in no torch and decodes plain PCM WAV
with the standard library `wave` module, the tests build real WAV bytes and run them through the real
decoder, resampler and window planner rather than mocking any of it. The earlier version of the audio
routing test patched out `decode_audio` and `make_windows`, and consequently proved only that a mock
had been called.

There is a second, sharper reason inside `HFImageClassifier.load`. The check that a local checkpoint
path is actually a directory happens *before* `import torch`, not merely earlier in the logic, so
that failure stays reachable and testable on a machine where torch is absent. A test blocks
`sys.modules["torch"]` to prove it. `HFAudioClassifier.load` does the same in the same order.

## 3. Startup, in order

### 3.1 Import time

Importing `veritrust.config` reads every environment variable through small `_env_*` helpers that
fall back to a default on a missing or unparseable value, so a typo in `VT_AI_MIN` silently yields
the default rather than crashing the boot. It also builds the model specs: three whole image
checkpoints in `SYNTHETIC_MODELS`, one face checkpoint in `FACE_MODELS`, each with a Hub repo id, a
kind, a prior weight, an ordered tuple of fallback repos and an optional Grad-CAM target layer name.

Then `load_local_specs` reads `backend/models.local.json` if it exists. This is how privately held
or unreleased weights join the ensemble without editing code and without inventing a Hub repo id
for something that has no Hub presence. Each entry needs a key and a path; relative paths resolve
against `backend/`. A malformed file, an entry that is not an object, an unknown kind or a key that
collides with a built in or another local entry becomes a reported string in
`LOCAL_SPEC_PROBLEMS`, never an exception. Nothing here checks that the path contains a loadable
model, because that is the registry's problem and a missing directory should be reported the same
way a dead Hub repo is.

### 3.2 Construction

`Engine.__init__` builds a `Registry`, a `FaceDetector` and an empty `Calibration`. Nothing is
loaded yet. The registry resolves the device at this point: `VT_DEVICE=auto` tries to import torch,
then prefers `cuda`, then `mps`, then falls back to `cpu`, and an ImportError also yields `cpu`.

### 3.3 The lifespan load

`Engine.load` does three things, all of them tolerant of failure.

`Registry.load_all` builds the whole image pathway and then the face pathway. For each spec it
constructs an `HFImageClassifier` and calls `load`, which walks the candidate chain of the primary
repo followed by each fallback in order. For each candidate it loads the checkpoint's own
`AutoImageProcessor`, preferring the fast implementation and retrying with `use_fast=False` on an
ImportError, because transformers routes the fast path through torchvision and a missing torchvision
would otherwise take down every checkpoint at once. It then loads the model with
`AutoModelForImageClassification`, reads `model.config.id2label`, and resolves which class indices
mean "generated" by matching label text.

That label resolution is the part with teeth. `resolve_fake_indices` tokenises each label and
matches tokens against two vocabularies, `FAKE_LABEL_TOKENS` and `REAL_LABEL_TOKENS`. Matching is
exact, or by prefix in either direction once both sides reach three characters. The outward
direction catches inflections like "fakes"; the inward direction catches the truncations real
checkpoints actually ship, since `Ateeqq/ai-vs-human-image-detector` labels its real class `hum`.
Plain substring matching is deliberately not used, because "ai" occurs inside "painting", "chair"
and "brain", which would resolve `real_painting` as generated. Neither vocabulary contains bare
digits, so `LABEL_0` and `LABEL_1` do not resolve and a checkpoint with generic labels is refused
rather than guessed at. A label that matches both vocabularies is also refused rather than settled
by which list happened to be searched first. Multi class checkpoints are handled by summing every
generated-ish class, which covers models that split fake into one class per generator.

A checkpoint that raises anywhere in that sequence contributes a message to a `failures` list and
the loop moves to the next candidate. If every candidate fails, the whole spec is recorded as a
failure and the ensemble simply continues one member smaller. For a local spec the failure message
is enriched by `describe_checkpoint`, which reads the checkpoint's `config.json` and reports the
`model_type`, declared `architectures` and label count, so a checkpoint that is actually a vision
language model identifies itself instead of surfacing as an opaque `from_pretrained` error.

After loading, the registry deduplicates per pathway. Specs carry fallbacks and `broad_swin`'s
fallback is `sdxl_swin`'s primary repo, so if `haywoodsloan/ai-image-detector-deploy` is
unavailable both keys resolve to `Organika/sdxl-detector`. The second one is dropped with an
explanatory failure entry, because identical weights under two names would cast two votes, double
their weight in the fused mean, and report perfect agreement with themselves. That reads as
corroboration when it is one opinion counted twice, and it inflates confidence at exactly the moment
the ensemble has actually shrunk. Deduplication is per pathway rather than global, since the same
checkpoint reading a face crop and a whole image is two genuinely separate observations. It has to
happen after `load` because `repo_used` is only known once the fallback chain has been walked.

`FaceDetector.load` tries YuNet first and Haar second. Before touching OpenCV it validates the ONNX
file with `yunet_file_problem`, which checks both that the file does not begin with the Git LFS
pointer magic and that it is at least 50000 bytes. This exists because the file shipped in this repo
once as 131 bytes of LFS pointer text: `opencv_zoo` tracks it with LFS and
`raw.githubusercontent.com` serves the pointer rather than the model. `is_file` was true, the
download script reported it already present forever, and `cv2.FaceDetectorYN.create` failed deep
inside OpenCV, so every request silently fell back to Haar. That is why the face pathway appeared to
never find anything. The same function is used by the download script so the two checks cannot
drift, and a present but invalid file is reported differently from a missing one. If YuNet is
unusable in `auto` mode the detector falls back to the Haar cascade bundled with OpenCV and says so
in a note that reaches the response, including the warning that Haar misses turned and small faces
so face swaps can go undetected entirely.

`Calibration.load` reads `backend/calibration.json` if it exists. Because this runs inside the
FastAPI lifespan, anything it raises aborts the boot, so every coercion sits inside one guard and a
corrupt or malformed file degrades to uncalibrated. Non finite coefficients are refused outright
here rather than allowed to reach a score, since a NaN coefficient would poison every response with
a value JSON cannot represent.

Startup logs the ensemble size, each failure, and a warning if scores are uncalibrated. Nothing in
this sequence can prevent the service from coming up. The worst case is an ensemble of size zero,
which serves requests that return a note saying no detector is loaded.

## 4. One request, stage by stage

### 4.1 HTTP entry and admission

`POST /api/v1/analyze` takes a multipart file and an optional `heatmap` boolean query parameter,
default true. Size is checked twice, once against `file.size` if the client declared one and again
against the real byte length after reading, both against `max_upload_bytes`, default 20 MB, and
either check returns 413. There is also `POST /api/v1/analyze-base64`, which accepts raw base64 or a
data URL, strips the data URL prefix if present, and returns 400 on a malformed payload. Both then
enter the semaphore and hand off to `engine.analyze` in a worker thread.

Audio has a larger allowance, `max_audio_bytes`, default 40 MB, since a few minutes of WAV clears 20
MB easily. The HTTP layer admits against the higher of the two ceilings because it does not yet know
which modality it has; the per modality limit is enforced during decode, where the file has already
identified itself.

### 4.2 Which modality, decided from the bytes

`engine.analyze` calls `sniff_audio_mime` first. It matches audio container headers only, so an image
falls straight through to the image path and nothing about routing depends on what the client claimed.
This matters more than it sounds. A browser's `file.type` comes from the operating system's file
association table rather than from the file, and it is routinely empty for `.flac`, `.m4a` and
`.opus`, so a client side gate on it would refuse files the server reads perfectly well. The frontend
therefore sends anything it cannot classify and lets the server decide.

Two traps in the sniffing are worth naming because both were live bugs waiting to happen. A RIFF
header alone is not enough for WAV: WebP is also RIFF, so the `WAVE` form type has to be checked, or
every WebP image routes to the audio pathway. And the bare MP3 frame sync has to be tested after
JPEG, because a JPEG's second byte is `0xD8` and `0xD8 & 0xE0 == 0xC0` matches the MP3 sync pattern,
so checking MP3 first sends every JPEG to the audio decoder. Both cases have tests.

### 4.3 Decode, on the image path

`decode_image` in `preprocessing.py` is the only place input is validated, and it never trusts the
client. Format comes from magic bytes via `sniff_mime`, not from the declared content type, and a
format outside `ALLOWED_MIME` (JPEG, PNG, WebP, BMP, TIFF) is rejected. The file is then opened
twice on purpose: once with `verify()` to reject truncated or corrupt data, and again for real use,
because `verify` leaves the file object unusable.

Pixel count is checked against `max_pixels`, default 50 MP, and an oversized image is rejected
outright as a decompression bomb guard. Orientation is normalised with `ImageOps.exif_transpose`,
which matters because a phone photo stored rotated would otherwise be analysed sideways and cropped
in the wrong place. The image is converted to RGB. Finally, if the longest edge exceeds `max_edge`,
default 2048, the image is downscaled with LANCZOS and the `downscaled` flag is set, which later
attaches a note to the response saying resampling weakens the high frequency cues these detectors
rely on.

The raw bytes are kept on the `DecodedImage` alongside the decoded pixels, because provenance has to
read the original file, not the normalised image.

Note what does not happen here: no rescaling, no normalisation, no resizing to a model input size.
Each detector does that itself through its own processor. The old version of this project divided by
255 in front of a model whose first layer already rescaled, halving the input range, and deferring
to the checkpoint's processor makes that class of mistake impossible.

### 4.4 Provenance

`provenance.inspect` reads metadata from the original bytes. It pulls EXIF including the Exif and
GPS sub-IFDs, PNG text chunks and other `image.info` entries, and any XMP packet found by a bounded
regular expression search for `<x:xmpmeta>`. Those get concatenated into one haystack which is then
searched for about twenty generator name patterns, for the IPTC `digitalSourceType` values that
declare algorithmic generation, and for generation metadata keys such as `parameters`, `prompt` and
`workflow` that diffusion tooling writes. C2PA container presence is detected by looking for JUMBF
and `caBX` markers, and if the optional `c2pa` package happens to be installed the manifest is
actually parsed and searched too.

Two design decisions here matter more than the pattern list. First, scanning is confined to metadata
regions rather than the whole file. A whole file byte scan would flag a photograph of a Midjourney
screenshot, or a file merely named `midjourney.jpg`, as generated. Second, the signal is
deliberately asymmetric. Positive evidence of generation is close to conclusive, so a C2PA AI
declaration returns `p_fake` 0.99 with `override` set, and a named generator or a generation metadata
key returns 0.97 with `override` set. A camera EXIF signature, meaning at least three of Make,
Model, DateTimeOriginal, ExposureTime, FNumber and ISOSpeedRatings are present, returns a weak 0.35
with no override and an evidence line saying metadata is trivially forged. Anything else returns
`p_fake` of None, which excludes provenance from fusion entirely, along with an explicit evidence
line stating that absence of metadata is not evidence either way because screenshots and social
uploads strip metadata from real photos too.

That last line is the whole point. Absence of provenance is treated as silence, never as evidence of
authenticity.

### 4.5 The whole image ensemble

Every loaded synthetic detector runs on the full decoded image. `HFImageClassifier.predict` calls
the checkpoint's own processor, moves the tensors to the device, casts to fp16 when running fp16 on
CUDA, runs a forward pass under `torch.inference_mode`, softmaxes the logits and sums the
probability mass on the resolved fake indices. Any exception is caught and returned as an unusable
result carrying the error text rather than raised.

Each usable result becomes a `Signal` carrying the detector key, the probability, the spec weight
and kind `synthetic`. Each unusable one appends to the response's `errors` list, which is how a
single broken detector shows up in the UI without affecting the verdict. The engine also tracks
which synthetic detector read highest, because that is the one the heatmap will be computed from.

### 4.6 The face pathway

`FaceDetector.detect` converts the image to BGR and runs the active backend. YuNet gets its input
size set to the actual image dimensions on every call, which is required because it was created with
a placeholder 320 by 320. Detections below `face_score_threshold`, default 0.6, are filtered by
YuNet itself. Haar has no confidence output at all, so detections from that path are recorded with a
synthetic score of 1.0, which is worth knowing before reading a Haar `score` as if it meant
something.

Boxes smaller than `face_min_size` on either side, default 64 pixels, are dropped. The survivors are
sorted by area descending and truncated to `max_faces`, default 8. That sort is why the face indices
in the response and the labels in the UI run largest first rather than left to right.

Each surviving box is expanded outward by `face_margin`, default 0.25, and cropped, clamped to the
image bounds. The margin is load bearing rather than cosmetic: face forgery cues concentrate at the
blend boundary, which sits outside a tight face box. The crop then goes to every loaded face
detector, and the per face score is the mean across them. With the current single face checkpoint
that mean is just that model's reading.

The whole pathway collapses to one signal. `p_fake` is the **maximum** across faces, on the
reasoning that one swapped face is enough to condemn an image, with a `detail` string naming how
many crops were scored and which backend found them. The per face numbers survive separately in the
response's `faces` array, which is what the UI draws boxes from, so the detail is visible even
though fusion only sees the maximum.

Two consequences of that maximum are worth stating. It means a group photo takes the highest of
several independent readings from an uncalibrated model, so more faces means more chances that one
of them reads high for reasons unrelated to manipulation. And because each crop is resized to the
checkpoint's input, a face near the 64 pixel floor is being upsampled by roughly a factor of three
before the model sees it, which removes exactly the high frequency detail these detectors use to
recognise camera pixels. Neither effect has been measured. Both are plausible enough that a size
gate is the first thing to try if the face pathway turns out to be trigger happy.

### 4.7 The audio pathway

Everything from 4.3 through 4.6 is the image path. When `sniff_audio_mime` matched back in 4.2, none
of that runs and `_analyze_audio` takes over instead. It reaches the same `fuse` call with the same
kind of signal list, so section 4.8 applies unchanged; what differs is only how those signals are
produced.

**Decode, in `decode_audio`.** The per modality size limit, `max_audio_bytes`, is enforced here now
that the file has identified itself. Decoding tries three backends in turn, each imported inside the
function so a missing one is a named rejection rather than an import error at startup: the standard
library `wave` module for plain PCM WAV, `soundfile` for WAV, FLAC and OGG, and `av` (PyAV) for MP3,
M4A and the other compressed containers. A file no available backend can read is refused with a
message that says the decoder is missing rather than that the format is unsupported, because those
call for different fixes. Multi channel audio is averaged to mono, not reduced to one channel, and the
`downmixed` flag with the original channel count is recorded. Non finite samples are scrubbed. A
recording longer than `max_audio_seconds`, default 600, is cut to that length with `truncated` set and
the original duration kept.

**Resample to 16 kHz, in `resample`.** These checkpoints were trained at 16 kHz and the feature
extractors expect it. The conversion is anti-aliased through `soxr`, or through `av` as a fallback,
and if neither is available the request is refused rather than run through naive decimation. That
refusal is not fussiness: decimation without a low pass folds high frequency content down into the
exact band the spoofing detectors read, which manufactures the artifacts they are looking for, so an
unfiltered resample would generate false positives. An identity rate returns the input untouched, and
an installed resampler that throws is reported rather than silently worked around.

**Plan windows, in `make_windows`.** The waveform is cut into `audio_window_seconds` windows, default
4, on an `audio_hop_seconds` hop, default 2. The overlap is deliberate: a splice landing on a window
boundary is divided between two windows and weakened in both, and a hop shorter than the window is
what gives every position two chances to be seen whole. A clip shorter than one window yields a single
window; a remainder shorter than the hop gets its own tail window, because the end of a file is a
natural place to graft something in. Above `audio_max_windows`, default 24, the windows are sampled
evenly across the whole recording rather than taken from the front, so a splice at minute four is not
missed by a run that still reports full coverage. Each window's RMS is measured, and windows below
`audio_min_rms`, default 0.004, are dropped as near silent with a note, because scoring room tone
feeds noise into the aggregate as though it were evidence. If every window is silent the loudest is
kept anyway, flagged as not to be trusted, so a file that clearly contained audio still returns a
reading rather than nothing. Every one of these reductions writes a note, because a capped or gated
run that reported nothing would read as full coverage.

**Score, per checkpoint.** Each loaded audio detector runs on every surviving window through its own
`AutoFeatureExtractor`, exactly as the image detectors defer resizing and normalisation to their
processors. The per window readings for one checkpoint are then reduced to a single number by the 0.9
`audio_quantile`, **not** by a max. A max over N windows only rises as N grows, so a longer file would
score higher purely for being longer, turning a spoofing detector into a duration detector; the
quantile interpolates and stays below the loudest window without drifting upward with clip length. The
result is one `Signal` per checkpoint with kind `audio` and a detail line naming the quantile, the
window count and the highest window reading, which is the same shape the image ensemble hands to
fusion.

**Report failures once.** A checkpoint that fails fails the same way on all 24 windows. The loop
records the first failure and emits exactly one error row for it, with a note of how many windows it
did score, because 24 identical rows in the error list would read as 24 separate faults. There is a
test asserting one row for one broken checkpoint across many windows.

From here the signals enter `fuse` and the rest is shared. There is no audio equivalent of the face
override: audio reaches the verdict through fusion plus the dissent floor like the whole image
ensemble, and the deliberate absence of an `audio_decides` is a rule in its own right, covered in
CLAUDE.md. Provenance is not read on this path, because metadata reading is implemented for images
only, and a note says so explicitly so the empty provenance result is not read as "checked, found
nothing".

### 4.8 Fusion

`fuse` in `fusion.py` receives the signal list, the thresholds, the fusion config and the
calibration, and applies the following checks strictly in this order. The order is the design.

**Non finite readings are dropped first.** A NaN survives clamping, `min`, `max`, `sigmoid` and
`round`, and FastAPI emits it as a bare `NaN` token that is not valid JSON, so the browser fails
before any explanation reaches the reader. Such a reading is discarded and named in the notes as a
fault. Counting it as a neutral 0.5 would be worse, because a broken detector would then silently
drag every verdict toward the middle.

**No signals at all** returns score 0.5, verdict `uncertain`, confidence 0 and a note saying no
verdict can be given.

**A provenance override wins outright.** If any signal carries `override`, its probability becomes
the score directly, bypassing all fusion, all clamping and all escalation. The rationale is that
provenance is read evidence rather than an estimate, so it outranks any model however confident. The
result reports `overridden_by` and `calibrated=False`.

**A decisive face reading wins next.** This is the one place in the project where a single
uncalibrated model is allowed to overrule the ensemble, and it is a product decision made on
2026-08-21 rather than an implementation detail. `_decisive_face` returns the face signal when
`face_decides` is on, the signal's effective weight is above zero, and its raw reading is at or
outside either threshold, meaning at least `ai_min` or at most `authentic_max`. That is exactly the
condition that gives the face box a colour in the UI, which is the point: a red box means AI
generated and a green box means real.

When it fires, `_face_verdict` sets the score to the raw face reading rather than the fused value,
because reporting a fused 0.207 next to a headline saying AI generated would put the needle deep in
the authentic zone under a contradicting headline, and the fused number is not what produced the
verdict anyway. It reports through `overridden_by` like a provenance override, marks only the face
signal as counted, and carries `calibrated=False` so `eval/calibrate.py` drops the record. The notes
name every overruled whole image reading, and if the override closed the case as authentic over a
detector that had cleared `ai_min`, a further note says so explicitly and calls it the configured
behaviour rather than a fault. Three limits keep the rule narrow, and each has a test: provenance
still outranks it, a reading inside the band decides nothing and falls through to ordinary fusion,
and a muted face weight cannot decide because a zero weight is an instruction to ignore that
detector entirely. `VT_FACE_DECIDES=false` restores fusion plus the dissent floor.

The rule is symmetric, which is the accepted cost. A face reading 0.04 over a whole image detector
reading 0.97 returns `likely_authentic` at 0.04. That is the one situation where this build can call
an image real over a detector that flagged it.

**Then the clamp is chosen.** Every model probability is bounded away from 0 and 1 by
`signal_clamp`, default 0.02, before its logit is taken. Logit space is unbounded, so an
uncalibrated checkpoint reporting 0.001 contributes about -6.9 and can outvote the entire rest of
the ensemble by itself. This is not hypothetical; it produced a confident authentic verdict on a
generated image in this project. The clamp caps any single member's pull at about 3.9 logits. If a
fitted calibration recorded the clamp it was fitted with, that one is used in preference to the
runtime setting and any mismatch is disclosed in the notes, because the fitted coefficients only
describe the transform that produced them.

**Then the score.** With no calibration, `_score_prior` takes a weighted mean of the clamped logits
using effective weights, divides by the temperature and applies a sigmoid. With a calibration,
`_score_calibrated` evaluates the fitted logistic regression directly as `z = sum(c_i * l_i) + b`.
Those two are not interchangeable and must not be folded into each other. That was tried, using
`w_i = c_i` and `T = 1/sum(c_i)`, and it only holds when every fitted signal is present and every
coefficient is positive: the weighted mean renormalises by the weights it can see while the
regression renormalises by nothing, and the weight check silently dropped negative coefficients.
Both failure cases are the common case, since most images have no faces and no metadata. A runtime
signal with no fitted coefficient is excluded from the score and named in the notes, never given a
guessed weight.

**Then contributing is distinguished from being eligible to escalate.** A signal's `counted` flag is
false when its weight is muted or the loaded calibration has no coefficient for it. Escalation and
the disagreement spread deliberately use every signal the operator has not muted, including signals
with no fitted coefficient, because a zero configured weight is an instruction to ignore a detector
while a missing coefficient just means nobody has calibrated it yet, and an uncalibrated model
reporting AI is a reason to abstain rather than to look away. Conflating those two silently disabled
escalation once.

**Then the verdict and the margin.** `Thresholds.verdict` maps the score to one of three bands:
`likely_ai_generated` at or above `ai_min`, `likely_authentic` at or below `authentic_max`,
`uncertain` in between. Confidence is `_margin_confidence`, which is the distance from the band as a
percentage of the space available on that side. It is not a probability that the verdict is correct,
and the response says so in `confidence_meaning`. Any score inside the band has zero margin by
definition.

**Then the dissent floor.** If the fused verdict is `likely_authentic` and any unmuted model reading
is at or above `ai_min`, the verdict is raised to `uncertain`, confidence is set to zero, and the
loudest dissenter is named in `escalated_by`. The reasoning is that these detectors recognise the
generator fingerprints they were trained on and fall back to "real" on anything unfamiliar,
including recompressed images and newer generators, so silence is weak evidence while a positive is
strong. Averaging them symmetrically manufactures false negatives. The score is deliberately left
untouched, because the score is what the eval harness ranks and what calibration is fitted against.
This trades false negatives for false positives on purpose. Note that with `face_decides` on, this
rule only ever sees a face reading that was inside the band, since anything outside it already
decided the verdict.

**Then disagreement.** Spread is the range of the clamped logits across unmuted model signals,
measured in logit space and never in probability space. On the probability scale 0.219 against 0.001
looks like a 0.22 gap that any sane tolerance waves through, while as raw odds those two readings
are a factor of about 280 apart, and these models sit hard against the ends of the range where the
probability scale goes blind. When spread exceeds `logit_spread_limit`, default 2.2, confidence is
scaled down proportionally and a note says the result deserves manual review. Both `logit_spread`
and `spread_exceeds_limit` are on the response so the frontend never has to pattern match a number
out of the note text, and so a confidence of zero can be explained: it means either a score inside
the band, which has no margin by definition, or disagreement that ate the margin, and rendering both
as "0%" would claim the system knows nothing in the exact case where one member is certain.

Finally the notes gain a line stating whether the score is calibrated, and a line about face driven
findings if a face reading is above 0.6, which points at face replacement or reenactment rather than
a fully generated image.

### 4.9 The heatmap

If a heatmap was requested, heatmaps are enabled and at least one synthetic detector produced a
usable reading, `grad_cam` runs on the single highest reading synthetic detector. It resolves the
layer named in the spec's `gradcam_target`, falling back to a guess at the last normalisation or
convolution layer, registers forward and backward hooks, runs a forward pass with gradients enabled,
backpropagates the summed fake class logits, and combines the captured activation and gradient into
a class activation map. `_to_grid` handles the three tensor shapes these backbones emit: channels
first convolution maps, Swin style `B H W C` blocks and flat `B N C` token sequences, including the
case of a leading class token. The result is colourised, blended over the image and returned as a
base64 PNG data URL.

Every failure path returns None, and the whole body is wrapped so hooks are always removed. Hooking
the internals of arbitrary checkpoints is inherently fragile and a missing heatmap must never break
an analysis. One consequence worth knowing: `siglip_ai_human` has no `gradcam_target`, so if it is
the highest reading synthetic detector there is no heatmap for that request.

The heatmap explains a whole image detector, not the verdict. On a face override the verdict came
from a different model entirely and the heat is showing you what the overruled detector was looking
at.

### 4.10 The response

`engine.analyze` assembles the `Analysis`, appending contextual notes that fusion could not know
about: the face backend's own note if the pathway is unavailable, a line if no faces were found, a
line if faces were found but no face model is loaded, a line if the image was downscaled, and a line
if no detection model loaded at all. It attaches the provenance dictionary, the per face array, the
heatmap, the image metadata, per stage timings and the errors list.

The audio path appends its own equivalents: the windowing notes from `make_windows`, a line if audio
is switched off by configuration, a line if no audio model is loaded, and a line each for truncation,
resampling and downmixing naming the before and after values. Every conversion the pipeline performed
on the input is disclosed rather than applied silently, since each one changes what the verdict rests
on. There is no heatmap on this path and the faces array is empty.

The `image` object carries whichever metadata fields apply to the modality that ran, which is also how
a caller tells which one that was: audio adds `duration`, `sample_rate`, `windows_scored`, `decoder`
and the `original_*` counterparts, and leaves the pixel dimensions absent. Both modalities go through
the same `MediaOut`, which is why every audio field has to be declared there too.

FastAPI then filters that dictionary through the `AnalyzeResponse` pydantic model. This is the one
place a silent bug can hide: a field the engine emits but the schema omits is dropped with a 200 and
no warning, which presents as a frontend bug. `test_every_response_key_is_declared_in_the_schema`
guards it by parsing `schemas.py` with `ast` rather than importing it, so the check still runs
without pydantic installed.

## 5. Every number in one place

All of these are defaults read from `config.py`. Every one is overridable by the named environment
variable, and an unparseable value silently falls back to the default rather than failing the boot.

### Verdict and fusion

| Setting | Env var | Default | What it does |
| --- | --- | --- | --- |
| authentic_max | `VT_AUTHENTIC_MAX` | 0.35 | At or below this the verdict is likely authentic |
| ai_min | `VT_AI_MIN` | 0.65 | At or above this the verdict is likely AI generated |
| synthetic_weight | `VT_W_SYNTHETIC` | 1.0 | Kind multiplier for whole image detectors |
| face_weight | `VT_W_FACE` | 1.2 | Kind multiplier for the face pathway |
| audio_weight | `VT_W_AUDIO` | 1.0 | Kind multiplier for the audio pathway |
| provenance_weight | `VT_W_PROVENANCE` | 0.4 | Kind multiplier for a non overriding provenance reading |
| temperature | `VT_TEMPERATURE` | 1.0 | Divides the weighted mean logit before the sigmoid, above 1 softens |
| signal_clamp | `VT_SIGNAL_CLAMP` | 0.02 | Bounds every model probability into [0.02, 0.98] before its logit |
| dissent_escalates | `VT_DISSENT_ESCALATES` | true | Floors an authentic verdict at uncertain when one member clears ai_min |
| face_decides | `VT_FACE_DECIDES` | true | A face reading outside the band takes the verdict and the score |
| logit_spread_limit | `VT_LOGIT_SPREAD_LIMIT` | 2.2 | Post clamp logit range above which confidence is reduced |

### Input handling

| Setting | Env var | Default |
| --- | --- | --- |
| max_upload_bytes | `VT_MAX_UPLOAD_BYTES` | 20971520, which is 20 MB |
| max_pixels | `VT_MAX_PIXELS` | 50000000 |
| max_edge | `VT_MAX_EDGE` | 2048 |
| max_concurrency | `VT_MAX_CONCURRENCY` | 2, floored at 1 |
| enable_heatmap | `VT_ENABLE_HEATMAP` | true |

Accepted formats are JPEG, PNG, WebP, BMP and TIFF, decided by magic bytes.

### Audio

| Setting | Env var | Default | What it does |
| --- | --- | --- | --- |
| enable_audio | `VT_ENABLE_AUDIO` | true | Off refuses audio with a note, rather than pretending to read it |
| max_audio_bytes | `VT_MAX_AUDIO_BYTES` | 41943040, which is 40 MB |
| max_audio_seconds | `VT_MAX_AUDIO_SECONDS` | 600.0 | Longer recordings are cut, and the cut is reported |
| audio_sample_rate | `VT_AUDIO_SAMPLE_RATE` | 16000 | What the checkpoints expect, so anything else is resampled |
| audio_window_seconds | `VT_AUDIO_WINDOW_SECONDS` | 4.0 | Length of each window handed to a checkpoint |
| audio_hop_seconds | `VT_AUDIO_HOP_SECONDS` | 2.0 | Window step. Shorter than the window on purpose |
| audio_max_windows | `VT_AUDIO_MAX_WINDOWS` | 24 | Cap, sampled across the whole file rather than the first 24 |
| audio_min_rms | `VT_AUDIO_MIN_RMS` | 0.004 | Below this a window is near silent and is skipped with a note |
| audio_quantile | `VT_AUDIO_QUANTILE` | 0.9 | How a checkpoint's window readings combine. Not a max |

Accepted audio containers are WAV, FLAC, OGG, MP3, M4A and MP4, AAC and WebM, decided by magic bytes.
Which of those actually decode depends on which of `soundfile` and `av` are installed; plain PCM WAV
works with neither.

### Faces

| Setting | Env var | Default |
| --- | --- | --- |
| enable_faces | `VT_ENABLE_FACES` | true |
| face_detector | `VT_FACE_DETECTOR` | auto, meaning YuNet then Haar. Also accepts yunet, haar, off |
| yunet_path | `VT_YUNET_PATH` | `backend/weights/face_detection_yunet_2023mar.onnx` |
| face_score_threshold | `VT_FACE_SCORE` | 0.6, YuNet only, Haar has no score |
| face_min_size | `VT_FACE_MIN_SIZE` | 64 pixels on either edge |
| face_margin | `VT_FACE_MARGIN` | 0.25 of box width and height, expanded outward |
| max_faces | `VT_MAX_FACES` | 8, taken largest first |

### Device and models

`VT_DEVICE` and `VT_DTYPE` both default to auto, resolving to cuda then mps then cpu, and to float16
on cuda. `VT_LOCAL_MODELS` overrides the path to `models.local.json`. `HF_HOME` relocates the
Hugging Face cache.

Prior spec weights, which are hand set and not fitted: sdxl_swin 1.0, broad_swin 1.0,
siglip_ai_human 0.8, face_vit 1.0. The effective weight used inside fusion is the spec weight times
the kind multiplier, so a face reading at spec weight 1.0 actually carries 1.2 while
siglip_ai_human carries 0.8. The signal table in the UI shows the spec weight, so the multiplier is
never visible there. Provenance signals are created with spec weight 1.0 and pick up the 0.4
multiplier.

## 6. A worked example, with real numbers

This is the group photo analysed on 2026-08-21. Four faces were found in a 960 by 1280 JPEG. The
figures below came from running the real `fuse()`, not from arithmetic by hand, because hand
arithmetic on effective weights has produced wrong numbers in this project before.

Readings, with the clamp at 0.02 and the spread limit at 2.2:

| Signal | Raw | Clamped | Logit | Effective weight |
| --- | --- | --- | --- | --- |
| sdxl_swin | 0.046 | 0.046 | -3.0320 | 1.00 |
| broad_swin | 0.004 | 0.020 | -3.8918 | 1.00 |
| siglip_ai_human | 0.000 | 0.020 | -3.8918 | 0.80 |
| face_pathway | 0.999 | 0.980 | +3.8918 | 1.20 |

With defaults, meaning `face_decides` on, the outcome is `likely_ai_generated` at score 0.999,
confidence 99.7, `overridden_by` face_pathway, `logit_spread` 7.7836 which is a factor of about 2400
in odds, `spread_exceeds_limit` true, `calibrated` false, and only the face signal marked counted.
The face reading is outside the band, so it took both the verdict and the score, and the three whole
image readings are named in the notes as overruled.

With `VT_FACE_DECIDES=false` the same four readings fuse to 0.2072, which is inside the authentic
band, and then the dissent floor lifts the verdict to `uncertain` with `escalated_by` face_pathway
and confidence 0. The spread is identical because clamping and spread do not depend on the override.

That pair of outcomes is the whole design tradeoff in one image. Fusion buries a confident face
finding at 0.2072, deep in the authentic band, while the override puts the needle where the headline
says it should be. Whether the override is right here depends on whether the face reading is right,
and that is exactly what has not been measured.

Worth noting about this specific case: the four per face readings were 1.00, 0.38, 0.00 and 0.00, and
the 1.00 was not the most sideways face. The face crops in a 960 by 1280 group photo sit not far
above the 64 pixel floor, so each is upsampled substantially before the model sees it, and the
maximum across four faces takes the highest of four independent draws from an uncalibrated model.
Both of those are reasons to suspect the reading rather than the image, and neither has been tested.
A minimum pixel width before a face may decide, or a rule that only the largest face may decide,
would address it, but no such gate exists in the code today.

## 7. What the frontend does with the response

`frontend/app.js` is vanilla ES modules with no build step. On load it calls `/api/v1/models` and
overwrites its module level `thresholds` from the response, so the bands the UI draws always match
the bands the server used. If that call fails it falls back to a hardcoded 0.35 and 0.65, which is
correct today and would silently disagree with a server running custom thresholds.

`render` draws seven things: the headline, a facts line, the score scale, the per signal table, the
provenance block, the notes and the face overlay, plus the raw JSON with the heatmap stripped out
for size.

The rule the UI is built around is that it must never contradict the verdict it is displaying. Face
boxes and per signal readings are coloured by `bandOf`, which uses the same three bands, never a 0.5
cut. A 0.5 cut previously painted a face reading 0.55 the same red as one reading 1.00 while the
headline said uncertain, so the colours read as a flat "fake" and "real" and the reader was right to
ask which to believe.

The headline distinguishes an escalated uncertain from a flat one. A set `escalated_by` means one
check was decisive and got outvoted, which is a different thing from not enough evidence either way,
and leaving that distinction in the notes while the headline flattens it understates what the
ensemble found.

The overlay text branches on `overridden_by === "face_pathway"` rather than describing fusion
generically, because on that path the generic description is the opposite of what happened. On the
override path the facts line says the face pathway decided alone and prints **no margin at all**,
even though the API returns confidence 99.7 for a 0.999 reading. That is deliberate: the score is one
raw model reading, and `_margin_confidence` measures distance from the band, so a face at 1.00
computes a 100 percent margin and showing it would claim a certainty no checkpoint here has earned.
`spread_exceeds_limit` also means something different on that path, disagreement overruled rather
than averaged, and the wording distinguishes the two.

The uncalibrated state is surfaced in the UI rather than hidden, and confidence of zero says which of
its two causes applies, using `logit_spread` and `spread_exceeds_limit` from the response rather than
pattern matching the note text.

Two things in the frontend change with modality, and both read the modality from the response rather
than from which tab is open, because the server decides modality from the file header and the open tab
can be wrong. The headline noun comes from `modalityOf(data)`, which reads the response `mime`, so an
audio file sent from the image tab still gets "recording" in its headline rather than "photograph".
And `renderMediaMeta` surfaces the audio facts the server reported, the analysed duration, the
original duration when it was cut, the window count, the sample rate conversion and the downmix, none
of which existed in the UI before audio and none of which the image path emits. The tab itself only
chooses the intro copy and which extensions the picker hints at; it never decides how a result is
read. The file picker also does not hard refuse on `file.type`, since that string is empty for many
real audio files, so an unrecognised file is sent and the server's 400 names the real problem.

## 8. Evaluation and calibration

Nothing in this section has been run yet. The machinery exists and `backend/eval_data/` is empty. It is
also image only: all three scripts walk image folders, so the audio pathway cannot be calibrated with
what is in the repo today and its responses report `calibrated: false` regardless of what you run.
Extending the harness to audio is the honest next step for that pathway and it has not been done.

`eval/collect.py` walks a labelled directory tree, runs the identical pipeline through `Engine`, and
writes one JSONL record per image with the path, the label where 1 means AI, the source taken from
the first path component under the root, the score, the verdict, `overridden_by`, each signal's
rounded reading and the face count. Inference runs once so that both downstream scripts are cheap.

`eval/run_eval.py` reads those records and reports ROC AUC with tie correction, a confusion matrix,
and a breakdown by verdict band and by source. It reads the recorded `score_ai` and does not re-fuse,
which is why comparing two fusion settings means running `collect.py` twice into separate files.

`eval/calibrate.py` fits a logistic regression on the logits of the recorded readings, with absent
signals imputed as zero and the same clamp applied, on a 70 percent train split, reporting holdout
AUC, confusion and Brier score on the rest. It refuses to fit below 25 usable records per class. It
drops every record carrying `overridden_by` unless told otherwise, printing the breakdown by override
name and an extra warning when `face_pathway` appears, because those records were not produced by the
expression being fitted. On a portrait heavy set with `face_decides` on, that is most of the set,
which makes `face_pathway`'s coefficient the weakest number in the resulting file. It writes the
bias, the per signal coefficients, the `signal_clamp` it fitted with, the fit date and the holdout
metrics.

Once `calibration.json` exists, serving changes in three ways. `_score_calibrated` replaces
`_score_prior`, so the score is the fitted regression rather than a weighted mean of priors. The
recorded clamp is used in preference to the runtime one, with any mismatch disclosed in the notes. And
`calibrated` becomes true in the response, which is what the UI keys its "uncalibrated" warning on.
Runtime signals with no fitted coefficient are excluded from the score and named, never given a
guessed weight.

## 9. How it behaves when things break

Every failure mode here is deliberate and reported rather than silent.

A dead, renamed or unreachable checkpoint walks its fallback chain and, failing that, drops out of the
ensemble with a reason in `/api/v1/models`. Two keys resolving to the same weights loses the second
with an explanation. A checkpoint whose labels cannot be resolved by name is refused rather than
guessed at. A local checkpoint whose path does not exist reports a missing directory, and one that is
not an image classifier reports the `model_type` and `architectures` its own `config.json` declares.
A malformed `models.local.json` degrades to no local models with the reason reported.

A corrupt or malformed `calibration.json` degrades to uncalibrated, because it is read inside the
FastAPI lifespan where an exception would abort the boot. Non finite coefficients are refused there.

A missing or LFS pointer YuNet file falls back to Haar in auto mode with a warning in the response
that Haar misses turned and small faces. A present but invalid file is reported differently from a
missing one.

A detector that throws during a forward pass returns an unusable result and its error text lands in
the response's `errors` array, leaving the rest of the ensemble to decide. A non numeric reading is
treated as a fault, dropped and named, never as a neutral 0.5 that would drag every verdict toward
the middle. A Grad-CAM failure returns no heatmap and nothing else changes.

An image that is empty, oversized, of an unaccepted format, truncated or too many pixels is rejected
with HTTP 400 and a specific message. An upload over the byte limit gets 413.

Audio fails the same way, with its own reasons. An empty, oversized or unrecognised file, or one under
0.1 seconds, is rejected with 400 and a message naming which of those it was. A container no installed
backend can decode is refused with a message saying the decoder is missing rather than that the format
is unsupported, since installing PyAV fixes one and nothing fixes the other. A rate conversion with no
anti-aliasing resampler available is refused outright rather than performed unfiltered, because naive
decimation would manufacture the artifacts the detectors look for; an installed resampler that throws is
reported rather than silently worked around. Windows that are near silent are skipped with a note, and a
file that is silent throughout still returns its loudest window's reading with a note saying it should
not be trusted. A broken audio checkpoint produces exactly one error row rather than one per window.
Audio switched off by configuration returns a verdict with a note saying no model read the file, which
is the same shape as an empty ensemble rather than an error.

The floor case is an ensemble of size zero, which still serves requests and returns score 0.5,
verdict uncertain, and a note saying no detection model is loaded.

## 10. What is verified and what is not

Verified: 171 tests run through `python tests/run_tests.py` with no pytest dependency, 97 core plus
34 engine plus 40 audio, of which 170 pass and one fails on Windows for the reason described below.
They use stub detectors, so they verify wiring, threshold logic, fusion
arithmetic, provenance patterns, preprocessing rejection paths and configuration degradation, and they
verify nothing about accuracy. The audio tests are the one place real input is used rather than a mock:
`audio.py` needs no torch and decodes PCM WAV with the standard library, so they build actual WAV bytes
at four bit depths and run them through the real decoder, resampler and window planner. The torch free
core invariant is checked by importing the core modules and asserting torch, pydantic, transformers and
fastapi never appear in `sys.modules`, and `audio.py` is checked the same way with `sys.modules["torch"]`
blocked outright. Every response key is checked against `AnalyzeResponse` by parsing `schemas.py` with
`ast`. The app runs end to end and renders a real Grad-CAM against a real Swin checkpoint. A real model
verification run on 2026-08-21 showed three of four image checkpoints loading on CUDA from their primary
repos.

A full boot on 2026-08-24 loaded all eleven checkpoints on CPU with zero failures and the face backend
resolving to YuNet rather than the Haar fallback, the first time every audio member has loaded here.
A real image was pushed through `/api/v1/analyze` and returned a three signal image verdict with no
audio member counted, and a real WAV was pushed through and returned a seven signal audio verdict with
clamping and the spread guard both firing and no image member counted, so the modality routing, both
ensembles and the fusion are confirmed to run against the live server. What none of that measures is
whether any reading is correct, which is the subject of the next section.

One test fails on Windows for a reason that is the test's fault rather than the code's.
`test_local_specs_are_parsed_into_model_specs` asserts a spec resolves to `/opt/checkpoints/detect-3b`,
which has no drive letter and is therefore not an absolute path on Windows, so `_coerce_local_spec`
correctly resolves it against `BASE_DIR` and the equality fails. The fixture is POSIX specific and
predates the audio work.

Not verified, and this list is the honest answer to "how accurate is it":

- No labelled evaluation set has been run. There is no accuracy figure, no F1, no AUC, no
  precision or recall for this system, for either modality. The API reports `calibrated: false` and the
  score is an ordering of files against each other, not a probability.
- The ensemble weights and the temperature are priors chosen by hand, not fitted values.
- The thresholds 0.35 and 0.65 are chosen, not derived from any measured operating point.
- The face pathway has not been confirmed against real face crops. `face_vit` loads and the YuNet
  backend resolves, but a test image with no face in it exercises neither, so the crop, the ViT
  reading and the `face_decides` override remain unobserved. No locally registered checkpoint has ever
  been loaded.
- How any audio checkpoint scores on real speech. All seven now load and score, but the only clip
  pushed through them was a 220 Hz sine tone, which is not speech and is out of distribution for every
  member. Four of the seven read it as certain and two as near zero, which tells you they are running
  and that they disagree, and tells you nothing about accuracy. No genuine or synthesised human speech
  has been through this pathway.
- Compressed audio decoding. PyAV is installed here and the code path exists, but only the standard
  library WAV decoder has actually been exercised end to end. The `soundfile` route for FLAC and OGG
  and the `av` route for MP3 and M4A are covered by tests and by nothing else.
- The audio window and quantile defaults. Four seconds on a two second hop at the 0.9 quantile is
  reasoned rather than tuned, and the splice detection the overlap exists for has never been tested on
  an actual spliced recording.
- The false positive cost of the dissent floor is unknown.
- Whether `face_decides` is a net gain is unmeasured in both directions.
- The suspicion that small upsampled face crops read high is a hypothesis with no experiment behind
  it.

The previous version of this project claimed 99.6 percent accuracy with nothing behind it. That claim
is the reason the rule against inventing numbers is written into `CLAUDE.md` as a non negotiable, and
the reason this section exists.

## 11. File map

Backend, under `backend/veritrust/`:

- `main.py` FastAPI app, the four routes, the concurrency semaphore, the static mount. The only file
  that knows about HTTP.
- `engine.py` pipeline orchestration and the `Analysis` result object. Knows nothing about HTTP.
- `config.py` specs, thresholds, fusion config, settings, label vocabularies, local spec loading.
  No torch, no pydantic.
- `preprocessing.py` decode, validate, orient, downscale, crop. No torch, no pydantic.
- `audio.py` sniff, decode, downmix, resample, window, aggregate. No torch, no pydantic, and the
  three decode libraries are imported inside the functions that need them.
- `provenance.py` C2PA, XMP, EXIF and PNG metadata reading. No torch, no pydantic.
- `fusion.py` clamping, scoring, the three band verdict, escalation, the face override, spread,
  calibration loading and serving. No torch, no pydantic. Knows nothing about either modality.
- `faces.py` YuNet and Haar detection, file validation, box filtering and ordering.
- `schemas.py` the pydantic response models. Every emitted key must be declared here, including the
  audio fields on `MediaOut`.
- `explain.py` Grad-CAM. Torch imported inside functions.
- `detectors/hf_image.py` the classifier wrapper, processor loading, label resolution, prediction.
  Torch imported inside functions.
- `detectors/hf_audio.py` the audio classifier wrapper, feature extractor loading, the same label
  resolution, per window prediction. Torch imported inside functions.
- `detectors/registry.py` builds all three pathways, records failures, deduplicates per pathway,
  reports status.

Also under `backend/`: `eval/collect.py`, `eval/run_eval.py`, `eval/calibrate.py` and
`eval/DATASET.md`, all image only; `eval_data/` for the labelled set, gitignored but with the tree
committed; `scripts/` for model prefetch, verification and the YuNet download, with prefetch and
verification each covering both modalities through their own auto classes; `tests/run_tests.py` plus
`test_core.py`, `test_engine.py` and `test_audio.py`; `weights/` for the YuNet ONNX file;
`models.local.json` optional, with `models.local.example.json` committed; `calibration.json` written by
the calibration script.

Frontend, under `frontend/`: `index.html`, `app.js`, `styles.css`. Served at `/` by the same process.

At the repo root: `CLAUDE.md` for the rules that govern changes here, `README.md`, `start.bat` which
activates `backend/venv`, checks that FastAPI imports and runs uvicorn on port 8000, and `legacy/`
which holds the dead v1 Flask app and its text classification component. Nothing imports from
`legacy/`.
