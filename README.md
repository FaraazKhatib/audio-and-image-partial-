# VeriTrust

Media authenticity checking, for images and for audio. Upload a file, get a three band verdict on
whether it is a real photograph or recording or was generated, with each contributing signal shown
separately.

On an image the system runs three independent checks and fuses them in logit space:

1. **Whole image synthetic detection.** An ensemble of Hugging Face image classifiers looking for
   the frequency and texture fingerprints diffusion and GAN pipelines leave behind.
2. **Face manipulation detection.** If faces are found, each is cropped with margin and scored by
   a face specific model. Face swaps leave local blend boundary artifacts that a whole image model
   largely misses, which is why this is a separate pathway rather than the same model.
3. **Provenance.** C2PA manifests, IPTC `digitalSourceType`, Stable Diffusion parameter blocks and
   generator strings in EXIF or XMP. This evidence is asymmetric: its presence is close to
   conclusive and hard overrides the models, while its absence means nothing at all, because
   screenshots and social platforms strip metadata from real photographs too.

On an audio file it runs an ensemble of voice spoofing and synthetic speech detectors over
overlapping windows instead. Same fusion, same three bands, same per signal display. The modality is
chosen from the file's own header rather than from anything the browser claimed, and "The audio
pathway" below covers what it does and what is unknown about it.

The verdict is `likely_authentic`, `uncertain`, or `likely_ai_generated`. The middle band is a
deliberate abstention rather than a rounding artifact.

`working.md` walks the whole pipeline start to end, from the upload arriving to the verdict being
drawn, including every default and the order the fusion checks run in. Read that if you want to know
what the code does. Read `CLAUDE.md` if you are going to change it.

## How fusion actually behaves

Worth reading before interpreting a result, because the aggregation is not a plain average and the
reasons are not cosmetic. Every number below was produced by running the code, and the case they
come from is named each time, because two different cases are easy to conflate here.

Signals are combined in logit space, which preserves the strength of agreement but is unbounded. A
single member reporting 0.001 contributes about -6.9 and can outvote every other signal by itself.
That produced a real failure. On a generated image the app returned a score of 0.012 with a 97
percent margin and the verdict "likely a real photograph", off two whole image models reading 0.219
and 0.001. Every model reading is therefore clamped to `VT_SIGNAL_CLAMP` (0.02 by default) before
its logit is taken. The raw reading is still what the UI shows, with a `capped` marker when the
clamp bit.

Reconstructing that failure from the two readings as displayed gives 0.016, not 0.012, because the
per signal figures on screen are rounded to three places. The reconstruction is what the figures
below are computed from, so they are consistent with each other rather than with the screenshot to
the last digit.

The clamp reopens the vote but does not decide it. What stops a wrong verdict is a second rule: when
any single detector clears `ai_min`, the verdict is floored at `uncertain` and the dissenter is named
in `escalated_by`. The reasoning is that these detectors recognise the generator fingerprints they
were trained on and fall back to "real" on anything unfamiliar, including recompressed images and
output from newer models, so silence from a member is weak evidence while a positive is strong.
Averaging the two symmetrically manufactures false negatives.

**On the observed case those two rules are not enough.** The clamp lifts the reconstructed score
from 0.016 to 0.070 and the verdict stays `likely_authentic`, because neither model that loaded ever
cleared `ai_min`, so there is no dissenter to escalate on. What does change is the reported margin,
which falls from 95 to 65. The old code reported the full band margin there because its probability
space spread check waved that pair through as agreement; the logit space check does not, and cuts the
margin accordingly. Fusion cannot invent evidence: two models both saying "real" will produce "real",
and the actual gap on that image is that only two of four configured detectors were loading and the
face pathway had never run.

Where the rules do decide the outcome is once the ensemble is complete. Add a third whole image
model at 0.60 and a face model at 0.85 to those same two readings and the old math returns 0.191,
still `likely_authentic`. Clamping lifts it to 0.334, which is *still* inside the authentic band
(edge 0.35). Only the escalation rule then produces `uncertain`, naming the face model. Neither fix
is sufficient alone, which is why both are in.

That trade is deliberate and it costs something. Expect more `uncertain` verdicts and more false
positives than a plain average would give. `VT_DISSENT_ESCALATES=false` restores plain averaging.

The score itself is never patched by the escalation rule, because the score is what `eval/` ranks
and what calibration is fitted against. So an escalated result shows a needle in the authentic band
next to an `uncertain` verdict. That is intended, and the UI explains it in place.

### The face pathway decides on its own, in both directions

This is the one place the project lets a single uncalibrated model overrule the ensemble, and it
contradicts a rule stated elsewhere in this repo on purpose. It was chosen deliberately on
2026-08-21, with the cost on the table, and the cost is stated below rather than buried.

When faces are found and the face model's reading lands outside the uncertain band, that reading
becomes the verdict and the score. Nothing is fused. It reports through `overridden_by` exactly like
a provenance override, and the face box in the UI is coloured by the same reading, so the box and the
headline can no longer disagree. This is disabled by default: set `VT_FACE_DECIDES=true` only after
measuring it on a representative face-swap set. With the default, face and whole-image signals are
fused and the dissent floor remains active.

The reason is that on a face swap the whole image detectors are answering a different question. Every
pixel outside the swapped region really is camera output, so reading "photograph" is correct of them
and still wrong of the image. Averaging that against the face model buries the finding: a face read at
1.00 against three whole image readings of 0.02 fuses to 0.174, deep in the authentic band. The
escalation rule stopped the wrong *verdict* there but left the needle at 0.174 under it, which reads
as a system arguing with itself.

**What this costs.** The rule is symmetric, so it also closes cases in the authentic direction. A face
read at 0.04 over a whole image detector reading 0.97 now returns `likely_authentic` at 0.04. That is
the one situation where this build calls an image real over a detector that flagged it, and a face
generator this model does not recognise is exactly the case that produces it. The response notes name
the overruled reading and say plainly that it cleared the AI threshold and was overruled anyway.
Nothing about the trade in either direction has been measured.

Three limits keep it narrow. Provenance still outranks it, because provenance reads evidence rather
than estimating. A face reading inside the band decides nothing and falls through to ordinary fusion.
And a face weight muted to zero cannot decide, since a zero weight is an instruction to ignore that
detector.

One consequence for calibration: every request with a decisive face becomes an override record, and
`eval/calibrate.py` drops override records because the fit never applies to them. On a face heavy
evaluation set that can remove enough records to fall under the 25 per class floor. `calibrate.py`
prints the dropped count broken down by source for that reason.

Disagreement is measured in logit space too. On the probability scale 0.219 against 0.001 is a 0.22
gap that passes any reasonable tolerance, while as raw odds it is a factor of 280. Spread is computed
after clamping, so the figure that pair actually reports is a factor of 14, still well past the
default limit of 2.2 logits. These models cluster hard against the ends of the range, which is
precisely where the probability scale stops discriminating.

Two eligibility questions get answered separately, and conflating them was a bug. Whether a signal
contributed to the score is reported as `counted` per signal, false when its weight is muted to zero
or when the loaded calibration has no coefficient for it. Whether a signal may escalate the verdict
is a different test: everything the operator has not explicitly muted qualifies, including a
detector with no fitted coefficient. A model that observed something and has not yet been calibrated
is a reason to abstain, not a reason to ignore it, so it can floor the verdict while contributing
nothing to the needle. The UI marks those readings `not counted` and prints no weight for them.

A detector returning a non numeric reading is discarded with a note naming it, rather than being
counted as an undecided 0.5. A NaN survives every clamp, comparison and sigmoid it passes through,
and FastAPI serialises it as a bare `NaN` token, which is not valid JSON and fails in the browser
before any of the reasoning above reaches the reader. A faulty detector is also a different thing
from an undecided one and should not be silently averaged in as neutral.

## The audio pathway

Same fusion, same bands, same honesty rules. What differs is everything before fusion, because a
recording is not one sample the way an image is one frame.

**Seven checkpoints across three architecture families.** Five wav2vec2 or XLS-R waveform models, one
WavLM, and one Audio Spectrogram Transformer that reads a mel spectrogram as a patch grid. The AST
carries a weight of 1.2 rather than 1.0, which is the only non uniform weight here and is a prior
rather than a measurement: an ensemble of one architecture largely agrees with itself, and that model
is the only member positioned to disagree for a structural reason instead of a training data one.
Their class orders genuinely conflict. One reports `0=fake 1=real`, two report `0=real 1=fake`, three
use the anti-spoofing vocabulary instead of "real", two of those as `bona-fide` and `spoof` and one as
`Bonafide` and `Spoof`, and one uses `AIVoice` and `HumanVoice`. Every one resolves by label name, none
by index, and a checkpoint whose labels stop being recognisable refuses to load rather than silently
inverting its predictions.

**Overlapping windows, not one score for the whole file.** Four second windows on a two second hop.
A cloned sentence spliced into an otherwise real recording is a small fraction of a three minute
file, and one score for the whole clip averages it into nothing. Overlap matters because a splice
that lands on a window boundary is split across two windows and weakened in both.

**Windows combine at the 0.9 quantile, deliberately not at the max.** A max over N windows only ever
climbs as N grows, so a longer recording would score higher for being longer, which is a length
detector wearing a spoofing detector's clothes. The quantile interpolates, so it sits below the
loudest window and does not drift upward with duration. Above 24 windows the cap samples across the
whole file rather than taking the first 24, because the first 48 seconds of a long recording is not a
sample of it.

**Silent windows are skipped and said to be skipped.** A window below the RMS floor carries no voice
to judge, and scoring room tone produces a reading about nothing. If every window is silent the
loudest one is kept anyway with a note saying it should not be trusted, because returning no signal
at all on a file that clearly contained audio is worse than returning a flagged one.

**Resampling refuses rather than degrades.** These checkpoints expect 16 kHz. Naive decimation
aliases high frequency content down into exactly the band the spoofing detectors read, manufacturing
the artifacts they are looking for, so a request is rejected with the reason named when no
anti-aliasing resampler is available. The message says the decoder or resampler is missing rather
than claiming the format is unsupported, since those call for different fixes.

**There is deliberately no `audio_decides`.** `face_decides` is the single declared exception to the
rule that no uncalibrated model holds a veto, and it was a product decision with its cost written
down. Audio reaches the verdict through fusion plus the dissent floor like everything else. A
weighted quorum on escalation was considered while adding the pathway and rejected: it would have
changed the escalation rule for images too, which is not something a new modality gets to do
quietly.

**Nothing here is measured.** No labelled audio set has been run, so `calibrated` is `false` for
audio exactly as it is for images, the fused audio score orders files rather than estimating a
probability, and the window and quantile defaults are reasoned choices rather than tuned ones.

## Status

Written and structurally verified, not yet benchmarked. 171 tests cover preprocessing, provenance
parsing, fusion, calibration handling, label resolution, registry assembly, local checkpoint
registration, face detector validation, audio decoding and windowing, and engine orchestration. 170
pass; the one failure is a POSIX specific path in a test fixture that fails on Windows, described under
"Tests". Nothing has been run against a labelled dataset yet, so **there are no accuracy numbers here
and you should not assume any.** See "Verification state" below for the exact split.

That applies with full force to the audio pathway, which is newer than the rest. All seven audio
checkpoints have been observed loading and scoring on a real file, across three architecture families,
each resolving its classes by name and reporting its load state through `/api/v1/models`, and the
decoding, resampling, windowing and aggregation are covered by tests. None of that is a measurement.
The only clip put through them was a synthetic tone, not speech. No audio file has been scored against
a labelled set here, `calibrated` is `false` for audio exactly as it is for images, and the fused audio
score is an ordering rather than a probability.

One confirmed miss so far: a generated image was called authentic, reported at score 0.012 with a 97
percent margin. Two fusion bugs behind that margin are fixed and covered by regression tests, but
**that image would still be called authentic today**, because only two of the four configured
detectors were loading and both of those read low. What the fixes change is the confidence: on the
reconstruction from its two published readings the margin drops from 95 to 65. Fusion cannot
manufacture evidence the detectors did not produce, so the remaining work there is getting the full
ensemble loading and then measuring it against labelled data.

Two reasons for that shrunken ensemble have since been found, both of which reported themselves as
mere degradation rather than as faults:

- The YuNet face detector file in this repo was a 131 byte Git LFS pointer rather than a model, so
  every request silently fell back to the Haar cascade and the face pathway contributed nothing on
  anything but a large frontal face. It is now validated rather than merely checked for existence,
  which also means the face model's real behaviour has still never been observed.
- `siglip_ai_human` never loaded on any machine, because it labels its two classes `ai` and `hum` and
  label matching was exact, so `hum` did not match `human`. Matching now allows a prefix in either
  direction. A verified run reported three of four checkpoints usable for this reason.

Both fixes restore detectors that were absent. Whether the restored ensemble is more accurate is
unknown until it is measured, and neither fix has been.

Restoring `siglip_ai_human` has one consequence worth stating before it surprises anyone. It is a
third whole image detector, and on a face swap the whole image detectors are right to read
"photograph", because outside the swapped region that is what the pixels are. On a reconstruction
where the face model reads 1.00 and the whole image models read 0.02, going from two of them to
three moves the fused score from 0.274 to 0.174, further into the authentic band. That dilution is
what `VT_FACE_DECIDES` was added to answer. It is now opt-in, so those two figures describe the
default fused result; enabling the setting instead reports the face reading directly. Whether that
override or the pathway weighting improves outcomes is a question for measurement rather than
intuition, and `eval/DATASET.md` asks for face swaps as their own source folder so the per-source
breakdown can answer it.

## Quickstart

Python 3.10 or newer. On Python 3.13 note that torch published no wheels before 2.6, which is why
`requirements.txt` uses a floor rather than an exact pin for the ML stack.

```bash
git clone https://github.com/aryan-indra/VeriTrustAI.git
cd VeriTrustAI
setup.bat            # Windows
./setup.sh           # macOS or Linux
```

Then start it:

```bash
start.bat            # Windows
./start.sh           # macOS or Linux
```

Open http://localhost:8000. The frontend is served by the API at the same origin, so there is no
separate dev server and no CORS setup.

`setup.bat` and `setup.sh` create `backend/venv`, install `backend/requirements.txt`, fetch the
YuNet face detector into `backend/weights/`, prefetch the checkpoints from the Hub and then run
`verify_models.py` to report what actually loaded. The prefetch is the slow part on a cold cache and
it prints each checkpoint as it goes. Neither the prefetch nor the verification is treated as fatal,
because a checkpoint that cannot be downloaded degrades the ensemble and reports itself at runtime
rather than preventing the service from starting.

**A fresh clone has no weights.** `backend/weights/` is empty because the YuNet ONNX file is not
committed, and no model checkpoints are committed either. The setup script fetches both. If you skip
it, the face pathway falls back to the Haar cascade bundled with OpenCV, which only reliably finds
roughly frontal faces, and the ensemble downloads its checkpoints on first request instead.

### GPU

The default PyPI torch wheel is CPU only on Windows, which works but is slow. Pass a CUDA tag to the
setup script to get a GPU build:

```bash
setup.bat cu124      # Windows
./setup.sh cu124     # Linux
```

That installs torch and torchvision together from the PyTorch index before anything else. They have
to go in one command, because torchvision pins an exact torch version and pulling it from PyPI
afterwards can silently replace a CUDA torch with a CPU one.

Check the current CUDA tag with the selector at https://pytorch.org/get-started/locally/ first, since
the available builds change over time. `scripts/verify_models.py` reports which device was resolved
and, if it landed on CPU, whether that is because the torch build has no CUDA support or because no
GPU is visible.

### Manual setup

The scripts do nothing clever, so if you would rather drive it yourself:

```bash
cd backend
python -m venv venv
venv\Scripts\activate                # Windows
# source venv/bin/activate           # macOS or Linux
pip install -r requirements.txt
python -m scripts.download_models    # prefetch checkpoints and YuNet face weights
python -m scripts.verify_models      # confirm each repo resolves and its labels are readable
uvicorn veritrust.main:app --reload --port 8000
```

Once everything runs, pin what actually worked:

```bash
pip freeze > requirements.lock.txt
```

`verify_models.py` is the step worth reading. It reports the device it resolved, the state of the
YuNet file, which local checkpoints were declared and found, then for each spec the label mapping it
reports and which class index was resolved as "fake". If a repo has been renamed or deleted, it fails
there with a clear message instead of at request time. It exits nonzero if nothing loaded.

The first request is slow while weights load into memory. Later requests reuse them.

### Why the YuNet download is not a plain URL fetch

`download_models.py` fetches YuNet from `media.githubusercontent.com`, not
`raw.githubusercontent.com`. opencv_zoo tracks the ONNX file with Git LFS, and the raw host serves
the LFS pointer for those: 131 bytes of text beginning `version https://git-lfs.github.com/spec/v1`.
That file satisfies every existence check, so the download script reported success on every run and
`faces.py` accepted it, then failed inside `cv2.FaceDetectorYN.create` and quietly demoted the
request to the Haar cascade. **This shipped in the first version, and it is why the face pathway
never appeared to find anything.** Both files now validate the bytes rather than the path, using the
same check, and a present but invalid file is reported differently from a missing one.

If you have an old copy of that file, the download script will replace it in place.
`verify_models.py` names it explicitly:

```
[fail] YuNet file is unusable: ...face_detection_yunet_2023mar.onnx is a Git LFS pointer,
       not a model. It is 131 bytes of text.
```

## Adding your own checkpoints

Weights that are private, internal or otherwise not on the Hub can join the ensemble without code
changes. Copy `backend/models.local.example.json` to `backend/models.local.json` and point each
entry at a directory:

```json
{
  "models": [
    { "key": "detect_world", "path": "weights/detect-world", "kind": "synthetic", "weight": 1.0 }
  ]
}
```

`key` is any name not already used by a built in spec. `path` must be a directory in Hugging Face
layout, holding `config.json`, the weights and `preprocessor_config.json`; a relative path resolves
against `backend/`. `kind` is `synthetic` for whole image models or `face` for models that expect a
face crop, and it decides which pathway the model votes in. `weight` is a prior, exactly like the
built in weights, and carries no implication that the model has been evaluated. `fake_index` is an
escape hatch for checkpoints whose labels are generic such as `LABEL_0`; leave it out unless you have
confirmed the class order against known samples, because guessing it inverts every prediction
silently.

Nothing about this path is guessed. A missing directory, a malformed JSON file, a duplicate key or an
unknown `kind` degrades to an ensemble without that model and reports the reason through
`/api/v1/models`, on the same principle as a dead Hub repo. `models.local.json` is gitignored, since
it holds machine specific paths.

Two limits are worth knowing before you drop weights in. The wrapper is
`AutoModelForImageClassification` and resolves classes by matching label text from
`model.config.id2label`, so a checkpoint that is not an image classifier will not load. If that
happens, the failure names the `model_type` and `architectures` its `config.json` declares, so a
vision language model reports itself as one instead of producing an opaque error. Such a model needs
its own detector wrapper implementing the `Detector` interface in `detectors/base.py`, because
generated text is not a probability and there is nothing honest to feed fusion. Second, a new model
starts uncalibrated and unmeasured. Adding it changes the score distribution, so any existing
`calibration.json` was fitted without it and its coefficient is absent; the response notes will name
the signal as excluded from the fit until you rerun `eval/calibrate.py`.

`verify_models.py` reports which local paths exist, which loaded, and what labels each resolved.

## API

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/api/v1/health` | Liveness plus whether any detector is ready |
| GET | `/api/v1/models` | Loaded detectors, failures with reasons, device, thresholds, calibration state |
| POST | `/api/v1/analyze` | Multipart upload, image or audio, modality read from the file's bytes |
| POST | `/api/v1/analyze-base64` | JSON body with base64 content, same sniffing |

Interactive docs at `/docs`.

There is one upload field for both modalities on purpose. The client does not declare what it has,
because it is not a reliable witness: `file.type` in a browser comes from the operating system's
association table rather than from the file, and it is routinely empty for `.flac`, `.m4a` and
`.opus`. The magic bytes decide, and a file that sniffs as neither is refused with a message naming
the real problem.

The `image` object on the response carries whichever fields apply to the modality that was analysed,
which is also how a caller can tell which one that was. Audio adds `duration`, `sample_rate`,
`windows_scored`, and the before values `original_duration`, `original_sample_rate` and
`original_channels` alongside the `truncated` and `downmixed` flags that say those conversions
happened. Anything the analysis changed about the input is reported rather than applied silently.

Two fields on the analyse response exist for the frontend rather than for a caller making a decision.
`escalated_by` names the check that stopped an authentic verdict, and `spread_exceeds_limit` with
`logit_spread` say whether the reported confidence was reduced by members contradicting each other.
A confidence of zero otherwise has two indistinguishable causes, since every score inside the
uncertainty band has no margin by definition, and the two need telling apart before anything is
rendered. Every key the engine emits has to be declared in `AnalyzeResponse`, because FastAPI filters
the payload through it and drops undeclared fields with a 200 and no warning.

`overridden_by` takes two values a caller should treat differently. `provenance` means metadata was
read and the models were not consulted. `face_pathway` means a single uncalibrated face model was
outside the bands and took the verdict, with `spread_exceeds_limit` then describing disagreement that
was overruled rather than averaged. On that path `confidence` is the band distance of one raw model
reading, so a face at 1.00 computes 100 percent and means nothing of the sort; the frontend
deliberately does not print it.

`GET /api/v1/models` is worth reading before trusting output. It reports failed checkpoints
verbatim, so a degraded ensemble is visible rather than silent. It also reports specs that loaded
but were dropped for resolving to a checkpoint another spec had already claimed, which happens when
a model's own repo is unavailable and its fallback belongs to a sibling. Those are dropped rather
than kept, because the same weights under two names would vote twice at double weight and agree
perfectly with themselves.

## Calibration

Out of the box, `calibrated` is `false` in every response and the score is an **ordering, not a
probability.** Two images can be compared, but "0.82" does not mean an 82 percent chance of being
generated. Fixing that needs your own labelled data. `eval/DATASET.md` covers what to collect and
what makes a set worth having; `backend/eval_data/` already has the folder tree.

```bash
# eval_data/real/<source>/*.jpg  and  eval_data/fake/<generator>/*.jpg
python -m eval.collect --real eval_data/real --fake eval_data/fake
python -m eval.run_eval --records eval_results/records.jsonl
python -m eval.calibrate --records eval_results/records.jsonl
```

Subdirectory names become source labels, so the report includes a per generator AUC breakdown.
That matters more than the headline number, since detectors typically do well on the generators
they were trained on and poorly on newer ones.

`calibrate` drops any record whose verdict bypassed fusion, from provenance or from the face pathway
deciding alone, since the fit never applies to those. It prints the dropped count by source. With
`VT_FACE_DECIDES` enabled and faces in most of your images, that can be most of the set, so check the count
against the 25 per class floor rather than the number of files you collected. `--include-overrides`
keeps them, at the cost of fitting on records the fitted expression never produced.

`calibrate` fits a logistic regression over per signal logits on a train split and reports AUC and
Brier score on a holdout split, then writes `calibration.json`. Fusion evaluates that fitted
expression directly, coefficients and intercept as they were fitted, so there is no train and serve
mismatch to reason about. A signal missing at request time contributes exactly what the fit imputed
for it, which is nothing, and a signal the fit never saw is excluded and named in the response notes
rather than given a guessed coefficient. It refuses to fit on fewer than 25 examples per class.
Restart the API to pick up the file; responses then report `calibrated: true` along with the
source. Those metrics describe your evaluation set, not images in general.

`calibration.json` also records the `signal_clamp` the fit ran with, because coefficients only
describe the feature transform that produced them. Serving applies the recorded clamp in preference
to `VT_SIGNAL_CLAMP` and says so in the notes when the two differ, rather than quietly evaluating a
different function than the one that was fitted. A file that predates this key falls back to the
runtime value with a note flagging the ambiguity.

An unreadable or malformed `calibration.json` degrades the service to uncalibrated instead of
stopping it. Loading happens during startup, so a raised exception there would mean the API refuses
to boot over a bad optional file. Non finite coefficients are refused on the same pass, since a NaN
would propagate into every score and JSON cannot represent the result.

**The eval harness is image only.** `collect.py`, `run_eval.py` and `calibrate.py` walk image
folders, so the audio pathway cannot be calibrated with what is in the repo today and its responses
will report `calibrated: false` no matter what you run. Extending the harness to audio is the honest
next step for that pathway, and it is not done. Nothing about the audio signals is wired into a fit
in the meantime, which means no guessed audio coefficients exist either.

## Tests

From `backend/`, with the venv active:

```bash
python tests/run_tests.py     # no pytest needed
pytest tests                  # also works, if you installed pytest separately
```

The tests use stub detectors and download nothing, so they run on a machine with no GPU and no
weights. They verify wiring, threshold logic, fusion arithmetic, provenance parsing, preprocessing
rejection paths and configuration degradation. They verify no accuracy, because there is none to
verify yet.

The audio tests are an exception to the stub rule in one respect. `audio.py` imports no torch and
decodes PCM WAV with the standard library, so the audio tests build real WAV bytes and push them
through the real decoder, resampler and window planner rather than mocking any of it. The detectors
are still stubs. A test that mocks `decode_audio` proves only that a mock was called.

One test fails on Windows, and the fixture rather than the code is at fault.
`test_local_specs_are_parsed_into_model_specs` asserts a spec resolves to `/opt/checkpoints/detect-3b`.
That path has no drive letter, so it is not absolute on Windows and `_coerce_local_spec` correctly
resolves it against `BASE_DIR`, which is the documented behaviour for a relative path. The assertion is
POSIX specific and predates the audio work. On Linux and macOS all 171 pass.

## Layout

```
setup.bat / setup.sh    one time setup: venv, dependencies, weights, verification
start.bat / start.sh    serve on http://localhost:8000
working.md              the whole pipeline start to end, in detail
CLAUDE.md               the rules that govern changes here, and the reasons behind them
backend/
  models.local.example.json   template for registering checkpoints held on disk
  veritrust/
    config.py           model specs, thresholds, env driven settings, local spec loading
    preprocessing.py    decode, magic byte sniffing, size limits, EXIF orientation, cropping
    audio.py            decode, downmix, resample, overlapping windows, quantile aggregation
    provenance.py       C2PA, XMP, EXIF and PNG text metadata inspection
    detectors/
      base.py           detector interface and result type
      hf_image.py       Hugging Face classifier wrapper, name based label resolution
      hf_audio.py       Hugging Face audio classifier wrapper, same label resolution
      registry.py       builds the ensemble, records per checkpoint failures
    faces.py            YuNet then Haar face detection, validates the YuNet file itself
    fusion.py           logit space weighted fusion, override handling, three band verdict
    explain.py          Grad-CAM for conv, Swin and ViT activation shapes
    engine.py           pipeline orchestration, HTTP agnostic
    schemas.py          pydantic response models
    main.py             FastAPI app, routes, static frontend mount
  scripts/              download_models.py, verify_models.py
  eval/                 collect.py, run_eval.py, calibrate.py, DATASET.md
  eval_data/            folder tree for your labelled set, images gitignored
  weights/              YuNet ONNX, fetched by setup, not committed
  tests/                test_core.py, test_engine.py, test_audio.py, run_tests.py
frontend/               index.html, styles.css, app.js
legacy/                 the previous version, kept for reference only, nothing imports it
```

Not in the repository, by design: `backend/venv/`, downloaded checkpoints, the YuNet ONNX file, your
images in `backend/eval_data/`, `backend/eval_results/`, `backend/calibration.json` and
`backend/models.local.json`. The last two are per deployment rather than shared, and the first three
are fetched or generated.

## Configuration

Every setting is an environment variable, listed in `veritrust/config.py`. The ones most worth
knowing:

| Variable | Default | Effect |
| --- | --- | --- |
| `VT_DEVICE` | `auto` | `cuda`, `mps`, `cpu` |
| `VT_AUTHENTIC_MAX` | `0.35` | Upper edge of the authentic band |
| `VT_AI_MIN` | `0.65` | Lower edge of the AI band. Widen the gap to abstain more |
| `VT_MAX_EDGE` | `2048` | Longest edge before downscaling, which is disclosed in the response |
| `VT_ENABLE_HEATMAP` | `true` | Grad-CAM overlay, the slowest part of a request |
| `VT_FACE_DETECTOR` | `auto` | `yunet`, `haar`, `off` |
| `VT_YUNET_PATH` | `backend/weights/face_detection_yunet_2023mar.onnx` | Validated, not just existence checked |
| `VT_LOCAL_MODELS` | `backend/models.local.json` | Checkpoints held on disk rather than on the Hub |
| `VT_MAX_CONCURRENCY` | `2` | Simultaneous inferences. Raise only if you have VRAM to spare |
| `VT_SIGNAL_CLAMP` | `0.02` | Bound on each model reading before its logit is taken |
| `VT_DISSENT_ESCALATES` | `true` | One detector past `ai_min` floors the verdict at `uncertain` |
| `VT_FACE_DECIDES` | `false` | Opt in only after evaluation: a face reading outside the bands takes the verdict outright, both ways |
| `VT_ALLOW_UNCALIBRATED_AUTHENTIC` | `false` | Keeps an uncalibrated all-negative image result at `uncertain`, not “real” |
| `VT_LOGIT_SPREAD_LIMIT` | `2.2` | Odds gap between members before confidence is cut, in logits |
| `VT_ENABLE_AUDIO` | `true` | Off refuses audio uploads with a note saying so, rather than pretending |
| `VT_AUDIO_WINDOW_SECONDS` | `4.0` | Window length fed to each audio checkpoint |
| `VT_AUDIO_HOP_SECONDS` | `2.0` | Window step. Overlap is what catches a splice on a boundary |
| `VT_AUDIO_MAX_WINDOWS` | `24` | Cap per file, sampled across the whole clip rather than the first N |
| `VT_AUDIO_QUANTILE` | `0.9` | How windows combine. See below for why this is not a max |
| `VT_W_AUDIO` | `1.0` | Audio pathway weight in fusion. Zero mutes it entirely |

Inference runs in a worker thread so the event loop stays responsive, with `VT_MAX_CONCURRENCY`
capping how many run at once. Concurrent forward passes contend for the same VRAM, so a burst of
uploads on an unbounded pool is an out of memory error waiting to happen.

## Verification state

Verified by running:

- Preprocessing: format sniffing, rejection of empty, malformed, oversize and pixel bomb input,
  EXIF orientation, downscaling, crop margin clamping.
- Provenance: SD parameter chunks, generator strings in EXIF, IPTC AI source declarations, camera
  signatures, no metadata returning no opinion, and no false positive from bytes outside parsed
  metadata regions.
- Fusion: logit and sigmoid, all three bands, agreement sharpening, disagreement reducing
  confidence, override short circuit, empty signal abstention, the clamp bounds and its per signal
  reporting, escalation from a lone dissenter, escalation leaving the score untouched, escalation
  acting as a floor and never a promotion, and the observed overconfident veto as an explicit
  regression test.
- Fusion edge cases: `VT_LOGIT_SPREAD_LIMIT` at zero and negative, a clamp high enough to make the
  outer bands unreachable being disclosed in the notes using the effective rather than the requested
  value, a zero weight signal being neither described as clamped nor able to escalate a verdict it
  cast no vote in, a non numeric reading being dropped with a note instead of averaged, and a set of
  readings that are all non numeric abstaining rather than emitting a score JSON cannot encode.
- The face override: a decisive face taking the verdict in each direction, the authentic direction
  naming the overruled detector that cleared `ai_min`, a reading inside the band falling through to
  fusion, provenance still outranking it, a muted face weight unable to decide, `VT_FACE_DECIDES=false`
  restoring fusion and the dissent floor with the fused score pinned, the score being the raw face
  reading, the overruled spread being reported when confidence cannot show it, the result staying
  uncalibrated even with a fitted calibration loaded so `calibrate.py` drops the record, only the
  deciding signal being marked `counted`, and a sweep across both band edges confirming a deciding
  face never reports `uncertain`. End to end through the engine as well, so the box and the verdict
  are confirmed to read the same number.
- Calibration: the served score equalling the fitted expression to within floating point, a signal
  absent at request time contributing exactly what the fit imputed for it, a negative coefficient
  being applied rather than dropped, a signal the fit never saw being excluded and named while still
  being allowed to escalate, a calibration matching no runtime signal abstaining, an empty weight map
  not counting as fitted, the recorded clamp overriding the runtime one, a file without a recorded
  clamp saying so, and eight shapes of malformed `calibration.json` degrading to uncalibrated instead
  of raising during startup.
- Label resolution: both class orders, alternate vocabularies, multiclass summing, refusal to guess
  on generic `LABEL_0` style labels, the four label maps the configured repos actually report, a
  truncated label such as `hum` matching `human`, a vocabulary word buried inside an unrelated word
  such as the `ai` in `painting` not matching, a label reading as both real and generated being
  refused rather than settled by search order, the failure naming what it could not recognise, and the
  vocabularies containing no bare digits.
- Engine: signal assembly, per detector failure isolation, face aggregation across faces and
  models, override propagation, downscale disclosure, response shape, status with nothing loaded.
- Registry assembly: two specs resolving to the same checkpoint loading only once with the drop
  reported, distinct checkpoints both loading, deduplication not masking an ordinary load failure,
  and the face and whole image pathways being allowed to share a checkpoint.
- Local checkpoint registration: a well formed `models.local.json` producing specs with relative
  paths resolved and absolute ones left alone, a bare list accepted, an absent file staying silent,
  a malformed or wrongly shaped file degrading with a reason, one bad entry not discarding the good
  ones, a local key being unable to shadow a built in one, duplicates within the file keeping the
  first, empty keys and paths refused, two local specs pointing at one directory loading once, config
  problems reaching `/api/v1/models`, a nonexistent directory failing with a message that names the
  path before torch is imported, a checkpoint that is not an image classifier having its declared
  architecture reported, and the shipped example file parsing.
- Face detector validation: a Git LFS pointer rejected with the fix named, a truncated file
  rejected, a plausible file accepted, and a missing file reported rather than raising.
- Audio decoding, against real WAV bytes rather than a mocked decoder: 8, 16, 24 and 32 bit PCM read
  to within a per width tolerance, stereo averaged to mono rather than one channel being picked,
  empty, oversize, unrecognised and sub 0.1 second input each refused with its own reason, an
  unreadable WAV degrading to a named rejection instead of raising, truncation and resampling both
  reported as before and after values, and non finite samples scrubbed.
- Audio resampling: an identity rate returning the same object untouched, refusal naming aliasing and
  false positives when no anti-aliasing resampler is available, an installed but failing resampler
  being reported rather than worked around, and a 300 Hz tone resampled from 44100 to 16000 keeping
  its dominant frequency bin within 5 Hz, which is what proves a real low pass ran rather than
  decimation.
- Audio windowing: one window for a clip shorter than the window, overlap confirmed by a later
  window starting before an earlier one ends, a tail window for the remainder, the window cap
  sampling across the whole file rather than the first N, silent windows skipped with a note, an
  entirely silent file keeping its loudest window with a note saying it cannot be trusted, and window
  bounds never running past the waveform.
- Window aggregation: the quantile interpolating below the maximum, out of range settings clamped,
  the result not drifting upward with clip length on a steady signal, and still responding to a
  couple of loud windows.
- Audio label resolution: camelCase labels split into tokens, the ASVspoof `bonafide` and `spoof`
  vocabulary, checkpoints disagreeing about class order each resolved correctly, `LABEL_0` style
  labels refused, and the audio vocabulary staying inert on image labels.
- Audio routing and reporting end to end through the engine: audio bytes reaching the audio pathway
  on their header alone, an image still reaching the image pathway, windowing and fusion notes both
  surviving into the response, a long recording reporting that it was cut, a broken checkpoint
  reporting once rather than once per window, an empty audio registry and `VT_ENABLE_AUDIO=false`
  each disclosing themselves, a lone audio dissenter flooring the verdict without moving the score,
  a non numeric audio reading dropped and named while the payload stays JSON serialisable, and
  audio metadata reported as unread rather than as clean.
- The torch free boundary holding for audio: `veritrust.audio` imports with `sys.modules["torch"]`
  blocked, the audio wrapper's local directory check runs ahead of its torch import, its `describe`
  keys match the image wrapper's, and an unloaded audio detector returns no reading rather than a
  neutral 0.5.

Verified by running on a real machine with weights loaded:

- The service boots, serves the frontend, loads checkpoints and returns an analysis end to end.
- The current image registry has one whole-image Hugging Face ViT and one raw FaceForensics++
  face-crop checkpoint. Historical references below to a four-model image ensemble are from an
  earlier registry and are not evidence about this configuration.
- `scripts/verify_models.py` is the source of truth for the exact models that loaded on a deployment;
  run it after setup and before collecting evaluation records. A successful forward pass proves only
  that the wiring works, never that a checkpoint is accurate on current generators.
- That the shipped YuNet file was a Git LFS pointer and not a model, by reading its bytes, and that
  `verify_models.py` now reports it as a failure rather than as present.

Not yet verified:

- **Anything about accuracy.** No labelled evaluation set has been run, so precision, recall and
  the per generator breakdown are all unknown. One generated image is confirmed to be called
  authentic, both before and after the fusion fixes, on an ensemble where half the detectors were
  not loading.
- `eval/calibrate.py` end to end. The fit needs scikit-learn and at least 25 labelled examples per
  class, and neither has been available here. The serving side of the same expression is unit
  tested against hand computed values, the fitting side is not.
- That the YuNet URL actually returns the model. The weight file present here validates and the face
  backend resolves to YuNet rather than the Haar fallback, so the file itself is good, but it was
  already in place; no download has been performed from this environment.
- Any locally registered checkpoint. The loading path is tested with stubs and with a nonexistent
  directory. No real private weights have been present, so nothing is known about whether a given
  checkpoint loads, what labels it reports, or how it scores.
- The face pathway end to end against a real face model, including whether YuNet finds faces
  reliably at the sizes social uploads arrive in.
- Whether the default ensemble weights are sensible. They are priors, not measurements.
- Whether the escalation rule's false positive cost is acceptable in practice. The direction of
  the trade is deliberate, the magnitude is unmeasured.
- Whether letting the face pathway decide is a net gain. It removes the dilution it was added for,
  and in exchange one uncalibrated model can now close a case as authentic over a detector that
  flagged it. Both directions are unmeasured, and the face model itself has never been observed
  running on a real face crop, so its reliability at either band edge is unknown.
- The `use_fast=False` image processor fallback in `load_processor`. Installing torchvision is the
  real fix and that path has not been exercised.
- **Anything about audio accuracy.** All seven checkpoints now load and score, so the pathway runs,
  but the only clip ever pushed through it was a synthetic sine tone, which is not speech and is out
  of distribution for every member. Four read it as certain and two as near zero, which confirms they
  are running and that they disagree, and establishes nothing about accuracy. No genuine or
  synthesised human speech has been through this pathway, and no labelled audio set has been scored.
- Compressed audio decoding. PyAV is installed here, so MP3 and M4A are not refused, but only the
  standard library PCM WAV decoder has actually been exercised end to end. The `soundfile` route for
  FLAC and OGG and the `av` route for the compressed containers are covered by tests and by nothing
  else.
- The audio window and quantile defaults. Four seconds on a two second hop at the 0.9 quantile is
  reasoned from how these checkpoints are trained and from the length independence argument above.
  Neither is tuned, and the splice detection the overlap exists for has never been tested on an
  actual spliced recording.

## Known limits

Recompression, screenshotting and resizing erode the high frequency cues these detectors read, and
push results toward `uncertain`. Detectors generalise best to generators they were trained on, so
output from newer models is detected less reliably. A confident score is not proof, in either
direction. The `confidence` field is the margin outside the uncertainty band, not a probability
that the verdict is correct.

Audio has the same shape of limit and one of its own. Lossy compression, phone codecs and re-encoding
strip exactly the fine detail spoofing detectors read, so a voice note forwarded through two
messaging apps is a harder case than the same audio as a WAV. Background music and overlapping
speakers are outside what these checkpoints were trained on. Detection is strongest against the TTS
and voice cloning systems each model saw, so a newer synthesiser is detected less reliably, and the
same asymmetry as images applies: a positive is stronger evidence than silence, which is why one
dissenting member can floor the verdict at `uncertain`.

Provenance is image only. There is no audio equivalent of C2PA parsing here, so audio metadata is
reported as unread rather than as clean, and an absent generator tag on an audio file means nothing
at all.
