# Building the evaluation set

Nothing in this project has an accuracy number yet. This is the file that changes that. Until it is
filled in, every score the API returns is an ordering rather than a probability, and the UI says so.

## Where the images go

`backend/eval_data/` already has the tree. Drop image files into the leaf folders, run the three
commands below from `backend/`, and nothing else needs configuring.

```
backend/eval_data/
  real/
    camera/       photos straight off a camera, ideally with EXIF intact
    phone/        phone captures, which are heavily processed in-camera
    web/          real photos that have been through a social platform and recompressed
  fake/
    whole_image/  fully generated images, Midjourney, SDXL, Flux, DALL-E, Imagen, whatever you have
    face_swap/    real photographs with a swapped or reenacted face
```

The folder name directly under `real/` or `fake/` becomes the source label, and `run_eval.py`
reports AUC per source as well as overall. That split is the point of the layout: an aggregate
number hides the fact that these detectors collapse on generators they were not trained on. Add
more leaf folders whenever you can name the generator, for example `fake/whole_image` split into
`fake/midjourney` and `fake/flux`. Deeper nesting is fine, only the first level is used as the label.

Accepted extensions are jpg, jpeg, png, webp, bmp, tif and tiff. `eval_data/` and `eval_results/`
are both gitignored, so nothing you put here gets committed.

## How many

`calibrate.py` refuses to fit below **25 usable images per class**, so 25 real and 25 AI is the
floor, and 30 per class is a more comfortable target. The count that matters is the one after two
filters:

Images that fail to decode are skipped and reported by `collect.py`.

Images whose verdict came from a provenance override are dropped by `calibrate.py`, because those
bypass fusion entirely and would teach the fit a relationship it never applies at serving time. This
matters more than it sounds: some generators write C2PA or XMP metadata, so downloading fresh AI
images may hand you a set where a chunk of the fake class is decided by metadata and never reaches
the models. `collect.py` records `overridden_by` per image, so you can count them before wondering
where the records went.

**Images with a decisive face reading are dropped for the same reason, and there will be far more of
them when the override is enabled.** With `VT_FACE_DECIDES=true`, any image where a face is found and the face
model reads outside 0.35 to 0.65 has its verdict taken from that model alone, so it is an override
record too. On a portrait heavy set that can be most of what you collected. `calibrate.py` prints the
dropped count broken down by source, so read that line rather than counting files. Three ways out, in
order of preference: collect more images without faces or with faces the model finds ambiguous; run
`collect.py` with `VT_FACE_DECIDES=false` so the fused score is recorded, accepting that you are then
calibrating a path that serving does not take while the setting is on; or pass
`--include-overrides`, which is the least honest of the three and fits on records the fitted
expression never produced.

Note also that fitting a coefficient for `face_pathway` at all is only meaningful for images where
the face reading was inside the band, since every other face image is excluded. Treat that
coefficient as the weakest number in the file.

Then 30 percent is held out for reporting, so 50 usable records means fitting on 35 and reporting on
15. Numbers off 15 held out images are indicative and nothing stronger. Say that when quoting them.

## What makes the set worth having

Include face swaps as their own source. This is the case where the whole image detectors and the face
model answer different questions, since the photograph really is camera pixels everywhere except the
face, and the per source breakdown is what will show whether the current pathway weighting handles
that or drowns it. It is also the case `VT_FACE_DECIDES` was turned on for, and this folder is the
only way to find out whether that was the right call. Comparing the two settings means running
`collect.py` twice, once with each, into separate record files: `run_eval.py` reads the `score_ai`
each record was written with and does not re-fuse, so rerunning it against one file cannot show you
the other setting.

Vary the real class. If every real image is a clean high resolution capture and every fake is a
downloaded JPEG, the fit learns compression rather than generation, scores well on the holdout, and
falls apart in use. Recompressed real photos are the ones that get called fake in practice, so they
belong in the set rather than being kept out of it to protect the number.

Do not include an image in both classes, and do not include screenshots of AI images in the real
class. A screenshot of a generated image is still generated.

## Running it

From `backend/`, with the venv active:

```
python -m eval.collect --real eval_data/real --fake eval_data/fake --out eval_results/records.jsonl
python -m eval.run_eval --records eval_results/records.jsonl
python -m eval.calibrate --records eval_results/records.jsonl
```

`collect.py` runs inference once and writes per image signal values, so the other two are cheap to
rerun and re-split. Check `scripts/verify_models.py` first: collecting with a degraded ensemble fits
a calibration for an ensemble you are not going to serve, and the fitted coefficients name the
signals they saw, so a detector that was missing at collection time is excluded at serving time and
reported in the notes rather than silently given a guessed weight.

`calibrate.py` writes `calibration.json`. Restart the API to pick it up, after which the response
carries `calibrated: true` and names what it was fitted on.

## Reading the output honestly

The holdout AUC and Brier score describe your set. They are not a general accuracy claim and must
not be written down as one anywhere, which is the single rule this project has never bent. The
previous version of VeriTrust claimed 99.6 percent accuracy with nothing behind it at all.

Watch for a non positive coefficient in the fit. `calibrate.py` prints a warning for any it finds and
applies it as fitted rather than dropping it. It means that signal pushes toward "real" as it rises,
which is either an uninformative detector or an inverted label mapping, and the second one looks
exactly like the first from the outside.
