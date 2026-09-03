"""Run the pipeline over a labelled folder tree and dump per image signal values.

Both run_eval.py and calibrate.py consume the output, so inference happens once.

Expected layout, where the subdirectory under each root becomes the source label. That is what
makes a per generator breakdown possible, which matters because aggregate accuracy hides the
fact that most detectors collapse on generators they never saw.

    eval_data/real/dslr/...
    eval_data/real/phone/...
    eval_data/fake/sdxl/...
    eval_data/fake/midjourney/...

    python -m eval.collect --real eval_data/real --fake eval_data/fake --out eval_results/records.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from veritrust.config import settings
from veritrust.engine import Engine

SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def iter_images(root: Path, limit: int | None) -> list[Path]:
    files = sorted(p for p in root.rglob("*") if p.suffix.lower() in SUFFIXES and p.is_file())
    return files[:limit] if limit else files


def source_of(path: Path, root: Path) -> str:
    try:
        rel = path.relative_to(root)
    except ValueError:
        return root.name
    return rel.parts[0] if len(rel.parts) > 1 else root.name


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real", type=Path, required=True)
    parser.add_argument("--fake", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("eval_results/records.jsonl"))
    parser.add_argument("--limit", type=int, default=None, help="Max images per class.")
    args = parser.parse_args()

    for root in (args.real, args.fake):
        if not root.is_dir():
            print(f"Not a directory: {root}")
            return 2

    engine = Engine(settings)
    engine.load()
    status = engine.status()
    if not status["ensemble_size"]:
        print("No detector loaded. Run scripts/verify_models.py first.")
        return 1
    print(f"Ensemble: {status['ensemble_size']} model(s) on {status['device']}")
    print(f"Face backend: {status['face_backend']}")

    jobs = [(p, 0, source_of(p, args.real)) for p in iter_images(args.real, args.limit)]
    jobs += [(p, 1, source_of(p, args.fake)) for p in iter_images(args.fake, args.limit)]
    if not jobs:
        print("No images found.")
        return 2
    print(f"Scoring {len(jobs)} image(s). Label 1 means AI generated.\n")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    failed = 0
    started = time.perf_counter()

    with args.out.open("w", encoding="utf-8") as handle:
        for index, (path, label, source) in enumerate(jobs, start=1):
            try:
                analysis = engine.analyze(path.read_bytes(), want_heatmap=False)
            except Exception as exc:
                failed += 1
                print(f"  skip {path.name}: {type(exc).__name__}: {exc}")
                continue
            record = {
                "path": str(path),
                "label": label,
                "source": source,
                "score_ai": analysis.score_ai,
                "verdict": analysis.verdict,
                "overridden_by": analysis.overridden_by,
                # p_ai is rounded to 4 places by the API, so the fit sees slightly coarser inputs
                # than serving does. Measured effect on a fitted score is below 1e-6, which is far
                # under any threshold, but it is a real train and serve difference rather than none.
                "signals": {s["name"]: s["p_ai"] for s in analysis.signals},
                "faces": len(analysis.faces),
            }
            handle.write(json.dumps(record) + "\n")
            written += 1
            if index % 25 == 0 or index == len(jobs):
                rate = index / max(time.perf_counter() - started, 1e-6)
                print(f"  {index}/{len(jobs)} at {rate:.1f} img/s")

    print(f"\nWrote {written} record(s) to {args.out}")
    if failed:
        print(f"{failed} image(s) failed and were skipped.")
    print("Next: python -m eval.run_eval --records " + str(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
