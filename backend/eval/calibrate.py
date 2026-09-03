"""Fit per signal fusion coefficients from labelled records, then write calibration.json.

Until this runs, every API response carries calibrated=False and the score is only an
ordering. After it runs the score is a probability estimate, valid for data resembling what you
calibrated on and no further.

The fit happens on a train split and is reported on a held out split. Fitting and reporting on
the same data is how projects end up quoting 99 percent accuracy that evaporates in the wild.

The model is a logistic regression over per signal logits, z = sum(c_i * l_i) + b, and fusion
evaluates that same expression directly when a calibration file is present. An earlier version
claimed the fit could be re expressed in the weighted mean form fusion uses for priors, via
w_i = c_i and T = 1 / sum(c_i). That is false whenever a signal is missing at request time, because
the weighted mean renormalises by the weights present while the regression renormalises by nothing,
and false again for any non positive coefficient, which the weighted mean dropped. Both cases are
ordinary rather than exotic: most images have no faces and no metadata. Fusion now applies the
coefficients as fitted, so there is no equivalence left to get wrong.

    python -m eval.calibrate --records eval_results/records.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from veritrust.config import CALIBRATION_PATH, settings
from veritrust.fusion import clamp_probability, logit
from eval.run_eval import confusion, roc_auc

MIN_PER_CLASS = 25


def build_matrix(records: list[dict], names: list[str]) -> tuple[list[list[float]], list[int]]:
    """Missing signals are imputed as logit(0.5) which is zero, a neutral vote.

    Fusion omits an absent signal entirely, which adds zero to its sum, so the imputation here and
    the omission there agree by construction. The same clamp fusion applies at request time is
    applied here too, and the clamp used is written into calibration.json so serving can honour the
    one that was fitted rather than whatever is configured later. Fitting on unclamped logits would
    fit a different function from the one that actually runs.
    """
    clamp = settings.fusion.signal_clamp
    features: list[list[float]] = []
    labels: list[int] = []
    for record in records:
        row = [
            logit(clamp_probability(record["signals"][n], clamp)) if n in record["signals"] else 0.0
            for n in names
        ]
        features.append(row)
        labels.append(record["label"])
    return features, labels


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, default=Path("eval_results/records.jsonl"))
    parser.add_argument("--out", type=Path, default=CALIBRATION_PATH)
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--include-overrides",
        action="store_true",
        help="Keep records whose verdict bypassed fusion, from provenance or from the face pathway "
        "deciding alone. Off by default, since those would teach the fit a relationship it never "
        "applies.",
    )
    args = parser.parse_args()

    if not args.records.is_file():
        print(f"No records at {args.records}. Run eval.collect first.")
        return 2

    records = [json.loads(l) for l in args.records.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not args.include_overrides:
        dropped = sum(1 for r in records if r.get("overridden_by"))
        by_source: dict[str, int] = {}
        for r in records:
            if r.get("overridden_by"):
                by_source[r["overridden_by"]] = by_source.get(r["overridden_by"], 0) + 1
        records = [r for r in records if not r.get("overridden_by")]
        if dropped:
            detail = ", ".join(f"{name} {count}" for name, count in sorted(by_source.items()))
            print(f"Dropped {dropped} record(s) whose verdict bypassed fusion: {detail}.")
            if "face_pathway" in by_source:
                print(
                    "  Face pathway records are dropped because VT_FACE_DECIDES lets that signal "
                    "set the score directly. With faces in the set that can remove a lot of them, "
                    "so check the count above against the 25 per class floor."
                )

    n_ai = sum(r["label"] for r in records)
    n_real = len(records) - n_ai
    if min(n_ai, n_real) < MIN_PER_CLASS:
        print(
            f"Only {n_real} real and {n_ai} AI usable records. Need at least {MIN_PER_CLASS} per "
            f"class for a fit worth trusting. Collect more before calibrating."
        )
        return 2

    names = sorted({name for r in records for name in r["signals"]})
    print(f"Signals: {names}")

    rng = random.Random(args.seed)
    shuffled = records[:]
    rng.shuffle(shuffled)
    split = int(len(shuffled) * (1 - args.test_size))
    train, test = shuffled[:split], shuffled[split:]
    print(f"Train {len(train)}, holdout {len(test)}")

    try:
        import numpy as np
        from sklearn.linear_model import LogisticRegression
    except ImportError:
        print("scikit-learn and numpy are required for calibration.")
        return 2

    x_train, y_train = build_matrix(train, names)
    x_test, y_test = build_matrix(test, names)

    model = LogisticRegression(max_iter=2000, C=1.0)
    model.fit(np.array(x_train), np.array(y_train))
    coefficients = model.coef_[0].tolist()
    bias = float(model.intercept_[0])

    weights = {name: float(c) for name, c in zip(names, coefficients)}

    negative = [n for n, w in weights.items() if w <= 0]
    if negative:
        print(
            f"Note: {negative} received a non positive coefficient and will be applied as fitted, "
            f"which means the signal pushes toward 'real' as it rises. That usually means it is "
            f"uninformative or anti correlated on this set. Worth investigating rather than "
            f"shipping, since an inverted label mapping looks exactly like this."
        )

    test_scores = [float(model.predict_proba(np.array([row]))[0][1]) for row in x_test]
    auc = roc_auc(y_test, test_scores)
    metrics = {
        "holdout_auc": auc,
        "holdout_at_half": confusion(y_test, test_scores, 0.5),
        "holdout_n": len(y_test),
        "brier": sum((s - y) ** 2 for s, y in zip(test_scores, y_test)) / max(len(y_test), 1),
    }

    payload = {
        "bias": bias,
        "weights": weights,
        "signal_clamp": settings.fusion.signal_clamp,
        "fitted_on": f"{args.records.name}, {len(train)} train records",
        "holdout_metrics": metrics,
        "signal_names": names,
    }
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\nFitted calibration")
    for name, weight in weights.items():
        print(f"  {name}: {weight:+.4f}")
    print(f"  bias: {bias:+.4f}")
    print(f"\nHoldout AUC: {auc:.4f}" if auc is not None else "\nHoldout AUC: not computable")
    print(f"Holdout Brier score: {metrics['brier']:.4f} (lower is better, 0.25 is chance)")
    print(f"\nSaved to {args.out}. Restart the API to pick it up.")
    print("Reminder: these numbers describe your evaluation set, not images in general.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
