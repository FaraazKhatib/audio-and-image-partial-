"""Compute real metrics from collected records and write a report.

Reports ROC AUC, which is threshold free, alongside the operating point your configured
thresholds actually produce. It also reports the abstention rate, because a detector that sends
40 percent of inputs to uncertain is not comparable to one that always commits, and per source
AUC, because aggregate numbers hide collapse on unseen generators.

    python -m eval.run_eval --records eval_results/records.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from veritrust.config import VERDICT_AI, VERDICT_AUTHENTIC, VERDICT_UNCERTAIN, settings


def load_records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def roc_auc(labels: list[int], scores: list[float]) -> float | None:
    """Rank based AUC with tie correction. Returns None when a class is missing."""
    pairs = sorted(zip(scores, labels))
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None

    ranks = [0.0] * len(pairs)
    index = 0
    while index < len(pairs):
        end = index
        while end + 1 < len(pairs) and pairs[end + 1][0] == pairs[index][0]:
            end += 1
        average = (index + end) / 2.0 + 1.0
        for position in range(index, end + 1):
            ranks[position] = average
        index = end + 1

    rank_sum = sum(rank for rank, (_, label) in zip(ranks, pairs) if label == 1)
    return (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def confusion(labels: list[int], scores: list[float], threshold: float) -> dict:
    tp = sum(1 for l, s in zip(labels, scores) if l == 1 and s >= threshold)
    fp = sum(1 for l, s in zip(labels, scores) if l == 0 and s >= threshold)
    tn = sum(1 for l, s in zip(labels, scores) if l == 0 and s < threshold)
    fn = sum(1 for l, s in zip(labels, scores) if l == 1 and s < threshold)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "threshold": threshold,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": (tp + tn) / max(len(labels), 1),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_rate": fp / max(fp + tn, 1),
    }


def band_breakdown(records: list[dict]) -> dict:
    counts: dict[str, dict[str, int]] = {
        VERDICT_AUTHENTIC: {"real": 0, "ai": 0},
        VERDICT_UNCERTAIN: {"real": 0, "ai": 0},
        VERDICT_AI: {"real": 0, "ai": 0},
    }
    for record in records:
        bucket = counts.setdefault(record["verdict"], {"real": 0, "ai": 0})
        bucket["ai" if record["label"] == 1 else "real"] += 1
    total = max(len(records), 1)
    abstained = sum(counts[VERDICT_UNCERTAIN].values())
    decided = [r for r in records if r["verdict"] != VERDICT_UNCERTAIN]
    correct = sum(
        1
        for r in decided
        if (r["verdict"] == VERDICT_AI) == (r["label"] == 1)
    )
    return {
        "counts": counts,
        "abstention_rate": abstained / total,
        "accuracy_when_committed": correct / max(len(decided), 1),
        "committed": len(decided),
    }


def render(report: dict) -> str:
    lines = ["# VeriTrust evaluation", ""]
    lines.append(f"Records: {report['n']} ({report['n_real']} real, {report['n_ai']} AI)")
    lines.append(f"Calibrated at collection time: {report['calibrated']}")
    lines.append("")

    auc = report["fused"]["auc"]
    lines.append("## Fused score")
    lines.append("")
    lines.append(f"ROC AUC: {auc:.4f}" if auc is not None else "ROC AUC: not computable")
    for key in ("at_ai_min", "at_half"):
        cm = report["fused"][key]
        lines.append(
            f"At threshold {cm['threshold']:.2f}: accuracy {cm['accuracy']:.4f}, "
            f"precision {cm['precision']:.4f}, recall {cm['recall']:.4f}, "
            f"FPR {cm['false_positive_rate']:.4f}"
        )
    bands = report["bands"]
    lines.append(
        f"Abstention rate: {bands['abstention_rate']:.4f}. "
        f"Accuracy on the {bands['committed']} committed cases: "
        f"{bands['accuracy_when_committed']:.4f}"
    )
    lines.append("")

    lines.append("## Individual signals")
    lines.append("")
    lines.append("| signal | n | AUC |")
    lines.append("| --- | --- | --- |")
    for name, stats in sorted(report["signals"].items(), key=lambda kv: -(kv[1]["auc"] or 0)):
        value = f"{stats['auc']:.4f}" if stats["auc"] is not None else "n/a"
        lines.append(f"| {name} | {stats['n']} | {value} |")
    lines.append("")

    lines.append("## Per source")
    lines.append("")
    lines.append("| source | label | n | AUC vs all real | mean score |")
    lines.append("| --- | --- | --- | --- | --- |")
    for source, stats in sorted(report["per_source"].items()):
        value = f"{stats['auc']:.4f}" if stats["auc"] is not None else "n/a"
        lines.append(
            f"| {source} | {stats['label']} | {stats['n']} | {value} | {stats['mean_score']:.4f} |"
        )
    lines.append("")
    lines.append("Generated by eval/run_eval.py. Every number above comes from the records file.")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, default=Path("eval_results/records.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("eval_results"))
    args = parser.parse_args()

    if not args.records.is_file():
        print(f"No records at {args.records}. Run eval.collect first.")
        return 2

    records = load_records(args.records)
    if not records:
        print("Records file is empty.")
        return 2

    labels = [r["label"] for r in records]
    scores = [r["score_ai"] for r in records]

    signal_stats: dict[str, dict] = {}
    by_signal: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for record in records:
        for name, value in record["signals"].items():
            by_signal[name].append((record["label"], value))
    for name, rows in by_signal.items():
        signal_labels = [l for l, _ in rows]
        signal_scores = [s for _, s in rows]
        signal_stats[name] = {"n": len(rows), "auc": roc_auc(signal_labels, signal_scores)}

    real_scores = [r["score_ai"] for r in records if r["label"] == 0]
    per_source: dict[str, dict] = {}
    for source in sorted({r["source"] for r in records}):
        subset = [r for r in records if r["source"] == source]
        label = 1 if subset[0]["label"] == 1 else 0
        if label == 1:
            combo_labels = [1] * len(subset) + [0] * len(real_scores)
            combo_scores = [r["score_ai"] for r in subset] + real_scores
            auc = roc_auc(combo_labels, combo_scores)
        else:
            auc = None
        per_source[source] = {
            "label": "ai" if label == 1 else "real",
            "n": len(subset),
            "auc": auc,
            "mean_score": sum(r["score_ai"] for r in subset) / len(subset),
        }

    report = {
        "n": len(records),
        "n_real": labels.count(0),
        "n_ai": labels.count(1),
        "calibrated": None,
        "fused": {
            "auc": roc_auc(labels, scores),
            "at_ai_min": confusion(labels, scores, settings.thresholds.ai_min),
            "at_half": confusion(labels, scores, 0.5),
        },
        "bands": band_breakdown(records),
        "signals": signal_stats,
        "per_source": per_source,
    }

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    markdown = render(report)
    (args.out / "report.md").write_text(markdown, encoding="utf-8")
    print(markdown)
    print(f"\nWritten to {args.out / 'report.md'} and {args.out / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
