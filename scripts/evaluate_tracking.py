"""Evaluate predicted MOT files against ThirdEye ground truth.

By default reads:
  data/annotations/mot/<clip>.txt          ground truth
  outputs/reports/<clip>_mot.txt           predictions
  outputs/reports/<clip>_tracks.csv        predicted class per track
  data/annotations/attributes.csv          GT class per track
  configs/data.yaml                        clip list and splits

Usage:
    .\\.venv\\Scripts\\python.exe scripts\\evaluate_tracking.py
    .\\.venv\\Scripts\\python.exe scripts\\evaluate_tracking.py --pred-dir outputs/trackers/botsort/reports
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from road_safety.metrics import (  # noqa: E402
    aggregate_sequences,
    evaluate,
    filter_by_class,
    load_mot,
)

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "data.yaml"
DEFAULT_GT = PROJECT_ROOT / "data" / "annotations" / "mot"
DEFAULT_ATTRIBUTES = PROJECT_ROOT / "data" / "annotations" / "attributes.csv"
DEFAULT_PRED = PROJECT_ROOT / "outputs" / "reports"
DEFAULT_OUT = PROJECT_ROOT / "outputs" / "reports" / "evaluation.csv"
DEFAULT_JSON = PROJECT_ROOT / "outputs" / "reports" / "evaluation.json"
DEFAULT_CLASSES = ["pedestrian", "two_wheeler", "car", "autorickshaw", "bus_truck"]


def load_config(path: Path) -> dict:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def majority(values: Counter) -> str:
    return values.most_common(1)[0][0] if values else "unknown"


def load_gt_classes(path: Path) -> dict[str, dict[int, str]]:
    classes: dict[str, Counter] = {}
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            key = row.get("clip_id", "")
            track_id = row.get("track_id", "")
            class_name = row.get("class_name", "")
            if not key or not track_id:
                continue
            classes.setdefault(key, Counter())[(int(track_id), class_name)] += 1

    # Re-key from (track_id, class_name) counters to a single class per track.
    result: dict[str, dict[int, str]] = {}
    for clip_id, counter in classes.items():
        per_track: dict[int, Counter] = {}
        for (track_id, class_name), count in counter.items():
            per_track.setdefault(track_id, Counter())[class_name] += count
        result[clip_id] = {
            track_id: majority(values)
            for track_id, values in per_track.items()
        }
    return result


def load_pred_classes(path: Path) -> dict[int, str]:
    if not path.is_file():
        return {}
    per_track: dict[int, Counter] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                track_id = int(row["track_id"])
            except (KeyError, ValueError):
                continue
            class_name = row.get("class_name", "unknown")
            per_track.setdefault(track_id, Counter())[class_name] += 1
    return {track_id: majority(values) for track_id, values in per_track.items()}


def safe_round(value: float | None, digits: int = 4) -> float | None:
    return round(float(value), digits) if value is not None else None


def row_from_metrics(
    scope: str,
    clip_id: str,
    split: str,
    class_name: str,
    metrics: dict,
) -> dict:
    return {
        "scope": scope,
        "clip_id": clip_id,
        "split": split,
        "class": class_name,
        "gt": metrics["gt"],
        "pred": metrics["pred"],
        "tp": metrics["tp"],
        "fp": metrics["fp"],
        "fn": metrics["fn"],
        "idsw": metrics["idsw"],
        "mota": safe_round(metrics["mota"]),
        "motp": safe_round(metrics["motp"]),
        "idf1": safe_round(metrics["idf1"]),
        "precision": safe_round(metrics["precision"]),
        "recall": safe_round(metrics["recall"]),
    }


def _summary_row(class_name: str, aggregate: dict) -> dict:
    return {
        "scope": "macro",
        "clip_id": "",
        "split": "",
        "class": class_name,
        "gt": aggregate["gt"],
        "pred": aggregate["pred"],
        "tp": aggregate["tp"],
        "fp": aggregate["fp"],
        "fn": aggregate["fn"],
        "idsw": aggregate["idsw"],
        "mota": safe_round(aggregate.get("macro_mota")),
        "motp": safe_round(aggregate.get("macro_motp")),
        "idf1": safe_round(aggregate.get("macro_idf1")),
        "precision": safe_round(aggregate.get("macro_precision")),
        "recall": safe_round(aggregate.get("macro_recall")),
        "macro_mota": safe_round(aggregate.get("macro_mota")),
        "macro_motp": safe_round(aggregate.get("macro_motp")),
        "macro_idf1": safe_round(aggregate.get("macro_idf1")),
        "macro_precision": safe_round(aggregate.get("macro_precision")),
        "macro_recall": safe_round(aggregate.get("macro_recall")),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate MOT predictions.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--gt-dir", type=Path, default=DEFAULT_GT)
    parser.add_argument("--attributes", type=Path, default=DEFAULT_ATTRIBUTES)
    parser.add_argument("--pred-dir", type=Path, default=DEFAULT_PRED)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--clips", nargs="*", default=None)
    parser.add_argument("--classes", nargs="*", default=None)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    config_clips = {
        item["clip_id"]: item.get("split", "")
        for item in config.get("clips", [])
        if "clip_id" in item
    }
    classes = args.classes or [
        name for _index, name in sorted((config.get("classes") or {}).items())
    ]
    if not classes:
        classes = DEFAULT_CLASSES

    gt_classes = load_gt_classes(args.attributes)
    clip_ids = args.clips or sorted(config_clips) or sorted(
        path.stem for path in args.gt_dir.glob("*.txt")
    )
    if not clip_ids:
        print("No clips to evaluate.", file=sys.stderr)
        return 2

    rows: list[dict] = []
    missing: list[str] = []
    for clip_id in clip_ids:
        gt_path = args.gt_dir / f"{clip_id}.txt"
        pred_path = args.pred_dir / f"{clip_id}_mot.txt"
        tracks_path = args.pred_dir / f"{clip_id}_tracks.csv"
        if not gt_path.is_file() or not pred_path.is_file():
            missing.append(clip_id)
            continue

        ground_truth = load_mot(gt_path)
        predictions = load_mot(pred_path)
        gt_track_classes = gt_classes.get(clip_id, {})
        pred_track_classes = load_pred_classes(tracks_path)
        split = config_clips.get(clip_id, "")

        overall = evaluate(ground_truth, predictions, iou_threshold=args.iou)
        rows.append(row_from_metrics("clip", clip_id, split, "all", overall))

        for class_name in classes:
            gt_class = filter_by_class(ground_truth, gt_track_classes, class_name)
            pred_class = filter_by_class(predictions, pred_track_classes, class_name)
            class_metrics = evaluate(gt_class, pred_class, iou_threshold=args.iou)
            if class_metrics["gt"] == 0 and class_metrics["pred"] == 0:
                continue
            rows.append(
                row_from_metrics("clip", clip_id, split, class_name, class_metrics)
            )

    if not rows:
        print("No matching GT/prediction files.", file=sys.stderr)
        if missing:
            print(f"Missing: {', '.join(missing)}", file=sys.stderr)
        return 2

    all_rows = [row for row in rows if row["class"] == "all"]
    macro = aggregate_sequences(all_rows)
    summary_rows = [_summary_row("all", macro)]

    for class_name in classes:
        class_rows = [row for row in rows if row["class"] == class_name]
        if not class_rows:
            continue
        summary_rows.append(_summary_row(class_name, aggregate_sequences(class_rows)))

    fieldnames = [
        "scope",
        "clip_id",
        "split",
        "class",
        "gt",
        "pred",
        "tp",
        "fp",
        "fn",
        "idsw",
        "mota",
        "motp",
        "idf1",
        "precision",
        "recall",
        "macro_mota",
        "macro_motp",
        "macro_idf1",
        "macro_precision",
        "macro_recall",
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows + summary_rows)

    payload = {
        "pred_dir": str(args.pred_dir),
        "iou_threshold": args.iou,
        "clips": clip_ids,
        "missing": missing,
        "rows": rows,
        "summary": summary_rows,
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    with args.json.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")

    print(f"evaluated {len(all_rows)} clips; {len(rows)} rows")
    print(f"csv: {args.out}")
    print(f"json: {args.json}")
    header = f"{'clip':12} {'gt':>5} {'tp':>5} {'fp':>4} {'fn':>4} {'idsw':>4} {'MOTA':>7} {'IDF1':>7}"
    print(header)
    for row in all_rows:
        print(
            f"{row['clip_id'][:8]:12} {row['gt']:5d} {row['tp']:5d} "
            f"{row['fp']:4d} {row['fn']:4d} {row['idsw']:4d} "
            f"{_fmt(row['mota']):>7} {_fmt(row['idf1']):>7}"
        )
    macro_row = summary_rows[0]
    print(
        f"{'MACRO':12} {'':>5} {'':>5} {'':>4} {'':>4} {'':>4} "
        f"{_fmt(macro_row.get('macro_mota')):>7} {_fmt(macro_row.get('macro_idf1')):>7}"
    )
    if missing:
        print(f"missing predictions: {', '.join(missing)}", file=sys.stderr)
    return 0


def _fmt(value: float | None) -> str:
    return f"{value:.3f}" if value is not None else "-"


if __name__ == "__main__":
    raise SystemExit(main())
