"""Run multiple trackers on the same clips and compare MOT metrics.

Usage:
    .\\.venv\\Scripts\\python.exe scripts\\compare_trackers.py
    .\\.venv\\Scripts\\python.exe scripts\\compare_trackers.py --clips <id1> <id2>
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import subprocess
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from road_safety.pipeline import run_pipeline  # noqa: E402

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "data.yaml"
DEFAULT_OUT = PROJECT_ROOT / "outputs" / "reports" / "tracker_comparison.csv"
DEFAULT_JSON = PROJECT_ROOT / "outputs" / "reports" / "tracker_comparison.json"


def slug(tracker: str) -> str:
    return Path(tracker).stem.replace(" ", "_")


def load_clips(config_path: Path) -> list[dict]:
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    return config.get("clips", [])


def run_tracker(
    tracker: str,
    clips: list[dict],
    image_size: int,
    output_root: Path,
    reuse: bool = False,
) -> dict:
    name = slug(tracker)
    report_dir = output_root / name / "reports"
    video_dir = output_root / name / "videos"
    report_dir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)

    for clip in clips:
        video = PROJECT_ROOT / clip["video"]
        mot_path = report_dir / f"{clip['clip_id']}_mot.txt"
        if reuse and mot_path.is_file() and mot_path.stat().st_size > 0:
            print(f"[{name}] {clip['clip_id'][:8]} ... reuse", flush=True)
            continue
        print(f"[{name}] {clip['clip_id'][:8]} ...", flush=True)
        run_pipeline(
            config_path=PROJECT_ROOT / "configs" / "tracking.yaml",
            root=PROJECT_ROOT,
            overrides={
                "input_video": video,
                "tracker": tracker,
                "image_size": image_size,
                "output_dir": video_dir,
                "report_dir": report_dir,
                "show_preview": False,
                "save_output": True,
            },
            show=False,
        )

    eval_csv = output_root / name / "evaluation.csv"
    eval_json = output_root / name / "evaluation.json"
    clip_ids = [clip["clip_id"] for clip in clips]
    subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "evaluate_tracking.py"),
            "--config",
            str(DEFAULT_CONFIG),
            "--pred-dir",
            str(report_dir),
            "--out",
            str(eval_csv),
            "--json",
            str(eval_json),
            "--clips",
            *clip_ids,
        ],
        check=True,
        cwd=PROJECT_ROOT,
    )
    return {"tracker": name, "csv": eval_csv, "json": eval_json}


def read_macro(csv_path: Path) -> dict:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("scope") == "macro" and row.get("class") == "all":
                return row
    return {}


def floor_number(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare trackers on MOT metrics.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--trackers",
        nargs="+",
        default=["bytetrack.yaml", "botsort.yaml"],
    )
    parser.add_argument("--clips", nargs="*", default=None)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument(
        "--reuse",
        action="store_true",
        help="Reuse existing predictions in the tracker report folders.",
    )
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs" / "trackers")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.WARNING)
    clips = load_clips(args.config)
    if args.clips:
        wanted = set(args.clips)
        clips = [clip for clip in clips if clip["clip_id"] in wanted]
    if not clips:
        print("No clips to compare.", file=sys.stderr)
        return 2

    results = []
    rows = []
    for tracker in args.trackers:
        result = run_tracker(
            tracker=tracker,
            clips=clips,
            image_size=args.imgsz,
            output_root=args.output_root,
            reuse=args.reuse,
        )
        macro = read_macro(result["csv"])
        row = {
            "tracker": result["tracker"],
            "clips": len(clips),
            "gt": macro.get("gt", ""),
            "pred": macro.get("pred", ""),
            "tp": macro.get("tp", ""),
            "fp": macro.get("fp", ""),
            "fn": macro.get("fn", ""),
            "idsw": macro.get("idsw", ""),
            "mota": macro.get("mota", ""),
            "motp": macro.get("motp", ""),
            "idf1": macro.get("idf1", ""),
            "precision": macro.get("precision", ""),
            "recall": macro.get("recall", ""),
        }
        rows.append(row)
        results.append(
            {
                "tracker": result["tracker"],
                "csv": str(result["csv"]),
                "json": str(result["json"]),
                "macro": row,
            }
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    with args.json.open("w", encoding="utf-8") as handle:
        json.dump({"clips": [clip["clip_id"] for clip in clips], "results": results}, handle, indent=2)
        handle.write("\n")

    print(f"\n{'tracker':12} {'MOTA':>8} {'IDF1':>8} {'IDSW':>6} {'TP':>6} {'FP':>6} {'FN':>6}")
    for row in rows:
        print(
            f"{row['tracker']:12} {str(row['mota']):>8} {str(row['idf1']):>8} "
            f"{str(row['idsw']):>6} {str(row['tp']):>6} {str(row['fp']):>6} {str(row['fn']):>6}"
        )
    print(f"\ncsv: {args.out}")
    print(f"json: {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
