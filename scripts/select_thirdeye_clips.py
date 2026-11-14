"""Rank ThirdEye clips from downloaded tar shards and extract selected clips.

Why this does not read annotations/detection.json:
the per-frame JSON files inside the tar shards already contain the labels for
the frames you actually downloaded. Scanning those is fast and avoids loading
the 1.2 GB global annotation file into memory.

Usage:
    .\\.venv\\Scripts\\python.exe scripts\\select_thirdeye_clips.py --select 7
    .\\.venv\\Scripts\\python.exe scripts\\select_thirdeye_clips.py --select 7 --extract
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
import tarfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "public" / "indian-road"
DEFAULT_REPORT = PROJECT_ROOT / "outputs" / "reports" / "thirdeye_clip_candidates.csv"
DEFAULT_JSON = PROJECT_ROOT / "outputs" / "reports" / "thirdeye_clip_candidates.json"
DEFAULT_CLIPS_DIR = PROJECT_ROOT / "data" / "processed" / "clips"

# ThirdEye category -> project group.
# "rider" is a motorcyclist/cyclist with a rider; use it as the moving
# two-wheeler signal. "motorcycle" is an unridden motorcycle.
GROUP_CATEGORIES = {
    "pedestrian": {"person"},
    "two_wheeler": {"rider"},
    "autorickshaw": {"autorickshaw"},
    "car": {"car"},
    "bus_truck": {"bus", "truck"},
}

CLASS_COLUMNS = [
    "person",
    "rider",
    "motorcycle",
    "autorickshaw",
    "car",
    "bus",
    "truck",
    "bicycle",
    "animal",
    "vehicle fallback",
    "traffic light",
    "traffic sign",
]


@dataclass
class ClipStats:
    clip_id: str
    frames: set[int] = field(default_factory=set)
    classes: Counter = field(default_factory=Counter)
    tracks: dict[str, set[int]] = field(default_factory=lambda: defaultdict(set))
    conf_sum: float = 0.0
    conf_count: int = 0

    def add_frame(self, frame: int, labels: list[dict]) -> None:
        if frame in self.frames:
            return
        self.frames.add(frame)
        for label in labels:
            category = str(label.get("category", "unknown"))
            self.classes[category] += 1
            track_id = label.get("track_id")
            if track_id is not None:
                self.tracks[category].add(int(track_id))
            confidence = label.get("confidence")
            if confidence is not None:
                self.conf_sum += float(confidence)
                self.conf_count += 1

    @property
    def max_frame(self) -> int:
        return max(self.frames) if self.frames else -1

    @property
    def min_frame(self) -> int:
        return min(self.frames) if self.frames else 0

    @property
    def coverage(self) -> float:
        if not self.frames:
            return 0.0
        return len(self.frames) / (self.max_frame + 1)

    @property
    def complete(self) -> bool:
        return self.min_frame == 0 and self.coverage >= 0.98


def parse_frame_from_name(name: str) -> int | None:
    stem = name.rsplit("/", 1)[-1]
    if "_" not in stem:
        return None
    frame_part = stem.rsplit("_", 1)[-1].split(".", 1)[0]
    try:
        return int(frame_part)
    except ValueError:
        return None


def scan_shards(data_dir: Path) -> dict[str, ClipStats]:
    shard_dir = data_dir / "data"
    shards = sorted(shard_dir.glob("*.tar"))
    if not shards:
        raise FileNotFoundError(f"No .tar shards found in {shard_dir}")

    clips: dict[str, ClipStats] = {}
    for shard in shards:
        print(f"scanning {shard.name} ...", file=sys.stderr)
        with tarfile.open(shard) as archive:
            for member in archive:
                if not member.name.endswith(".json"):
                    continue
                handle = archive.extractfile(member)
                if handle is None:
                    continue
                try:
                    payload = json.load(handle)
                except json.JSONDecodeError:
                    continue

                frame_name = payload.get("name", "")
                clip_id = frame_name.split("/", 1)[0]
                if not clip_id:
                    clip_id = member.name.rsplit("_", 1)[0]
                frame_index = parse_frame_from_name(member.name)
                if frame_index is None:
                    continue

                stats = clips.setdefault(clip_id, ClipStats(clip_id=clip_id))
                stats.add_frame(frame_index, payload.get("labels", []))

    return clips


def load_scene_attributes(data_dir: Path) -> dict[str, dict]:
    path = data_dir / "annotations" / "scene_attributes.json"
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def group_counts(stats: ClipStats) -> dict[str, int]:
    return {
        group: sum(stats.classes.get(category, 0) for category in categories)
        for group, categories in GROUP_CATEGORIES.items()
    }


def clip_row(stats: ClipStats, scene: dict) -> dict:
    groups = group_counts(stats)
    target_objects = sum(groups.values())
    frames = len(stats.frames)
    density = target_objects / frames if frames else 0.0

    coverage = sum(1 for count in groups.values() if count > 0)
    priority = (
        (15 if groups["autorickshaw"] > 0 else 0)
        + (10 if groups["pedestrian"] > 0 else 0)
        + (10 if groups["bus_truck"] > 0 else 0)
        + (5 if groups["two_wheeler"] > 0 else 0)
        + (5 if groups["car"] > 0 else 0)
    )
    score = coverage * 15 + priority + min(density * 8, 20) + min(frames / 20, 9)

    row = {
        "clip_id": stats.clip_id,
        "frames": frames,
        "duration_s": frames,  # ThirdEye keyframes are 1 fps
        "min_frame": stats.min_frame,
        "max_frame": stats.max_frame,
        "coverage": round(stats.coverage, 3),
        "complete": stats.complete,
        "scene": scene.get("scene", ""),
        "weather": scene.get("weather", ""),
        "timeofday": scene.get("timeofday", ""),
        "pedestrian": groups["pedestrian"],
        "two_wheeler": groups["two_wheeler"],
        "autorickshaw": groups["autorickshaw"],
        "car": groups["car"],
        "bus_truck": groups["bus_truck"],
        "target_objects": target_objects,
        "objects_per_frame": round(density, 2),
        "class_coverage": coverage,
        "score": round(score, 1),
        "selected": False,
        "selection_reason": "",
    }
    for category in CLASS_COLUMNS:
        row[category.replace(" ", "_")] = stats.classes.get(category, 0)
    for group, categories in GROUP_CATEGORIES.items():
        row[f"{group}_tracks"] = len(
            set().union(*(stats.tracks.get(category, set()) for category in categories))
        )
    row["mean_confidence"] = (
        round(stats.conf_sum / stats.conf_count, 3) if stats.conf_count else 0.0
    )
    return row


def select_clips(
    rows: list[dict],
    count: int,
    min_frames: int = 50,
    min_autorickshaw: int = 5,
    min_two_wheeler: int = 10,
) -> list[dict]:
    eligible = [
        row
        for row in rows
        if (
            row["complete"]
            and row["frames"] >= min_frames
            and row["autorickshaw"] >= min_autorickshaw
            and row["two_wheeler"] >= min_two_wheeler
            and row["target_objects"] > 0
        )
    ]
    if not eligible:
        return []

    selected: list[dict] = []
    used_scenes: set[str] = set()
    used_times: set[str] = set()
    used_weather: set[str] = set()
    covered_groups: set[str] = set()

    remaining = {row["clip_id"]: row for row in eligible}
    while remaining and len(selected) < count:
        best_id = None
        best_value = -math.inf

        for clip_id, row in remaining.items():
            bonus = 0.0
            if row["scene"] and row["scene"] not in used_scenes:
                bonus += 8
            elif row["scene"]:
                bonus -= 4
            if row["timeofday"] and row["timeofday"] not in used_times:
                bonus += 5
            elif row["timeofday"]:
                bonus -= 2
            if row["weather"] and row["weather"] not in used_weather:
                bonus += 3

            for group in GROUP_CATEGORIES:
                if row[group] > 0 and group not in covered_groups:
                    bonus += 10

            value = row["score"] + bonus
            if value > best_value:
                best_value = value
                best_id = clip_id

        if best_id is None:
            break

        row = remaining.pop(best_id)
        row["selected"] = True

        dynamic: list[str] = []
        if row["scene"] and row["scene"] not in used_scenes:
            dynamic.append(f"new scene: {row['scene']}")
        if row["timeofday"] and row["timeofday"] not in used_times:
            dynamic.append(f"new time: {row['timeofday']}")
        if row["weather"] and row["weather"] not in used_weather:
            dynamic.append(f"new weather: {row['weather']}")
        for group in GROUP_CATEGORIES:
            if row[group] > 0 and group not in covered_groups:
                dynamic.append(f"covers {group}")

        row["selection_reason"] = "; ".join(
            [
                f"score={row['score']}",
                f"coverage={row['class_coverage']}/5",
                f"density={row['objects_per_frame']}/frame",
                *dynamic,
            ]
        )

        selected.append(row)
        used_scenes.add(row["scene"])
        used_times.add(row["timeofday"])
        used_weather.add(row["weather"])
        covered_groups.update(
            group for group in GROUP_CATEGORIES if row[group] > 0
        )

    return selected


def extract_selected(
    data_dir: Path,
    selected: list[dict],
    clips_dir: Path,
    with_masks: bool,
) -> None:
    wanted = {row["clip_id"] for row in selected}
    if not wanted:
        print("Nothing selected; skipping extraction.", file=sys.stderr)
        return

    manifest_rows: list[dict] = []
    for clip in selected:
        (clips_dir / clip["clip_id"]).mkdir(parents=True, exist_ok=True)

    shards = sorted((data_dir / "data").glob("*.tar"))
    extracted = 0
    for shard in shards:
        with tarfile.open(shard) as archive:
            for member in archive:
                suffix = Path(member.name).suffix.lower()
                if suffix not in {".jpg", ".json"} and not (
                    with_masks and suffix == ".png"
                ):
                    continue
                clip_id = member.name.rsplit("_", 1)[0]
                if clip_id not in wanted:
                    continue
                destination = clips_dir / clip_id / member.name
                handle = archive.extractfile(member)
                if handle is None:
                    continue
                with destination.open("wb") as out:
                    shutil.copyfileobj(handle, out)
                extracted += 1
                if suffix == ".jpg":
                    frame = parse_frame_from_name(member.name)
                    manifest_rows.append(
                        {
                            "clip_id": clip_id,
                            "frame": frame,
                            "image": str(destination.relative_to(PROJECT_ROOT)),
                            "annotation": str(
                                (clips_dir / clip_id / member.name.replace(".jpg", ".json"))
                                .relative_to(PROJECT_ROOT)
                            ),
                        }
                    )

    manifest_path = clips_dir / "manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["clip_id", "frame", "image", "annotation"]
        )
        writer.writeheader()
        writer.writerows(sorted(manifest_rows, key=lambda r: (r["clip_id"], r["frame"])))

    print(f"extracted {extracted} files for {len(wanted)} clips")
    print(f"manifest: {manifest_path}")


def write_reports(
    rows: list[dict],
    selected: list[dict],
    csv_path: Path,
    json_path: Path,
) -> None:
    fieldnames = [
        "clip_id",
        "selected",
        "score",
        "frames",
        "duration_s",
        "complete",
        "coverage",
        "min_frame",
        "max_frame",
        "scene",
        "weather",
        "timeofday",
        "class_coverage",
        "pedestrian",
        "two_wheeler",
        "autorickshaw",
        "car",
        "bus_truck",
        "target_objects",
        "objects_per_frame",
        "mean_confidence",
        *[category.replace(" ", "_") for category in CLASS_COLUMNS],
        *[f"{group}_tracks" for group in GROUP_CATEGORIES],
        "selection_reason",
    ]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "selected_count": len(selected),
        "selected": selected,
        "candidates": rows,
    }
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def print_table(selected: list[dict]) -> None:
    headers = [
        "clip",
        "frames",
        "auto",
        "ped",
        "2W",
        "car",
        "bus/truck",
        "score",
        "scene",
        "time",
    ]
    print(" | ".join(headers))
    print("-" * 110)
    for row in selected:
        print(
            " | ".join(
                [
                    row["clip_id"][:8],
                    str(row["frames"]),
                    str(row["autorickshaw"]),
                    str(row["pedestrian"]),
                    str(row["two_wheeler"]),
                    str(row["car"]),
                    str(row["bus_truck"]),
                    str(row["score"]),
                    (row["scene"] or "?")[:18],
                    (row["timeofday"] or "?")[:12],
                ]
            )
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Rank ThirdEye clips from downloaded shards."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--select", type=int, default=7)
    parser.add_argument("--min-frames", type=int, default=50)
    parser.add_argument("--min-autorickshaw", type=int, default=5)
    parser.add_argument("--min-two-wheeler", type=int, default=10)
    parser.add_argument("--csv", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--extract", action="store_true")
    parser.add_argument("--with-masks", action="store_true")
    parser.add_argument("--clips-dir", type=Path, default=DEFAULT_CLIPS_DIR)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.data_dir.is_dir():
        print(f"Data directory not found: {args.data_dir}", file=sys.stderr)
        return 2

    clips = scan_shards(args.data_dir)
    scenes = load_scene_attributes(args.data_dir)
    rows = [clip_row(stats, scenes.get(clip_id, {})) for clip_id, stats in clips.items()]
    rows.sort(key=lambda row: row["score"], reverse=True)

    selected = select_clips(
        rows,
        args.select,
        min_frames=args.min_frames,
        min_autorickshaw=args.min_autorickshaw,
        min_two_wheeler=args.min_two_wheeler,
    )
    write_reports(rows, selected, args.csv, args.json)

    print(f"scanned {len(rows)} clips, {sum(len(c.frames) for c in clips.values())} frames")
    print(f"candidates: {args.csv}")
    print(f"selected {len(selected)} clips:\n")
    print_table(selected)

    if args.extract:
        extract_selected(
            data_dir=args.data_dir,
            selected=selected,
            clips_dir=args.clips_dir,
            with_masks=args.with_masks,
        )
        print(f"\nclips: {args.clips_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
