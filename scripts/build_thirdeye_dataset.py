"""Build project annotations from the extracted ThirdEye clip folders.

For every clip it writes:
  data/annotations/mot/<clip>.txt          MOT 1.1 boxes + track IDs
  data/annotations/cvat/<clip>/annotations.xml   CVAT video tracks
  data/annotations/attributes.csv          per-detection class/confidence rows
  data/processed/videos/<clip>.mp4         clip video at 1 fps
  configs/data.yaml                        clips + train/val/test split

Mapping:
  person       -> pedestrian
  motorcycle   -> two_wheeler
  autorickshaw -> autorickshaw
  car          -> car
  bus, truck   -> bus_truck

`rider` boxes are intentionally ignored here: they are rider-only boxes, while
the project tracks the two-wheeler vehicle. Helmet attributes are added later
from AI City data.

Usage:
    .\\.venv\\Scripts\\python.exe scripts\\build_thirdeye_dataset.py
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from xml.dom import minidom

import cv2
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CLIPS_DIR = PROJECT_ROOT / "data" / "processed" / "clips"
DEFAULT_ANNOTATIONS_DIR = PROJECT_ROOT / "data" / "annotations"
DEFAULT_VIDEOS_DIR = PROJECT_ROOT / "data" / "processed" / "videos"
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "data.yaml"

CLASS_NAMES = ["pedestrian", "two_wheeler", "car", "autorickshaw", "bus_truck"]
CATEGORY_TO_CLASS = {
    "person": "pedestrian",
    "motorcycle": "two_wheeler",
    "autorickshaw": "autorickshaw",
    "car": "car",
    "bus": "bus_truck",
    "truck": "bus_truck",
}


@dataclass
class Box:
    frame: int  # 0-based
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_name: str
    source_category: str


@dataclass
class ClipAnnotation:
    clip_id: str
    boxes: list[Box] = field(default_factory=list)
    track_id_map: dict[tuple[str, int], int] = field(default_factory=dict)

    def add_box(self, key: tuple[str, int], box: Box) -> None:
        if key not in self.track_id_map:
            self.track_id_map[key] = len(self.track_id_map) + 1
        box_dict = box.__dict__
        self.boxes.append((self.track_id_map[key], box_dict))

    def sorted_boxes(self) -> list[tuple[int, dict]]:
        return sorted(self.boxes, key=lambda item: (item[1]["frame"], item[0]))


def load_clip(clip_dir: Path) -> ClipAnnotation:
    clip_id = clip_dir.name
    annotation = ClipAnnotation(clip_id=clip_id)
    files = sorted(clip_dir.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No frame JSON files in {clip_dir}")

    for path in files:
        try:
            frame_index = int(path.stem.rsplit("_", 1)[-1])
        except ValueError:
            continue
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        for label in payload.get("labels", []):
            category = str(label.get("category", ""))
            class_name = CATEGORY_TO_CLASS.get(category)
            if class_name is None:
                continue
            track_id = label.get("track_id")
            if track_id is None:
                continue
            box2d = label.get("box2d") or {}
            try:
                x1 = float(box2d["x1"])
                y1 = float(box2d["y1"])
                x2 = float(box2d["x2"])
                y2 = float(box2d["y2"])
            except (KeyError, TypeError, ValueError):
                continue
            if x2 <= x1 or y2 <= y1:
                continue
            annotation.add_box(
                (category, int(track_id)),
                Box(
                    frame=frame_index,
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    confidence=float(label.get("confidence") or 0.0),
                    class_name=class_name,
                    source_category=category,
                ),
            )

    return annotation


def filter_short_tracks(annotation: ClipAnnotation, min_length: int) -> None:
    """Drop noisy one-frame tracks and renumber the remaining IDs."""
    if min_length <= 1:
        return

    counts: Counter = Counter()
    for track_id, _box in annotation.boxes:
        counts[track_id] += 1
    keep = {track_id for track_id, count in counts.items() if count >= min_length}

    new_map: dict[int, int] = {}
    kept_boxes: list[tuple[int, dict]] = []
    for track_id, box in annotation.boxes:
        if track_id not in keep:
            continue
        if track_id not in new_map:
            new_map[track_id] = len(new_map) + 1
        kept_boxes.append((new_map[track_id], box))

    annotation.boxes = kept_boxes
    annotation.track_id_map = {
        key: new_map[track_id]
        for key, track_id in annotation.track_id_map.items()
        if track_id in new_map
    }


def write_mot(annotation: ClipAnnotation, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for track_id, box in annotation.sorted_boxes():
            width = box["x2"] - box["x1"]
            height = box["y2"] - box["y1"]
            handle.write(
                f"{box['frame'] + 1},{track_id},"
                f"{box['x1']:.2f},{box['y1']:.2f},{width:.2f},{height:.2f},"
                f"{box['confidence']:.4f},-1,-1,-1\n"
            )


def write_attributes(annotation: ClipAnnotation, writer: csv.DictWriter) -> int:
    rows = 0
    for track_id, box in annotation.sorted_boxes():
        writer.writerow(
            {
                "clip_id": annotation.clip_id,
                "frame": box["frame"] + 1,
                "track_id": track_id,
                "class_name": box["class_name"],
                "source_category": box["source_category"],
                "confidence": f"{box['confidence']:.4f}",
                "x1": f"{box['x1']:.2f}",
                "y1": f"{box['y1']:.2f}",
                "x2": f"{box['x2']:.2f}",
                "y2": f"{box['y2']:.2f}",
                "helmet_status": "unknown" if box["class_name"] == "two_wheeler" else "",
                "pillion_helmet": "",
                "rider_count": "",
            }
        )
        rows += 1
    return rows


def write_cvat(annotation: ClipAnnotation, path: Path, width: int, height: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    root = ET.Element("annotations")
    ET.SubElement(root, "version").text = "1.1"

    meta = ET.SubElement(root, "meta")
    task = ET.SubElement(meta, "task")
    ET.SubElement(task, "id").text = "0"
    ET.SubElement(task, "name").text = annotation.clip_id
    ET.SubElement(task, "size").text = str(
        max((box["frame"] for _, box in annotation.boxes), default=0) + 1
    )
    ET.SubElement(task, "mode").text = "interpolation"
    ET.SubElement(task, "overlap").text = "0"
    ET.SubElement(task, "bugtracker").text = ""
    ET.SubElement(task, "created").text = "2026-11-13 00:00:00.000000"
    ET.SubElement(task, "updated").text = "2026-11-13 00:00:00.000000"
    ET.SubElement(task, "start_frame").text = "0"
    ET.SubElement(task, "stop_frame").text = str(
        max((box["frame"] for _, box in annotation.boxes), default=0)
    )
    ET.SubElement(task, "frame_filter").text = ""

    original_size = ET.SubElement(task, "original_size")
    ET.SubElement(original_size, "width").text = str(width)
    ET.SubElement(original_size, "height").text = str(height)

    labels = ET.SubElement(task, "labels")
    for class_name in CLASS_NAMES:
        label = ET.SubElement(labels, "label")
        ET.SubElement(label, "name").text = class_name
        ET.SubElement(label, "color").text = {
            "pedestrian": "#ff0000",
            "two_wheeler": "#00ff00",
            "car": "#0000ff",
            "autorickshaw": "#ffff00",
            "bus_truck": "#ff00ff",
        }[class_name]
        ET.SubElement(label, "type").text = "rectangle"
        ET.SubElement(label, "attributes").text = ""

    # Group boxes by track id; a track keeps one label.
    tracks: dict[int, list[dict]] = defaultdict(list)
    track_label: dict[int, str] = {}
    for track_id, box in annotation.boxes:
        tracks[track_id].append(box)
        track_label.setdefault(track_id, box["class_name"])

    for track_id in sorted(tracks):
        track = ET.SubElement(
            root,
            "track",
            id=str(track_id),
            label=track_label[track_id],
            source="auto",
        )
        for box in sorted(tracks[track_id], key=lambda item: item["frame"]):
            ET.SubElement(
                track,
                "box",
                frame=str(box["frame"]),
                xtl=f"{box['x1']:.2f}",
                ytl=f"{box['y1']:.2f}",
                xbr=f"{box['x2']:.2f}",
                ybr=f"{box['y2']:.2f}",
                outside="0",
                occluded="0",
                z_order="0",
            )

    xml_bytes = ET.tostring(root, encoding="utf-8")
    pretty = minidom.parseString(xml_bytes).toprettyxml(indent="  ", encoding="utf-8")
    path.write_bytes(pretty)


def encode_video(clip_dir: Path, output_path: Path) -> tuple[int, int, int]:
    images = sorted(clip_dir.glob("*.jpg"))
    if not images:
        raise FileNotFoundError(f"No JPG frames in {clip_dir}")

    first = cv2.imread(str(images[0]))
    if first is None:
        raise RuntimeError(f"Unable to read {images[0]}")
    height, width = first.shape[:2]
    output_path.parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        1.0,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Unable to create {output_path}")

    try:
        for image_path in images:
            frame = cv2.imread(str(image_path))
            if frame is None:
                continue
            if frame.shape[1] != width or frame.shape[0] != height:
                frame = cv2.resize(frame, (width, height))
            writer.write(frame)
    finally:
        writer.release()

    return len(images), width, height


def build_splits(clip_ids: list[str]) -> dict[str, list[str]]:
    ordered = sorted(clip_ids)
    if len(ordered) < 3:
        return {"train": ordered, "val": [], "test": []}
    if len(ordered) == 3:
        return {"train": [ordered[0]], "val": [ordered[1]], "test": [ordered[2]]}
    test_count = 1
    val_count = 1
    train_count = len(ordered) - test_count - val_count
    return {
        "train": ordered[:train_count],
        "val": ordered[train_count:train_count + val_count],
        "test": ordered[train_count + val_count:],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build MOT/CVAT annotations and clip videos from ThirdEye clips."
    )
    parser.add_argument("--clips-dir", type=Path, default=DEFAULT_CLIPS_DIR)
    parser.add_argument("--annotations-dir", type=Path, default=DEFAULT_ANNOTATIONS_DIR)
    parser.add_argument("--videos-dir", type=Path, default=DEFAULT_VIDEOS_DIR)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument(
        "--min-track-length",
        type=int,
        default=2,
        help="Drop tracks shorter than this many frames. Default: 2.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.clips_dir = args.clips_dir.resolve()
    args.annotations_dir = args.annotations_dir.resolve()
    args.videos_dir = args.videos_dir.resolve()
    args.config = args.config.resolve()
    if not args.clips_dir.is_dir():
        print(f"Clips directory not found: {args.clips_dir}", file=sys.stderr)
        return 2

    clip_dirs = sorted(path for path in args.clips_dir.iterdir() if path.is_dir())
    if not clip_dirs:
        print(f"No clip folders in {args.clips_dir}", file=sys.stderr)
        return 2

    mot_dir = args.annotations_dir / "mot"
    cvat_dir = args.annotations_dir / "cvat"
    attributes_path = args.annotations_dir / "attributes.csv"
    summary_path = args.annotations_dir / "thirdeye_summary.json"
    mot_dir.mkdir(parents=True, exist_ok=True)
    cvat_dir.mkdir(parents=True, exist_ok=True)
    attributes_path.parent.mkdir(parents=True, exist_ok=True)

    clip_summaries = []
    attributes_handle = attributes_path.open("w", encoding="utf-8", newline="")
    attributes_writer = csv.DictWriter(
        attributes_handle,
        fieldnames=[
            "clip_id",
            "frame",
            "track_id",
            "class_name",
            "source_category",
            "confidence",
            "x1",
            "y1",
            "x2",
            "y2",
            "helmet_status",
            "pillion_helmet",
            "rider_count",
        ],
    )
    attributes_writer.writeheader()

    try:
        for clip_dir in clip_dirs:
            print(f"building {clip_dir.name} ...", file=sys.stderr)
            annotation = load_clip(clip_dir)
            filter_short_tracks(annotation, args.min_track_length)
            if not annotation.boxes:
                print(f"  no mapped boxes, skipping", file=sys.stderr)
                continue

            mot_path = mot_dir / f"{annotation.clip_id}.txt"
            cvat_path = cvat_dir / annotation.clip_id / "annotations.xml"
            write_mot(annotation, mot_path)
            rows = write_attributes(annotation, attributes_writer)

            if args.no_video:
                images = sorted(clip_dir.glob("*.jpg"))
                frames = len(images)
                first = cv2.imread(str(images[0])) if images else None
                height, width = first.shape[:2] if first is not None else (1080, 1920)
            else:
                video_path = args.videos_dir / f"{annotation.clip_id}.mp4"
                frames, width, height = encode_video(clip_dir, video_path)
            write_cvat(annotation, cvat_path, width, height)

            class_counts = Counter(box["class_name"] for _, box in annotation.boxes)
            clip_summaries.append(
                {
                    "clip_id": annotation.clip_id,
                    "frames": frames,
                    "detections": rows,
                    "tracks": len(annotation.track_id_map),
                    "classes": dict(sorted(class_counts.items())),
                    "width": width,
                    "height": height,
                    "mot": str(mot_path.relative_to(PROJECT_ROOT)),
                    "cvat": str(cvat_path.relative_to(PROJECT_ROOT)),
                    "video": None
                    if args.no_video
                    else str(
                        (args.videos_dir / f"{annotation.clip_id}.mp4").relative_to(
                            PROJECT_ROOT
                        )
                    ),
                }
            )
    finally:
        attributes_handle.close()

    clip_ids = [item["clip_id"] for item in clip_summaries]
    splits = build_splits(clip_ids)
    config = {
        "name": "road-safety-thirdeye-7clip",
        "root": ".",
        "frame_rate": 1,
        "classes": {index: name for index, name in enumerate(CLASS_NAMES)},
        "splits": splits,
        "clips": [
            {
                "clip_id": item["clip_id"],
                "video": item["video"],
                "frames": item["frames"],
                "mot": item["mot"],
                "cvat": item["cvat"],
                "split": next(
                    (split for split, ids in splits.items() if item["clip_id"] in ids),
                    "train",
                ),
            }
            for item in sorted(clip_summaries, key=lambda row: row["clip_id"])
        ],
    }
    args.config.parent.mkdir(parents=True, exist_ok=True)
    with args.config.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)

    totals = {
        "clips": len(clip_summaries),
        "frames": sum(item["frames"] for item in clip_summaries),
        "detections": sum(item["detections"] for item in clip_summaries),
        "tracks": sum(item["tracks"] for item in clip_summaries),
        "classes": dict(
            sorted(
                sum(
                    (Counter(item["classes"]) for item in clip_summaries),
                    Counter(),
                ).items()
            )
        ),
    }
    summary = {"splits": splits, "totals": totals, "clips": clip_summaries}
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")

    print(f"clips: {totals['clips']}")
    print(f"frames: {totals['frames']}")
    print(f"detections: {totals['detections']}")
    print(f"tracks: {totals['tracks']}")
    print(f"classes: {totals['classes']}")
    print(f"config: {args.config}")
    print(f"summary: {summary_path}")
    print(f"attributes: {attributes_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
