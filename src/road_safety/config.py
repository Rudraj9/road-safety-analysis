"""Configuration loading for the road-safety tracking pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml

# src/road_safety/config.py -> src/road_safety -> src -> project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULTS: dict[str, Any] = {
    "input_video": "data/raw/videos/traffic_sample_01.mp4",
    "model_weights": "models/pretrained/yolov8n.pt",
    "tracker": "bytetrack.yaml",
    "confidence_threshold": 0.3,
    "iou_threshold": 0.5,
    "image_size": 1280,
    "class_ids": [0, 1, 2, 3, 5, 7],
    "class_names": {
        0: "pedestrian",
        1: "two_wheeler",
        2: "car",
        3: "two_wheeler",
        5: "bus_truck",
        7: "bus_truck",
    },
    "output_dir": "outputs/videos",
    "report_dir": "outputs/reports",
    "output_width": 1280,
    "show_preview": True,
    "save_output": True,
    "fourcc": "mp4v",
    "device": None,
}


def resolve_path(value: str | Path, root: Path = PROJECT_ROOT) -> Path:
    """Resolve a config path against the project root."""
    path = Path(value)
    return path if path.is_absolute() else (root / path)


def load_config(
    config_path: str | Path,
    overrides: Mapping[str, Any] | None = None,
    root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    """Load YAML config, apply defaults, then CLI overrides.

    Overrides with a value of ``None`` are ignored so callers can pass
    ``argparse`` results directly.
    """
    path = resolve_path(config_path, root)
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Config must be a mapping: {path}")

    config: dict[str, Any] = {**DEFAULTS, **raw}
    if overrides:
        config.update({key: value for key, value in overrides.items() if value is not None})

    config["class_ids"] = [int(class_id) for class_id in config["class_ids"]]
    if not config["class_ids"]:
        raise ValueError("class_ids must not be empty")

    config["class_names"] = {
        int(class_id): str(name)
        for class_id, name in config["class_names"].items()
    }

    config["confidence_threshold"] = float(config["confidence_threshold"])
    config["iou_threshold"] = float(config["iou_threshold"])
    config["image_size"] = int(config["image_size"])
    config["output_width"] = int(config["output_width"])

    return config
