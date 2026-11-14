"""Export tracking results to MOT, CSV, and JSON formats."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable

from road_safety.tracking import Track

MOT_FIELDS = "frame,track_id,x,y,width,height,confidence,-1,-1,-1"


class MOTWriter:
    """Write MOTChallenge 1.1 lines (one row per object per frame).

    MOT does not carry class labels or attributes. Use ``TrackCSVWriter``
    alongside it for class and helmet information.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("w", encoding="utf-8", newline="\n")

    def write(self, frame_number: int, tracks: Iterable[Track]) -> None:
        for track in tracks:
            x, y, width, height = track.xywh
            self._handle.write(
                f"{frame_number},{track.track_id},"
                f"{x:.2f},{y:.2f},{width:.2f},{height:.2f},"
                f"{track.confidence:.4f},-1,-1,-1\n"
            )

    def close(self) -> None:
        self._handle.close()

    def __enter__(self) -> "MOTWriter":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


class TrackCSVWriter:
    """Write per-object rows with class and a helmet attribute placeholder."""

    FIELDS = [
        "frame",
        "track_id",
        "class_id",
        "class_name",
        "confidence",
        "x1",
        "y1",
        "x2",
        "y2",
        "width",
        "height",
        "center_x",
        "center_y",
        "helmet_status",
    ]

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("w", encoding="utf-8", newline="")
        self._writer = csv.DictWriter(self._handle, fieldnames=self.FIELDS)
        self._writer.writeheader()

    def write(self, frame_number: int, tracks: Iterable[Track]) -> None:
        for track in tracks:
            x1, y1, x2, y2 = track.xyxy
            center_x, center_y = track.center
            self._writer.writerow(
                {
                    "frame": frame_number,
                    "track_id": track.track_id,
                    "class_id": track.class_id,
                    "class_name": track.class_name,
                    "confidence": f"{track.confidence:.4f}",
                    "x1": f"{x1:.2f}",
                    "y1": f"{y1:.2f}",
                    "x2": f"{x2:.2f}",
                    "y2": f"{y2:.2f}",
                    "width": f"{x2 - x1:.2f}",
                    "height": f"{y2 - y1:.2f}",
                    "center_x": f"{center_x:.2f}",
                    "center_y": f"{center_y:.2f}",
                    "helmet_status": "",
                }
            )

    def close(self) -> None:
        self._handle.close()

    def __enter__(self) -> "TrackCSVWriter":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
