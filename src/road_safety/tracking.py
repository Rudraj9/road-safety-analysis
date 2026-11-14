"""Multi-object tracking wrappers around Ultralytics trackers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ultralytics import YOLO
from ultralytics.engine.results import Results


@dataclass(frozen=True)
class Track:
    """One tracked object in one frame."""

    track_id: int
    class_id: int
    class_name: str
    confidence: float
    xyxy: tuple[float, float, float, float]

    @property
    def xywh(self) -> tuple[float, float, float, float]:
        x1, y1, x2, y2 = self.xyxy
        return x1, y1, x2 - x1, y2 - y1

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.xyxy
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def extract_tracks(
    result: Results,
    class_names: dict[int, str],
) -> list[Track]:
    """Convert an Ultralytics result into plain ``Track`` records."""
    boxes = result.boxes
    if boxes is None or len(boxes) == 0 or boxes.id is None:
        return []

    ids = boxes.id.int().cpu().tolist()
    class_ids = boxes.cls.int().cpu().tolist()
    confidences = boxes.conf.cpu().tolist()
    coordinates = boxes.xyxy.cpu().numpy()

    tracks: list[Track] = []
    for track_id, class_id, confidence, xyxy in zip(
        ids, class_ids, confidences, coordinates
    ):
        tracks.append(
            Track(
                track_id=int(track_id),
                class_id=int(class_id),
                class_name=class_names.get(int(class_id), str(class_id)),
                confidence=float(confidence),
                xyxy=tuple(float(value) for value in xyxy),
            )
        )

    return tracks


class ObjectTracker:
    """Run detection plus multi-object tracking on a frame stream.

    ``persist=True`` keeps the tracker state between ``update`` calls, so the
    same instance must be used for an entire video.
    """

    def __init__(
        self,
        model_path: str | Path,
        class_ids: list[int],
        class_names: dict[int, str] | None = None,
        confidence: float = 0.3,
        iou: float = 0.5,
        image_size: int = 1280,
        tracker: str = "bytetrack.yaml",
        device: str | None = None,
    ) -> None:
        model_path = Path(model_path)
        if not model_path.is_file():
            raise FileNotFoundError(
                f"Model weights not found: {model_path}. "
                "Place the .pt file under models/pretrained/."
            )

        self.model = YOLO(str(model_path))
        self.class_ids = class_ids
        self.class_names = class_names or {
            class_id: self.model.names[class_id] for class_id in class_ids
        }
        self.confidence = confidence
        self.iou = iou
        self.image_size = image_size
        self.tracker = tracker
        self.device = device

    def update(self, frame: np.ndarray) -> tuple[Results, list[Track]]:
        """Track objects in one BGR frame."""
        results = self.model.track(
            source=frame,
            persist=True,
            tracker=self.tracker,
            conf=self.confidence,
            iou=self.iou,
            imgsz=self.image_size,
            classes=self.class_ids,
            device=self.device,
            verbose=False,
        )
        result = results[0]
        return result, extract_tracks(result, self.class_names)
