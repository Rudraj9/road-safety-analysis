"""Object detection using a pretrained YOLO model."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from ultralytics import YOLO
from ultralytics.engine.results import Results


class ObjectDetector:
    """Detect road users and vehicles in video frames (no tracking)."""

    # COCO class IDs: person, bicycle, car, motorcycle, bus, truck.
    ROAD_CLASSES = [0, 1, 2, 3, 5, 7]

    def __init__(
        self,
        model_path: str | Path,
        confidence: float = 0.3,
        iou: float = 0.5,
        image_size: int = 640,
        classes: list[int] | None = None,
        device: str | None = None,
    ) -> None:
        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)

        # Ultralytics downloads the pretrained weights if they are missing.
        self.model = YOLO(str(model_path))
        self.confidence = confidence
        self.iou = iou
        self.image_size = image_size
        self.classes = classes if classes is not None else self.ROAD_CLASSES
        self.device = device

    def detect(self, frame: np.ndarray) -> Results:
        """Run object detection on a single BGR image."""
        results = self.model.predict(
            source=frame,
            conf=self.confidence,
            iou=self.iou,
            imgsz=self.image_size,
            classes=self.classes,
            device=self.device,
            verbose=False,
        )

        return results[0]
