"""Object detection using a pretrained YOLO model."""

from pathlib import Path

import numpy as np
from ultralytics import YOLO
from ultralytics.engine.results import Results


class ObjectDetector:
    """Detect road users and vehicles in video frames."""

    # COCO class IDs: person, bicycle, car, motorcycle, bus, truck.
    ROAD_CLASSES = [0, 1, 2, 3, 5, 7]

    def __init__(self, model_path: Path, confidence: float = 0.3):
        model_path.parent.mkdir(parents=True, exist_ok=True)

        # Ultralytics downloads the pretrained weights if they are missing.
        self.model = YOLO(str(model_path))
        self.confidence = confidence

    def detect(self, frame: np.ndarray) -> Results:
        """Run object detection on a single BGR image."""
        results = self.model.predict(
            source=frame,
            conf=self.confidence,
            classes=self.ROAD_CLASSES,
            verbose=False,
        )

        return results[0]