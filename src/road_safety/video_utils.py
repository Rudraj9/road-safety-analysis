"""Video I/O helpers: probing, resizing, and writing annotated output."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class VideoInfo:
    """Basic metadata for an input video."""

    path: Path
    width: int
    height: int
    fps: float
    frame_count: int

    @property
    def duration_s(self) -> float:
        return self.frame_count / self.fps if self.fps else 0.0


def probe_video(path: str | Path) -> VideoInfo:
    """Read video metadata without loading frames into memory."""
    video_path = Path(path)
    if not video_path.is_file():
        raise FileNotFoundError(f"Input video not found: {video_path}")

    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            raise RuntimeError(f"Unable to open video: {video_path}")

        fps = float(capture.get(cv2.CAP_PROP_FPS)) or 25.0
        return VideoInfo(
            path=video_path,
            width=int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fps=fps,
            frame_count=int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        )
    finally:
        capture.release()


def fit_to_width(frame: np.ndarray, width: int) -> np.ndarray:
    """Downscale a frame to ``width`` while preserving aspect ratio."""
    height, current_width = frame.shape[:2]
    if current_width <= width:
        return frame

    scale = width / current_width
    return cv2.resize(
        frame,
        (width, max(2, int(round(height * scale)))),
        interpolation=cv2.INTER_AREA,
    )


def even_size(width: int, height: int) -> tuple[int, int]:
    """Round video dimensions to even numbers for broad codec support."""
    return width + (width % 2), height + (height % 2)


class VideoWriter:
    """Thin context-managed wrapper around ``cv2.VideoWriter``."""

    def __init__(
        self,
        path: str | Path,
        fps: float,
        size: tuple[int, int],
        fourcc: str = "mp4v",
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.size = size
        self._writer = cv2.VideoWriter(
            str(self.path),
            cv2.VideoWriter_fourcc(*fourcc),
            fps,
            size,
        )
        if not self._writer.isOpened():
            raise RuntimeError(
                f"Unable to open video writer: {self.path} (fourcc={fourcc})"
            )

    def write(self, frame: np.ndarray) -> None:
        if (frame.shape[1], frame.shape[0]) != self.size:
            frame = cv2.resize(frame, self.size, interpolation=cv2.INTER_AREA)
        self._writer.write(frame)

    def release(self) -> None:
        self._writer.release()

    def __enter__(self) -> "VideoWriter":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.release()
