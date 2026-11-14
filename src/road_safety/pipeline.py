"""End-to-end detection, tracking, export, and summary pipeline."""

from __future__ import annotations

import logging
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np

from road_safety.analytics import summarize_tracks
from road_safety.config import PROJECT_ROOT, load_config, resolve_path
from road_safety.export import MOTWriter, TrackCSVWriter, write_json
from road_safety.tracking import ObjectTracker, Track
from road_safety.video_utils import (
    VideoWriter,
    even_size,
    probe_video,
)

logger = logging.getLogger(__name__)

WINDOW_NAME = "Road Safety Analytics - Tracking"
TRACK_COLORS = [
    (56, 56, 255),
    (151, 157, 255),
    (31, 112, 255),
    (29, 178, 255),
    (49, 210, 207),
    (10, 249, 72),
    (23, 204, 146),
    (134, 219, 61),
    (52, 147, 26),
    (187, 212, 0),
    (168, 153, 44),
    (255, 194, 0),
    (147, 69, 52),
    (255, 115, 100),
    (236, 24, 0),
    (255, 56, 132),
    (133, 0, 82),
    (255, 56, 203),
    (200, 149, 255),
    (199, 55, 255),
]


def _color_for(track_id: int) -> tuple[int, int, int]:
    return TRACK_COLORS[track_id % len(TRACK_COLORS)]


def draw_tracks(
    frame: np.ndarray,
    tracks: list[Track],
    scale: float = 1.0,
) -> np.ndarray:
    """Draw boxes and ID labels on a (possibly resized) frame."""
    for track in tracks:
        x1, y1, x2, y2 = (int(round(value * scale)) for value in track.xyxy)
        color = _color_for(track.track_id)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        label = f"#{track.track_id} {track.class_name} {track.confidence:.2f}"
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
        )
        if y1 < text_height + baseline + 4:
            # Box touches the top edge: put the label inside/below it.
            label_top = y1
            text_y = y1 + text_height + 2
        else:
            label_top = y1 - text_height - baseline - 4
            text_y = y1 - baseline - 2
        cv2.rectangle(
            frame,
            (x1, label_top),
            (x1 + text_width + 4, label_top + text_height + baseline + 4),
            color,
            cv2.FILLED,
        )
        cv2.putText(
            frame,
            label,
            (x1 + 2, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return frame


def draw_hud(
    frame: np.ndarray,
    frame_number: int,
    total_frames: int,
    tracks: list[Track],
    fps: float,
) -> np.ndarray:
    """Draw frame, track, and per-class counters on a translucent panel."""
    total = total_frames if total_frames > 0 else 0
    lines = [
        (
            f"Frame {frame_number}/{total}  Tracks {len(tracks)}  {fps:.1f} FPS",
            0.60,
        )
    ]

    counts = Counter(track.class_name for track in tracks)
    if counts:
        summary = "  ".join(f"{name}: {count}" for name, count in sorted(counts.items()))
        lines.append((summary, 0.50))

    widths = [
        cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0]
        for text, scale in lines
    ]
    panel_width = max(widths) + 24
    panel_height = 14 + len(lines) * 26
    panel_height = min(panel_height, frame.shape[0])

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (panel_width, panel_height), (0, 0, 0), cv2.FILLED)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

    y = 28
    for text, scale in lines:
        cv2.putText(
            frame,
            text,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        y += 26

    return frame


def run_pipeline(
    config_path: str | Path,
    root: Path = PROJECT_ROOT,
    overrides: Mapping[str, Any] | None = None,
    show: bool | None = None,
    max_frames: int | None = None,
    save_video: bool | None = None,
) -> dict[str, Any]:
    """Run the full tracking pipeline and return the summary payload."""
    config = load_config(config_path, overrides, root=root)

    video_path = resolve_path(config["input_video"], root)
    model_path = resolve_path(config["model_weights"], root)
    video_info = probe_video(video_path)

    show = config["show_preview"] if show is None else show
    save_video = config["save_output"] if save_video is None else save_video
    if max_frames is not None and max_frames <= 0:
        raise ValueError("max_frames must be a positive integer")

    output_width = min(int(config["output_width"]), video_info.width)
    output_height = int(round(video_info.height * output_width / video_info.width))
    output_width, output_height = even_size(output_width, output_height)
    coordinate_scale = output_width / video_info.width

    stem = video_path.stem
    output_dir = resolve_path(config["output_dir"], root)
    report_dir = resolve_path(config["report_dir"], root)

    video_output_path = output_dir / f"{stem}_tracked.mp4"
    mot_path = report_dir / f"{stem}_mot.txt"
    tracks_csv_path = report_dir / f"{stem}_tracks.csv"
    summary_path = report_dir / f"{stem}_summary.json"

    logger.info("Input: %s", video_path)
    logger.info(
        "Video: %dx%d @ %.2f fps, %d frames",
        video_info.width,
        video_info.height,
        video_info.fps,
        video_info.frame_count,
    )
    logger.info(
        "Model: %s | tracker: %s | imgsz: %d",
        model_path.name,
        config["tracker"],
        config["image_size"],
    )

    tracker = ObjectTracker(
        model_path=model_path,
        class_ids=config["class_ids"],
        class_names=config["class_names"],
        confidence=config["confidence_threshold"],
        iou=config["iou_threshold"],
        image_size=config["image_size"],
        tracker=config["tracker"],
        device=config["device"],
    )

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    video_writer: VideoWriter | None = None
    if save_video:
        video_writer = VideoWriter(
            video_output_path,
            fps=video_info.fps,
            size=(output_width, output_height),
            fourcc=config["fourcc"],
        )

    mot_writer = MOTWriter(mot_path)
    csv_writer = TrackCSVWriter(tracks_csv_path)
    frame_tracks: list[list[Track]] = []
    frame_index = 0
    start_time = time.perf_counter()

    try:
        while max_frames is None or frame_index < max_frames:
            success, frame = capture.read()
            if not success:
                if frame_index == 0:
                    raise RuntimeError("Unable to read the first video frame.")
                break

            _, tracks = tracker.update(frame)
            frame_tracks.append(tracks)

            annotated = cv2.resize(
                frame,
                (output_width, output_height),
                interpolation=cv2.INTER_AREA,
            )
            draw_tracks(annotated, tracks, scale=coordinate_scale)

            elapsed = time.perf_counter() - start_time
            fps = (frame_index + 1) / elapsed if elapsed > 0 else 0.0
            draw_hud(
                annotated,
                frame_number=frame_index + 1,
                total_frames=video_info.frame_count,
                tracks=tracks,
                fps=fps,
            )

            if video_writer is not None:
                video_writer.write(annotated)

            mot_writer.write(frame_index + 1, tracks)
            csv_writer.write(frame_index + 1, tracks)
            frame_index += 1

            if show:
                cv2.imshow(WINDOW_NAME, annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    logger.info("Preview stopped by the user at frame %d.", frame_index)
                    break

            if frame_index % 100 == 0:
                logger.info("Processed %d frames (%.1f FPS)", frame_index, fps)

    finally:
        capture.release()
        if video_writer is not None:
            video_writer.release()
        mot_writer.close()
        csv_writer.close()
        if show:
            cv2.destroyAllWindows()

    elapsed = time.perf_counter() - start_time
    summary = summarize_tracks(
        frame_tracks=frame_tracks,
        video_info=video_info,
        frames_processed=frame_index,
        elapsed_s=elapsed,
    )
    summary["outputs"] = {
        "video": str(video_output_path) if save_video else None,
        "mot": str(mot_path),
        "tracks_csv": str(tracks_csv_path),
        "summary_json": str(summary_path),
    }

    write_json(summary_path, summary)
    logger.info(
        "Done: %d frames, %d detections, %d unique tracks in %.1f s",
        frame_index,
        summary["total_detections"],
        summary["unique_tracks"],
        elapsed,
    )
    logger.info("Video: %s", video_output_path if save_video else "(not saved)")
    logger.info("MOT: %s", mot_path)
    logger.info("CSV: %s", tracks_csv_path)
    logger.info("Summary: %s", summary_path)

    return summary
