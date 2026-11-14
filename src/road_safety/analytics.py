"""Summary analytics over tracked frames."""

from __future__ import annotations

from typing import Any

from road_safety.tracking import Track
from road_safety.video_utils import VideoInfo


def summarize_tracks(
    frame_tracks: list[list[Track]],
    video_info: VideoInfo,
    frames_processed: int,
    elapsed_s: float,
) -> dict[str, Any]:
    """Aggregate detections, unique tracks, and throughput."""
    class_stats: dict[str, dict[str, Any]] = {}
    peak_simultaneous = 0
    total_detections = 0
    confidence_sum = 0.0

    for tracks in frame_tracks:
        peak_simultaneous = max(peak_simultaneous, len(tracks))
        for track in tracks:
            total_detections += 1
            confidence_sum += track.confidence
            stats = class_stats.setdefault(
                track.class_name,
                {"detections": 0, "track_ids": set()},
            )
            stats["detections"] += 1
            stats["track_ids"].add(track.track_id)

    classes = {
        name: {
            "detections": stats["detections"],
            "unique_tracks": len(stats["track_ids"]),
        }
        for name, stats in sorted(class_stats.items())
    }

    unique_track_ids: set[int] = set()
    for tracks in frame_tracks:
        unique_track_ids.update(track.track_id for track in tracks)

    processing_fps = frames_processed / elapsed_s if elapsed_s > 0 else 0.0

    return {
        "video": {
            "path": str(video_info.path),
            "width": video_info.width,
            "height": video_info.height,
            "fps": round(video_info.fps, 3),
            "frame_count": video_info.frame_count,
            "duration_s": round(video_info.duration_s, 3),
        },
        "frames_processed": frames_processed,
        "elapsed_s": round(elapsed_s, 3),
        "processing_fps": round(processing_fps, 3),
        "total_detections": total_detections,
        "unique_tracks": len(unique_track_ids),
        "peak_simultaneous": peak_simultaneous,
        "mean_confidence": (
            round(confidence_sum / total_detections, 4)
            if total_detections
            else 0.0
        ),
        "classes": classes,
    }
