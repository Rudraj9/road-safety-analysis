"""Unit tests for tracking export and summary components."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from road_safety.analytics import summarize_tracks
from road_safety.config import load_config
from road_safety.export import MOTWriter, TrackCSVWriter
from road_safety.tracking import Track
from road_safety.video_utils import VideoInfo, even_size, fit_to_width


def sample_tracks() -> list[Track]:
    return [
        Track(1, 2, "car", 0.9, (10.0, 20.0, 110.0, 120.0)),
        Track(2, 3, "two_wheeler", 0.8, (200.0, 50.0, 260.0, 150.0)),
    ]


class ExportTests(unittest.TestCase):
    def test_mot_and_csv_roundtrip(self) -> None:
        mot_path = PROJECT_ROOT / "outputs" / "reports" / "_test_mot.txt"
        csv_path = PROJECT_ROOT / "outputs" / "reports" / "_test_tracks.csv"
        try:
            with MOTWriter(mot_path) as writer:
                writer.write(1, sample_tracks())

            with TrackCSVWriter(csv_path) as writer:
                writer.write(1, sample_tracks())

            mot_lines = mot_path.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(mot_lines), 2)
            frame, track_id, x, y, width, height, confidence = mot_lines[0].split(",")[:7]
            self.assertEqual(frame, "1")
            self.assertEqual(track_id, "1")
            self.assertAlmostEqual(float(x), 10.0)
            self.assertAlmostEqual(float(y), 20.0)
            self.assertAlmostEqual(float(width), 100.0)
            self.assertAlmostEqual(float(height), 100.0)
            self.assertAlmostEqual(float(confidence), 0.9)

            csv_lines = csv_path.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(csv_lines), 3)  # header + two rows
            self.assertIn("helmet_status", csv_lines[0])
            self.assertIn("two_wheeler", csv_lines[2])
        finally:
            mot_path.unlink(missing_ok=True)
            csv_path.unlink(missing_ok=True)


class AnalyticsTests(unittest.TestCase):
    def test_summary_counts(self) -> None:
        info = VideoInfo(Path("clip.mp4"), 1920, 1080, 25.0, 500)
        frame_tracks = [sample_tracks(), [sample_tracks()[0]]]
        summary = summarize_tracks(frame_tracks, info, frames_processed=2, elapsed_s=1.0)

        self.assertEqual(summary["frames_processed"], 2)
        self.assertEqual(summary["total_detections"], 3)
        self.assertEqual(summary["unique_tracks"], 2)
        self.assertEqual(summary["peak_simultaneous"], 2)
        self.assertEqual(summary["classes"]["car"]["detections"], 2)
        self.assertEqual(summary["classes"]["car"]["unique_tracks"], 1)
        self.assertEqual(summary["classes"]["two_wheeler"]["detections"], 1)
        self.assertAlmostEqual(summary["processing_fps"], 2.0)


class VideoUtilsTests(unittest.TestCase):
    def test_even_size(self) -> None:
        self.assertEqual(even_size(1281, 721), (1282, 722))

    def test_fit_to_width_leaves_small_frames(self) -> None:
        import numpy as np

        frame = np.zeros((100, 50, 3), dtype=np.uint8)
        self.assertIs(fit_to_width(frame, 100), frame)


class ConfigTests(unittest.TestCase):
    def test_load_defaults_and_overrides(self) -> None:
        config = load_config(
            "configs/tracking.yaml",
            overrides={"image_size": 640, "confidence_threshold": None},
            root=PROJECT_ROOT,
        )
        self.assertEqual(config["image_size"], 640)
        self.assertEqual(config["confidence_threshold"], 0.3)
        self.assertEqual(config["class_names"][3], "two_wheeler")
        self.assertEqual(config["class_names"][7], "bus_truck")
        self.assertIsInstance(config["class_ids"], list)


if __name__ == "__main__":
    unittest.main()
