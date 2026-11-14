"""Run detection plus multi-object tracking on a traffic video."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from road_safety.pipeline import run_pipeline


logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Track road users in a video and export MOT/CSV results."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "tracking.yaml",
        help="Path to the tracking YAML config.",
    )
    parser.add_argument("--video", type=Path, help="Override input video path.")
    parser.add_argument("--model", type=Path, help="Override model weights path.")
    parser.add_argument("--tracker", help="Override tracker config (e.g. botsort.yaml).")
    parser.add_argument("--imgsz", type=int, help="Override inference image size.")
    parser.add_argument("--conf", type=float, help="Override confidence threshold.")
    parser.add_argument("--output-width", type=int, help="Override output video width.")
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Process at most N frames (useful for smoke tests).",
    )
    parser.add_argument(
        "--show",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Show a live preview window.",
    )
    parser.add_argument(
        "--save-video",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Write the annotated output video.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    overrides = {
        "input_video": args.video,
        "model_weights": args.model,
        "tracker": args.tracker,
        "image_size": args.imgsz,
        "confidence_threshold": args.conf,
        "output_width": args.output_width,
    }

    try:
        run_pipeline(
            config_path=args.config,
            root=PROJECT_ROOT,
            overrides=overrides,
            show=args.show,
            max_frames=args.max_frames,
            save_video=args.save_video,
        )
    except Exception:
        logger.exception("Tracking pipeline failed.")
        return 1

    return 0


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s | %(message)s",
    )
    raise SystemExit(main())
