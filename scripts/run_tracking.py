"""Read a traffic video and display YOLO object detections."""

import logging
import sys
from pathlib import Path

import cv2


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from road_safety.detection import ObjectDetector


logger = logging.getLogger(__name__)
WINDOW_NAME = "Road Safety Analytics - Object Detection"


def main():
    video_path = (
        PROJECT_ROOT / "data" / "raw" / "videos" / "traffic_sample_01.mp4"
    )
    model_path = PROJECT_ROOT / "models" / "pretrained" / "yolov8n.pt"

    if not video_path.is_file():
        raise FileNotFoundError(f"Input video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    processed_frames = 0

    try:
        if not cap.isOpened():
            raise RuntimeError(f"Unable to open video: {video_path}")

        # Load the detector once before processing frames.
        logger.info("Loading model: %s", model_path.name)
        detector = ObjectDetector(model_path=model_path, confidence=0.3)

        logger.info("Processing video: %s", video_path.name)
        logger.info("Press Q in the preview window to stop.")

        while True:
            success, frame = cap.read()

            if not success:
                if processed_frames == 0:
                    raise RuntimeError("Unable to read the first video frame.")

                logger.info("End of video or unreadable frame.")
                break

            # Detect objects and draw the predictions.
            result = detector.detect(frame)
            annotated_frame = result.plot()

            # Resize only the preview.
            height, width = annotated_frame.shape[:2]

            if width > 1100:
                preview_height = int(height * 1100 / width)
                annotated_frame = cv2.resize(
                    annotated_frame,
                    (1100, preview_height),
                )

            detection_count = (
                len(result.boxes) if result.boxes is not None else 0
            )

            cv2.putText(
                annotated_frame,
                f"Detections: {detection_count}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
            )

            cv2.imshow(WINDOW_NAME, annotated_frame)
            processed_frames += 1

            if cv2.waitKey(1) & 0xFF == ord("q"):
                logger.info("Processing stopped by the user.")
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        logger.info("Processed frames: %d", processed_frames)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s | %(message)s",
    )

    try:
        main()
    except KeyboardInterrupt:
        logger.info("Processing interrupted by the user.")
    except Exception:
        logger.exception("Video processing failed.")
        sys.exit(1)