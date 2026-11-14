"""MOT evaluation metrics implemented with NumPy and lap.

Implements the metrics required by the project:
  - MOTA
  - IDF1
  - ID switches
  - MOTP (mean IoU of matched boxes)
  - precision / recall

Matching is per frame using optimal IoU assignment (Jonker-Volgenant).
IDF1 uses a global assignment between ground-truth and predicted track IDs.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    import lap
except ImportError:  # pragma: no cover - lap is in requirements
    lap = None


@dataclass(frozen=True)
class Detection:
    frame: int
    track_id: int
    x: float
    y: float
    width: float
    height: float
    confidence: float = 1.0

    @property
    def xyxy(self) -> tuple[float, float, float, float]:
        return self.x, self.y, self.x + self.width, self.y + self.height


@dataclass
class MatchResult:
    gt: int = 0
    pred: int = 0
    tp: int = 0
    fp: int = 0
    fn: int = 0
    idsw: int = 0
    iou_sum: float = 0.0

    @property
    def mota(self) -> float | None:
        if self.gt == 0:
            return None
        return 1.0 - (self.fn + self.fp + self.idsw) / self.gt

    @property
    def motp(self) -> float | None:
        return self.iou_sum / self.tp if self.tp else None

    @property
    def precision(self) -> float | None:
        denominator = self.tp + self.fp
        return self.tp / denominator if denominator else None

    @property
    def recall(self) -> float | None:
        denominator = self.tp + self.fn
        return self.tp / denominator if denominator else None


def load_mot(path: str | Path) -> dict[int, list[Detection]]:
    """Load a MOT 1.1 file into frame -> detections."""
    frames: dict[int, list[Detection]] = defaultdict(list)
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            if len(parts) < 7:
                continue
            try:
                frame = int(parts[0])
                track_id = int(parts[1])
                x, y, width, height = (float(value) for value in parts[2:6])
                confidence = float(parts[6])
            except ValueError:
                continue
            frames[frame].append(
                Detection(frame, track_id, x, y, width, height, confidence)
            )
    for detections in frames.values():
        detections.sort(key=lambda detection: detection.track_id)
    return dict(frames)


def iou_matrix(
    ground_truth: list[Detection],
    predictions: list[Detection],
) -> np.ndarray:
    if not ground_truth or not predictions:
        return np.zeros((len(ground_truth), len(predictions)), dtype=np.float64)

    gt = np.array([detection.xyxy for detection in ground_truth], dtype=np.float64)
    pred = np.array([detection.xyxy for detection in predictions], dtype=np.float64)

    x1 = np.maximum(gt[:, None, 0], pred[None, :, 0])
    y1 = np.maximum(gt[:, None, 1], pred[None, :, 1])
    x2 = np.minimum(gt[:, None, 2], pred[None, :, 2])
    y2 = np.minimum(gt[:, None, 3], pred[None, :, 3])

    intersection = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    gt_area = (gt[:, 2] - gt[:, 0]) * (gt[:, 3] - gt[:, 1])
    pred_area = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])
    union = gt_area[:, None] + pred_area[None, :] - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


def _assign(cost: np.ndarray, limit: float) -> np.ndarray:
    if lap is None:
        raise RuntimeError("lap is required for MOT evaluation")
    _, row_to_col, _ = lap.lapjv(
        cost,
        extend_cost=True,
        cost_limit=limit,
    )
    return row_to_col


def evaluate(
    ground_truth: dict[int, list[Detection]],
    predictions: dict[int, list[Detection]],
    iou_threshold: float = 0.5,
) -> dict:
    """Evaluate one sequence."""
    result = MatchResult(
        gt=sum(len(items) for items in ground_truth.values()),
        pred=sum(len(items) for items in predictions.values()),
    )

    matches: Counter[tuple[int, int]] = Counter()
    last_prediction: dict[int, int] = {}

    for frame in sorted(set(ground_truth) | set(predictions)):
        gt_frame = ground_truth.get(frame, [])
        pred_frame = predictions.get(frame, [])
        result.fn += len(gt_frame)
        result.fp += len(pred_frame)

        if not gt_frame or not pred_frame:
            continue

        ious = iou_matrix(gt_frame, pred_frame)
        assignment = _assign(1.0 - ious, 1.0 - iou_threshold)

        matched_gt: set[int] = set()
        matched_pred: set[int] = set()
        for gt_index, pred_index in enumerate(assignment):
            if pred_index < 0:
                continue
            overlap = float(ious[gt_index, pred_index])
            if overlap < iou_threshold:
                continue

            gt_id = gt_frame[gt_index].track_id
            pred_id = pred_frame[pred_index].track_id
            result.tp += 1
            result.fn -= 1
            result.fp -= 1
            result.iou_sum += overlap
            matches[(gt_id, pred_id)] += 1
            matched_gt.add(gt_index)
            matched_pred.add(pred_index)

            if gt_id in last_prediction and last_prediction[gt_id] != pred_id:
                result.idsw += 1
            last_prediction[gt_id] = pred_id

    idtp, idfp, idfn, idf1 = _idf1(matches, result.pred, result.gt)
    return {
        "gt": result.gt,
        "pred": result.pred,
        "tp": result.tp,
        "fp": result.fp,
        "fn": result.fn,
        "idsw": result.idsw,
        "mota": result.mota,
        "motp": result.motp,
        "idf1": idf1,
        "idtp": idtp,
        "idfp": idfp,
        "idfn": idfn,
        "precision": result.precision,
        "recall": result.recall,
        "matches": matches,
    }


def _idf1(
    matches: Counter[tuple[int, int]],
    pred_total: int,
    gt_total: int,
) -> tuple[int, int, int, float | None]:
    if not matches or pred_total == 0 or gt_total == 0:
        return 0, pred_total, gt_total, (0.0 if pred_total or gt_total else None)

    gt_ids = sorted({gt_id for gt_id, _ in matches})
    pred_ids = sorted({pred_id for _, pred_id in matches})
    gt_index = {track_id: index for index, track_id in enumerate(gt_ids)}
    pred_index = {track_id: index for index, track_id in enumerate(pred_ids)}

    counts = np.zeros((len(gt_ids), len(pred_ids)), dtype=np.float64)
    for (gt_id, pred_id), count in matches.items():
        counts[gt_index[gt_id], pred_index[pred_id]] = count

    assignment = _assign(-counts, np.inf)
    idtp = 0
    for gt_i, pred_j in enumerate(assignment):
        if pred_j >= 0:
            idtp += int(counts[gt_i, pred_j])

    idfp = pred_total - idtp
    idfn = gt_total - idtp
    denominator = 2 * idtp + idfp + idfn
    idf1 = (2 * idtp / denominator) if denominator else None
    return idtp, idfp, idfn, idf1


def filter_by_class(
    frames: dict[int, list[Detection]],
    track_classes: dict[int, str],
    class_name: str | None,
) -> dict[int, list[Detection]]:
    if class_name is None:
        return frames
    filtered: dict[int, list[Detection]] = defaultdict(list)
    for frame, detections in frames.items():
        for detection in detections:
            if track_classes.get(detection.track_id) == class_name:
                filtered[frame].append(detection)
    return dict(filtered)


def aggregate_sequences(rows: Iterable[dict]) -> dict:
    """Aggregate per-clip rows into macro and total summaries."""
    rows = list(rows)
    valid = [row for row in rows if row["gt"] > 0]
    totals = {
        key: sum(row[key] for row in rows)
        for key in ("gt", "pred", "tp", "fp", "fn", "idsw")
    }
    metrics = {}
    for key in ("mota", "motp", "idf1", "precision", "recall"):
        values = [row[key] for row in valid if row.get(key) is not None]
        metrics[key] = float(np.mean(values)) if values else None
    return {
        "clips": len(rows),
        **totals,
        **{f"macro_{key}": value for key, value in metrics.items()},
    }
