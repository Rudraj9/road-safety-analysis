# Road Safety Analytics

Computer vision pipeline for traffic safety: detect and track pedestrians,
two-wheelers, cars, autorickshaws, and buses/trucks; measure helmet compliance;
and produce MOT/CSV outputs for evaluation and dashboards.

This repository is the prototype for **Project 3** of the AI/ML internship
project list: *Road-Safety Video Tracking: Pedestrians, Two-Wheelers, Helmets
(Video Object Tracking)*.

## Current status

| Area | Status |
|---|---|
| Detection + multi-object tracking (ByteTrack) | **Working** |
| Config-driven CLI, annotated video, MOT/CSV/JSON export | **Working** |
| Annotation schema, CVAT workflow, ethics/dataset templates | **Drafted** (`docs/`) |
| 15 clips + 500 annotated frames | **Pending** (data collection) |
| Custom autorickshaw class + helmet classifier | **Pending** |
| MOTA/IDF1 evaluation harness | **Pending** |
| Streamlit dashboard | **Pending** |

The included `traffic_sample_01.mp4` is a highway clip with only cars and
trucks. It is a **pipeline smoke test**, not evaluation data. The project needs
junction/market footage with two-wheelers and pedestrians — see
[`docs/dataset-card.md`](docs/dataset-card.md).

## What works today

- YOLOv8 detection with configurable inference size and class mapping.
- ByteTrack tracking (`bytetrack.yaml` by default; `botsort.yaml`/`ocsort.yaml`
  can be selected in the config or CLI).
- Annotated output video with persistent IDs and a live HUD.
- MOT 1.1 export for tracking metrics.
- Per-object CSV with class, box, confidence, and a `helmet_status` column
  reserved for the attribute stage.
- JSON summary: detections, unique tracks, peak simultaneous objects, FPS.
- ThirdEye clip selector: ranks downloaded clips, picks a balanced subset, and
  extracts frames + per-frame labels for CVAT.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Place model weights at `models/pretrained/yolov8n.pt` (or set
`model_weights` in `configs/tracking.yaml`). Place videos under
`data/raw/videos/`.

## Run

Preview window:

```powershell
.\.venv\Scripts\python.exe scripts\run_tracking.py
```

Headless smoke test (first 60 frames, smaller inference):

```powershell
.\.venv\Scripts\python.exe scripts\run_tracking.py `
  --no-show --max-frames 60 --imgsz 640 --output-width 1280
```

Custom video and tracker:

```powershell
.\.venv\Scripts\python.exe scripts\run_tracking.py `
  --video data/raw/videos/RS-01.mp4 `
  --tracker botsort.yaml `
  --no-show
```

CLI overrides: `--video`, `--model`, `--tracker`, `--imgsz`, `--conf`,
`--output-width`, `--max-frames`, `--show/--no-show`, `--save-video/--no-save-video`.
All defaults live in [`configs/tracking.yaml`](configs/tracking.yaml).

## Outputs

For an input named `RS-01.mp4`:

| File | Contents |
|---|---|
| `outputs/videos/RS-01_tracked.mp4` | Annotated video (1280 px wide by default) |
| `outputs/reports/RS-01_mot.txt` | MOT 1.1: frame, id, x, y, w, h, conf |
| `outputs/reports/RS-01_tracks.csv` | Per-object rows + `helmet_status` placeholder |
| `outputs/reports/RS-01_summary.json` | Counts, unique tracks, peak, FPS |

MOT coordinates refer to the **original** video resolution, even when the
output video is downscaled.

## Annotation workflow

Read these before collecting or labelling:

1. [`docs/public-datasets.md`](docs/public-datasets.md) — public video datasets
   that remove the need to record footage.
2. [`docs/dataset-download.md`](docs/dataset-download.md) — exact download,
   extraction, and folder-placement steps.
3. [`docs/annotation-schema.md`](docs/annotation-schema.md) — 5 classes,
   helmet/pillion attributes, track lifecycle, interpolation, QC.
4. [`docs/cvat-workflow.md`](docs/cvat-workflow.md) — CVAT setup, task
   creation, annotation, MOT/Datumaro export.
5. [`docs/dataset-card.md`](docs/dataset-card.md) — dataset collection plan.
6. [`docs/ethics.md`](docs/ethics.md) — public-space filming, privacy,
   face/plate blurring.

## Project structure

```text
road-safety-analytics/
├── configs/tracking.yaml          # pipeline configuration
├── data/
│   ├── raw/videos/                # source clips (gitignored)
│   ├── annotations/               # CVAT/MOT exports
│   └── processed/                 # trimmed clips, frames
├── docs/                          # schema, CVAT, dataset card, ethics
├── models/pretrained/             # YOLO weights
├── outputs/{videos,reports}/      # generated artifacts (gitignored)
├── scripts/run_tracking.py        # CLI entry point
├── src/road_safety/
│   ├── config.py                  # YAML loading + path resolution
│   ├── video_utils.py             # probe, resize, writer
│   ├── detection.py               # detector-only wrapper
│   ├── tracking.py                # Track + ByteTrack wrapper
│   ├── export.py                  # MOT, CSV, JSON writers
│   ├── analytics.py               # summary statistics
│   └── pipeline.py                # end-to-end orchestration
└── tests/
```

## Known limitations

- **Classes:** COCO YOLOv8 cannot detect `autorickshaw`, `rider-helmet`, or
  `rider-no-helmet`. `bicycle` is currently merged into `two_wheeler`. A custom
  detector and a helmet attribute classifier are required.
- **Sample footage:** the highway clip has no VRUs, so helmet/compliance
  analytics cannot be demonstrated on it.
- **Evaluation:** no ground truth yet, so MOTA/IDF1/ID-switch numbers cannot be
  produced. `motmetrics` is listed but not yet wired in.
- **Resolution:** 4K input at `imgsz=640` loses small objects; use 1280 or
  record at 1080p.
- **Helmet column:** the CSV column exists but is empty until the attribute
  classifier/annotation import is implemented.

## Tests

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## Roadmap

1. Record 15 clips; fill the dataset card.
2. Run the 50-frame pilot with 2+ annotators; freeze schema v1.
3. Annotate 750–1,000 frames in CVAT; export MOT + attributes.
4. Fine-tune the detector; add the helmet classifier.
5. Compare ByteTrack vs BoT-SORT vs DeepSORT with MOTA/IDF1/ID switches.
6. Build the Streamlit safety dashboard.
7. Write the final report, ethics note, and demo.

See [`docs/PROJECT_3_GAP_ANALYSIS.md`](docs/PROJECT_3_GAP_ANALYSIS.md) for the
full requirements audit and 8-week plan.
