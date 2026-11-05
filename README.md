# Road Safety Analytics

## Project Purpose
**Road Safety Analytics** is a computer vision and analytics system designed to enhance traffic safety monitoring. The project will detect and track pedestrians and vehicles, identify motorcycle and bicycle riders' helmet status, and generate road-safety reports from traffic surveillance and roadway video feeds.

## Current Status
> **Project Scaffold Only**  
> The repository is currently initialized with the core folder structure, configuration placeholders, and module scaffolding. Object detection, multi-object tracking, custom model training, and reporting dashboards are pending implementation.

## Planned Features
- **Vehicle & Pedestrian Detection**: Detect key road users including pedestrians, bicycles, cars, motorcycles, buses, and trucks.
- **Multi-Object Tracking**: Track detected objects across video frames with unique IDs (e.g., using ByteTrack).
- **Rider Helmet Status Identification**: Determine whether two-wheeler riders are wearing helmets or not.
- **Road Safety Reporting & Analytics**: Aggregate counts, analyze traffic patterns, and export analytical reports and annotated video output.

> [!NOTE]
> **Important Note on Helmet Detection**:  
> Off-the-shelf pre-trained models (such as COCO-pretrained YOLOv8) can detect common classes like persons and motorcycles, but do not classify helmet compliance out of the box. Helmet detection will require an appropriately trained custom model or fine-tuned weights specifically trained on helmet use datasets.

## Folder Structure and Responsibilities

```text
road-safety-analytics/
├── data/
│   ├── raw/
│   │   └── videos/        # Raw input traffic videos (e.g., traffic_sample_01.mp4)
│   ├── annotations/       # Ground-truth annotations and labeling files
│   └── processed/         # Preprocessed frames and dataset artifacts
├── models/
│   └── pretrained/        # Pretrained model weights (e.g., yolov8n.pt, helmet detector)
├── src/
│   └── road_safety/       # Core library package
│       ├── __init__.py    # Package initialization
│       ├── detection.py   # Loads detector and generates detections
│       ├── tracking.py    # Runs tracking and manages tracking results
│       └── video_utils.py # Video I/O, resizing, and frame preview helpers
├── scripts/
│   └── run_tracking.py    # Command-line entry point for tracking pipeline
├── configs/
│   └── tracking.yaml      # Configuration file for tracking pipeline parameters
├── outputs/
│   ├── videos/            # Rendered video files with bounding boxes & tracks
│   └── reports/           # Generated analytical reports, CSVs, and metrics
├── notebooks/             # Exploratory analysis and prototyping Jupyter notebooks
├── .gitignore             # Git ignore rules for environments, caches, data & models
├── README.md              # Project documentation and setup guide
└── requirements.txt       # Project Python dependencies
```

## Setup Instructions (Windows PowerShell)

### 1. Create a Virtual Environment
From the project root directory (`road-safety-analytics`), create a Python virtual environment named `.venv`:

```powershell
python -m venv .venv
```

### 2. Install Dependencies
Install dependencies directly using the virtual environment's Python executable:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

*(Optional)* If PowerShell script execution policies allow, you may also activate the environment before running commands:
```powershell
.\.venv\Scripts\Activate.ps1
```

## Adding Input Videos
Manually place your downloaded sample video at:
```text
data/raw/videos/traffic_sample_01.mp4
```

Path relative to project root: `data/raw/videos/traffic_sample_01.mp4`.

## Running the Entry Point
Run the placeholder script using your Python environment:

```powershell
python scripts/run_tracking.py
```
Or using the virtual environment's Python executable:
```powershell
.\.venv\Scripts\python.exe scripts/run_tracking.py
```

Expected output:
```text
Project scaffold ready. Tracking implementation is pending.
```

