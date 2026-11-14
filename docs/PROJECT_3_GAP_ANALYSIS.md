# Project 3 Readiness Audit — Road-Safety Video Tracking

**Project chosen:** Row 3 of the detailed project table in `Data_Annotation_Projects.pdf` (page 3) — *Road-Safety Video Tracking: Pedestrians, Two-Wheelers, Helmets (Video Object Tracking)*.

> Note: page 2 of the PDF lists a different "3rd project" (Sports Action Recognition). This audit follows the numbered row **3** in the detailed table on page 3, which matches the `road-safety-analytics` prototype.

**Repository:** `road-safety-analytics`
**Audit date:** 2026-11-13
**Method:** PDF requirements extracted and read in full; repository inspected file-by-file; environment and sample video tested; YOLO run on sample frames.

> **Update (same day):** the code foundation described as missing in sections 2–4 has since been implemented: config-driven CLI, ByteTrack/BoT-SORT tracking, annotated video, MOT/CSV/JSON export, summary analytics, tests, and annotation/CVAT/ethics documents. Public dataset alternatives to self-recording are catalogued in [`public-datasets.md`](public-datasets.md). Remaining gaps are dataset access, annotation, custom classes/helmet model, MOT metrics, and the dashboard. Sections below remain the baseline audit.


---

## 1. What the PDF actually requires

### 1.1 Core specification (page 3, row 3)

| Dimension | Requirement |
|---|---|
| **Annotation technique** | Video object tracking (frame-by-frame + interpolation), attribute labelling |
| **Raw data** | Self-recorded traffic video; practice/benchmark on MOT17 (`motchallenge.net/data/MOT17/`); own phone/dashcam clips |
| **Tools & libraries** | CVAT (track mode), OpenCV, Ultralytics YOLOv8, ByteTrack/DeepSORT, py-motmetrics, Streamlit |
| **Classes / labels** | `pedestrian`, `two-wheeler`, `rider-helmet`, `rider-no-helmet`, `car`, `autorickshaw`, `bus/truck` |
| **Minimum dataset** | **15 clips**, **500+ annotated frames with track IDs** |
| **Deliverables** | Annotation guidelines, MOT-format annotations, tracking comparison, safety summary dashboard |
| **Evaluation metrics** | MOTA, IDF1, ID switches, IoU, annotation time per frame |
| **Future AI/ML use** | Helmet-violation detector, junction risk analytics |
| **Ethics / permissions** | Film only in public places, never target individuals, blur faces/plates before sharing, get permission where needed |

### 1.2 Milestones (page 4)

| # | Milestone | Required evidence |
|---|---|---|
| 1 | Problem study, tool setup, annotation schema | Problem statement, tool installed (CVAT/Label Studio), draft label list |
| 2 | Raw data collection and cleaning | Dataset description (source, licence, size), duplicates/bad samples removed, 70/15/15 split |
| 3 | Annotation guidelines and pilot | Guideline doc with examples/edge cases, 30–50 pilot samples annotated by 2+ members |
| 4 | Full annotation | Annotated dataset in standard format (YOLO/COCO/MOT) |
| 5 | Quality control | Inter-annotator agreement (Kappa / IoU / WER), error analysis, corrected labels |
| 6 | Baseline model and evaluation | Trained model, metrics, confusion matrix / sample predictions, limitations |
| 7 | Demo, report and presentation | Streamlit/Gradio demo or notebook, GitHub repo, final report, ethics and privacy note |

---

## 2. Where the repository stands today

### 2.1 Inventory

| Area | Current state |
|---|---|
| Code | `detection.py` (32 lines), `tracking.py` (99 lines, duplicate of the script), `video_utils.py` (empty stub), `run_tracking.py` (106 lines) |
| Model | `yolov8n.pt` (COCO-pretrained, 6.5 MB) |
| Data | **1 video**: 3840×2160, 25 fps, 538 frames, **21.5 s** |
| Annotations | None — only `.gitkeep` files |
| Config | `configs/tracking.yaml` exists but **no code reads it** |
| Outputs | Only a `.gitkeep`; no saved annotated video, CSV, or report |
| Docs | README describes a "scaffold only", already out of date |
| Tests | None |
| Git | 1 commit ("The first Commit") |

### 2.2 What the prototype actually does

`scripts/run_tracking.py` opens `traffic_sample_01.mp4` in a preview window, runs YOLOv8n detection on every frame at the default 640 px inference size, draws boxes, and exits on `Q`. That is **detection only** — there is no tracker, no track ID, no class remapping, no saving, no metrics, and no analytics.

### 2.3 Verified test run

I ran the detector on four frames (0, 150, 300, 450) with the project's class filter:

```text
frame   0  -> {'car': 3, 'truck': 1}
frame 150  -> {'car': 2}
frame 300  -> {'car': 1, 'truck': 1}
frame 450  -> {'car': 2, 'truck': 1}
```

**Crucially, the sample clip is a German autobahn-style highway** with no pedestrians, no two-wheelers, no autorickshaws, and therefore no helmets. It cannot demonstrate any of the road-safety behaviours this project is about.

A sample annotated frame is saved at `outputs/reports/sample_detection_frame.jpg`.

### 2.4 Environment

| Package | Status |
|---|---|
| ultralytics 8.4.202 | installed |
| opencv-python 5.0.0.93 | installed |
| torch 2.15.0, numpy 2.5.4, matplotlib 3.11.2 | installed |
| PyYAML 6.0.3 | installed |
| `streamlit` | **missing** |
| `pandas`, `scipy`, `seaborn` | **missing** |
| `motmetrics` | **missing** |
| `lap` (needed by some trackers) | **missing** |
| `pytest` | **missing** |

### 2.5 Completion estimate by PDF milestone

| Milestone | Estimate | Why |
|---|---|---|
| 1. Problem study, tool setup, schema | **25%** | Problem is implicitly chosen, Python env works; no CVAT, no schema, no problem statement in repo |
| 2. Data collection and cleaning | **10%** | One unsuitable clip; no dataset description, licence, dedup, or split |
| 3. Guidelines and pilot | **0%** | Nothing exists |
| 4. Full annotation | **0%** | Nothing exists |
| 5. Quality control | **0%** | Nothing exists |
| 6. Baseline model and evaluation | **20%** | Detection works; no tracking, no training, no metrics, no evaluation harness |
| 7. Demo, report, presentation | **10%** | README only; no demo app, report, or ethics note |
| **Overall** | **≈10–15%** | A working detection demo, not yet a tracking/annotation project |

---

## 3. Requirement-by-requirement gap matrix

Legend: **Met** = present and adequate · **Partial** = exists but insufficient · **Missing** = absent.

| # | Requirement (PDF) | Current state | Status | What is missing / action | Priority |
|---|---|---|---|---|---|
| 1 | 15 traffic clips | 1 clip | **Missing** | Record 15 clips across junction/arterial/market/highway types | P0 |
| 2 | 500+ annotated frames with track IDs | 0 | **Missing** | Define sampling (e.g. 60 frames × 15 clips = 900), annotate in CVAT track mode | P0 |
| 3 | Classes: pedestrian, two-wheeler, rider-helmet, rider-no-helmet, car, autorickshaw, bus/truck | Only COCO person/bicycle/car/motorcycle/bus/truck | **Partial** | Add `autorickshaw`; model helmet as an attribute/class; remap bicycle+motorcycle → two-wheeler | P0 |
| 4 | Video object tracking (frame-by-frame + interpolation) | No tracker; class named "tracking" is a detector loop | **Missing** | Integrate ByteTrack/BoT-SORT; implement keyframe + interpolation workflow in CVAT | P0 |
| 5 | Attribute labelling (helmet compliance) | None | **Missing** | Define per-rider `helmet_status` attribute and annotation rules | P0 |
| 6 | CVAT track mode setup | Not installed/used | **Missing** | Install CVAT/Docker or Label Studio; create project, labels, attributes | P1 |
| 7 | MOT-format annotations | None | **Missing** | Export MOT 1.1 from CVAT; note MOT cannot carry class/attributes — add sidecar CSV/JSON | P1 |
| 8 | Guidelines document with edge cases | None | **Missing** | Write `docs/annotation-guidelines.md` before full annotation | P0 |
| 9 | Pilot: 30–50 samples, 2+ annotators | None | **Missing** | Run pilot on 50 frames from 3 clips; measure agreement | P1 |
| 10 | Inter-annotator agreement (Kappa / IoU) | None | **Missing** | Script to compute box IoU + helmet Cohen's Kappa | P1 |
| 11 | 70/15/15 split | None | **Missing** | Split **by clip** (10/3/2), not by frame | P1 |
| 12 | Dataset description (source, licence, size) | None | **Missing** | `docs/dataset-card.md` | P1 |
| 13 | ByteTrack / DeepSORT comparison | None | **Missing** | Tracker abstraction + `compare_trackers.py` | P1 |
| 14 | py-motmetrics evaluation (MOTA, IDF1, ID switches) | None | **Missing** | Eval harness; verify NumPy 2 compatibility | P1 |
| 15 | IoU + annotation time per frame | None | **Missing** | Log annotation timings; compute IoU vs GT | P2 |
| 16 | Safety summary dashboard (Streamlit) | None | **Missing** | Counts, helmet compliance %, violations, trajectories, heatmap | P1 |
| 17 | Baseline trained model + confusion matrix + limitations | COCO inference only | **Partial** | Fine-tune/replace detector for custom classes; add eval plots | P1 |
| 18 | Helmet-violation detector (future use) | None | **Missing** | Rider ROI classifier + violation event logic | P2 |
| 19 | Junction risk analytics (future use) | None | **Missing** | Zone/line events, near-miss proxies, risk heatmap | P3 |
| 20 | Ethics: public places, blur faces/plates, permission | Not operationalised | **Missing** | Consent log, anonymisation pass, ethics note in repo | P1 |
| 21 | Demo / report / presentation / GitHub repo | README only | **Partial** | Final report, slides, demo video, clean repo | P1 |
| 22 | Working config-driven pipeline | `tracking.yaml` unused | **Partial** | Wire config into CLI; add `--video`, `--save`, `--tracker` | P2 |
| 23 | Saved annotated output video | Config says `save_output: true`, code never saves | **Missing** | Add `VideoWriter` output path | P2 |
| 24 | Code quality / tests | Duplicate 99-line script; `video_utils` empty; no tests | **Partial** | Refactor into one pipeline + unit tests | P2 |

---

## 4. Critical findings (evidence-based)

### 4.1 The sample video is the wrong domain — this is the biggest risk
It is a 4K dashcam-style highway clip with only cars and trucks. The project is about pedestrians, two-wheelers, helmet compliance, autorickshaws, and junction risk. **No amount of modelling can demonstrate those behaviours on this footage.** Data collection is the true blocking task, not code.

### 4.2 "Tracking" does not exist yet
`src/road_safety/tracking.py` is a near-copy of `scripts/run_tracking.py`. It calls `detector.detect(frame)` and draws boxes. There is no `model.track()`, no ByteTrack/BoT-SORT, no track IDs, no persistence, no interpolation. The repository name and filenames currently overstate the implementation.

### 4.3 There is zero annotation infrastructure
No CVAT project, no label schema, no guidelines, no MOT files, no attribute definitions, no pilot, no agreement metrics. Since this is a *data annotation* internship project, annotation is the core deliverable — currently the largest gap.

### 4.4 Custom classes have no plan
COCO YOLOv8 cannot detect `autorickshaw`, `rider-helmet`, or `rider-no-helmet`. Autorickshaws will be predicted as car/truck; helmets are not a COCO class at all. A custom-labelled dataset and a fine-tuned model (or a dedicated helmet classifier) are mandatory. The README acknowledges the helmet limitation but the code has no strategy.

### 4.5 The 4K input is being crushed to 640 px
YOLOv8's default `imgsz=640` downsamples 3840×2160 by ~6×. Distant vehicles and small VRUs disappear. Options: inference at 1280/1536, tiling, or record at 1080p. This alone explains the sparse detections.

### 4.6 Config and output settings are dead code
`configs/tracking.yaml` specifies tracker, class IDs, output directory, `show_preview`, and `save_output`, but no module loads it. `save_output: true` is false in practice — nothing is written to `outputs/videos/`.

### 4.7 Concrete code defects
- `src/road_safety/tracking.py` computes `PROJECT_ROOT` as the `src/` directory and inserts `src/src` into `sys.path` (verified: path does not exist), so running it directly fails to import `road_safety`.
- Two divergent copies of the same script (`scripts/run_tracking.py`, `src/road_safety/tracking.py`) create maintenance drift.
- `video_utils.py` is an empty docstring, so there is no reusable video I/O.
- `requirements.txt` is unpinned and omits `streamlit`, `pandas`, `scipy`, `motmetrics`, `lap`, and test tooling.
- README says "scaffold only / tracking pending", which no longer matches `detection.py` and misrepresents progress in either direction.

### 4.8 No evaluation harness means no way to prove the project works
No ground truth, no `py-motmetrics`, no tracker comparison, no IoU computation, no annotation-time logging. The PDF's metrics (MOTA, IDF1, ID switches) cannot be produced today.

### 4.9 Ethics requirements are not operationalised
No consent/permission log, no face/plate blurring utility, no ethics note, no data-sharing policy. This is an explicit PDF requirement, not optional polish.

### 4.10 py-motmetrics compatibility risk
`py-motmetrics` is effectively unmaintained and has known issues with NumPy 2.x (the environment has NumPy 2.5.4). Budget time to pin `numpy<2` or implement/port the metrics. Do not discover this in the final week.

---

## 5. What is already good

- **Working environment**: Python 3.13 venv with Ultralytics, PyTorch, OpenCV — the hardest setup step is done.
- **Clean package layout**: `src/road_safety/`, `scripts/`, `configs/`, `data/`, `models/`, `outputs/` is a sensible skeleton.
- **Detection baseline is real**: it loads a local model, filters road classes, and runs at interactive speed.
- **Config-first instinct**: `tracking.yaml` already names tracker, confidence, classes, and outputs — it just needs to be honoured.
- **Clear documentation habit**: README is well structured; it needs updating, not replacing.
- **The project choice itself is strong**: local data, visible social impact, standard metrics, and a demo that is easy to explain. See section 6.

---

## 6. Scope for improvement — how to make this a great project

### 6.1 Data (highest leverage)
- **15 clips with deliberate diversity**: 5 signalised junctions, 5 arterial/mid-block, 3 market/school zones, 2 highway negatives; 60–120 s each; 1080p30; 2 night clips; 1 rain clip if safe.
- **Metadata sheet per clip**: location type, date/time, weather, camera, fps, duration, consent, annotator.
- **Clip-level splits** (10/3/2). Splitting frames randomly leaks near-identical images and inflates metrics.
- **Negative/empty frames** and hard cases: occlusion, truncated riders, multiple riders on one two-wheeler, helmet held on the arm, face-covered helmets.
- **Licence hygiene**: self-recorded data is cleanest; MOT17 is research-only (check licence) and has no helmet/auto-rickshaw labels, so use it for tracking practice, not for the final class set.
- **Public pretraining data**: use a public helmet dataset (verify licence) and/or auto-rickshaw images to supplement, but keep a clearly labelled local test set.

### 6.2 Annotation
- **Two-pass workflow**: (1) auto-label with YOLOv8 + ByteTrack, (2) human corrects in CVAT track mode. Measure manual vs AI-assisted time per frame — this is exactly the kind of evidence an annotation internship values.
- **Keyframe + interpolation**: annotate every 5th–10th frame, interpolate between, then correct. Record time saved.
- **Attribute design**: track `two_wheeler` boxes, and attach `helmet_status ∈ {helmet, no_helmet, unknown}` per rider/frame. This is cleaner than seven detector classes and still yields the PDF's label set.
- **MOT export nuance**: MOT 1.1 carries only frame, id, box, confidence — **not class or attributes**. Export MOT for metrics plus a sidecar `attributes.csv` (or Datumaro JSON) for class/helmet. Mention this in the report; it is a genuine data-engineering insight.
- **Guidelines before bulk work**: 2 pages of rules plus 20 labelled examples beat 20 pages of prose.
- **Pilot and adjudication**: 50 frames, 2+ annotators, compute IoU and Kappa, resolve disagreements, then freeze the schema.
- **QC**: double-annotate 10% of frames, maintain an error taxonomy (missed VRU, ID switch, wrong helmet, box drift) and a corrected-labels log.

### 6.3 Modelling
- **Detector**: start from YOLOv8s (not n) at `imgsz=1280`; fine-tune on the 5 object classes; consider tiling for wide junction shots.
- **Tracker comparison**: ByteTrack vs BoT-SORT (built into Ultralytics) and DeepSORT (`deep-sort-realtime`) on identical detections. Report MOTA, IDF1, ID switches, and speed (FPS).
- **Helmet classifier**: crop the rider region from the two-wheeler box, classify helmet/no-helmet with a small CNN (YOLOv8-cls or ResNet18); report per-class F1 and a confusion matrix; abstain (`unknown`) on low confidence/blur.
- **Association logic**: link rider crop to two-wheeler track; count one violation per track, not per frame, to avoid inflation.
- **Authorickshaw**: fine-tune is required; do not silently merge it into car.
- **Thresholds**: confidence 0.3 is a reasonable start but tune on the val split; report precision/recall, not just mAP.

### 6.4 Analytics and dashboard
- **Core KPIs**: counts by class and direction, two-wheeler share, helmet compliance rate, violations per minute, unique tracks, average track duration.
- **Events**: rider without helmet, two-wheeler on footpath/stop-line (zone-based), wrong-way movement, sudden deceleration/near-miss proxy.
- **Visuals**: trajectory plot, dwell/heat map, per-hour bar chart, clip selector, confidence slider.
- **Streamlit app**: upload/select clip → run pipeline → show KPIs, event table, downloadable CSV, and annotated video.
- **Near-miss analytics** (stretch): time-to-collision (TTC) or post-encroachment time (PET) between tracks; junction risk score by zone. This is the PDF's "junction risk analytics" future use and a strong differentiator.

### 6.5 Engineering quality
- One orchestrator (`pipeline.py`) used by both CLI and Streamlit; no duplicated scripts.
- Config-driven CLI (`--config`, `--video`, `--tracker`, `--save`, `--no-preview`).
- `video_utils.py` for reader/writer/resize; `export.py` for MOT/CSV; `metrics.py` for MOTA/IDF1/IoU; `analytics.py` for events.
- Unit tests for geometry/IoU, MOT parsing, helmet event de-duplication, and config loading; a tiny synthetic video fixture.
- Pin dependencies; add `pyproject.toml`; add a `Makefile` or PowerShell task script.
- Model card + dataset card + ethics note; log seeds and versions for reproducibility.
- Privacy pipeline: face/plate blurring before any shared output; keep raw footage local.

### 6.6 Presentation
- 60–90 s demo video with before/after (manual vs AI-assisted annotation, raw vs tracked).
- One chart that tells the story: helmet compliance by junction/time.
- Honest limitations slide: small local sample, daylight bias, occlusion, helmet ambiguity, tracker failure cases.
- The report should answer "how much time did AI-assisted annotation save?" — a concrete number is more memorable than mAP alone.

---

## 7. Recommended target architecture

```text
road-safety-analytics/
├── app/
│   └── streamlit_app.py          # dashboard: KPIs, events, video
├── configs/
│   ├── pipeline.yaml             # input, model, tracker, output
│   ├── data.yaml                 # dataset paths, class names
│   └── train.yaml                # fine-tuning hyperparameters
├── data/
│   ├── raw/videos/               # RS-01..RS-15 (gitignored)
│   ├── annotations/
│   │   ├── cvat/                 # CVAT exports (Datumaro/MOT)
│   │   ├── mot/                  # GT + predictions in MOT 1.1
│   │   └── attributes.csv        # track_id, class, helmet_status, frame
│   ├── processed/                # frames, crops, blurred copies
│   └── splits/                   # train/val/test clip lists
├── docs/
│   ├── annotation-guidelines.md
│   ├── dataset-card.md
│   ├── ethics.md
│   └── report.md
├── models/
│   ├── pretrained/               # yolov8s.pt
│   └── trained/                  # detector + helmet classifier
├── scripts/
│   ├── extract_frames.py
│   ├── run_pipeline.py
│   ├── evaluate_tracking.py
│   ├── compare_trackers.py
│   └── anonymize.py
├── src/road_safety/
│   ├── config.py
│   ├── video_utils.py
│   ├── detection.py              # detector wrapper
│   ├── tracking.py               # ByteTrack/BoT-SORT/DeepSORT wrappers
│   ├── helmet.py                 # rider crop + helmet classifier
│   ├── analytics.py              # counts, compliance, events, zones
│   ├── metrics.py                # MOTA/IDF1/ID switches/IoU
│   ├── export.py                 # MOT/CSV writers
│   └── pipeline.py               # single orchestration path
├── tests/
└── outputs/{videos,reports}/
```

---

## 8. Eight-week execution plan

| Week | PDF milestone | Work | Evidence produced |
|---|---|---|---|
| 1 | M1 | Fix scaffold (config wiring, one pipeline, path bug), install CVAT/Label Studio, write problem statement and draft label list | Working CLI, schema v0, CVAT project screenshot |
| 2 | M2 | Record 15 clips (Bhubaneswar junctions/arterials/markets), clean, build metadata sheet, clip-level split | Dataset card, 15 clips, split files |
| 3 | M3 | Write guidelines v1, auto-label + pilot 50 frames with 2 annotators, compute IoU/Kappa, revise schema | Guidelines, pilot MOT, agreement report |
| 4–5 | M4 | Full annotation: 750–900 frames (60/clip), keyframe + interpolation, AI-assisted time logging | MOT + attributes, annotation-time CSV |
| 6 | M5 | 10% double annotation, adjudication, error taxonomy, corrected labels | IAA report, QC log |
| 7 | M6 | Fine-tune detector, train helmet classifier, run ByteTrack/BoT-SORT/DeepSORT, evaluate | Metrics table, confusion matrix, tracker comparison |
| 8 | M7 | Streamlit dashboard, final report, demo video, ethics note, repo cleanup | App, report, slides, demo |

**Effort reality check:** 900 frames × ~3 min/frame manual ≈ 45 person-hours of annotation; AI-assisted can plausibly cut this by 30–50%. Start early; it is the critical path, not the model.

---

## 9. Definition of done (checklist)

- [ ] 15 clips, 60–120 s each, across at least 4 scene types, with metadata and permissions
- [ ] 500+ (target 750–900) frames with persistent track IDs
- [ ] 5 object classes + `helmet_status` attribute, documented rules
- [ ] 30–50 pilot frames annotated by 2+ people; IoU and Kappa reported
- [ ] 10% QC double annotation with adjudicated corrections
- [ ] 70/15/15 clip-level split and dataset card (source, licence, size)
- [ ] MOT 1.1 annotations + attributes sidecar, validated by a parser test
- [ ] Baseline detector + helmet classifier with metrics and confusion matrix
- [ ] ByteTrack/BoT-SORT/DeepSORT comparison: MOTA, IDF1, ID switches, FPS
- [ ] Annotation time per frame: manual vs AI-assisted
- [ ] Streamlit dashboard: counts, compliance rate, violations, trajectories
- [ ] Face/plate blurring utility and ethics/privacy note
- [ ] Final report, slides, 60–90 s demo, reproducible README
- [ ] Config-driven pipeline, tests, pinned dependencies

---

## 10. Key decisions to make early

1. **Helmet as class or attribute?** Recommended: object class `two_wheeler` + per-track `helmet_status` attribute. Simpler detector, cleaner analytics, and still produces the PDF's `rider-helmet` / `rider-no-helmet` labels.
2. **Autorickshaw**: fine-tune a custom class; do not merge into car, or the class list will not match the PDF.
3. **MOT vs Datumaro**: use MOT for metrics, Datumaro/CSV for class and attributes. Document the limitation.
4. **Resolution**: record 1080p for speed, or keep 4K and infer at 1280–1536 with tiling for wide junctions.
5. **Scope control**: helmet + compliance + dashboard is already a full project. Treat wrong-way, near-miss, and speed estimation as stretch goals, not commitments.
6. **Metric tooling**: test `py-motmetrics` with NumPy 2 immediately; if broken, pin NumPy < 2 in a separate evaluation environment or vendor the metric implementations.

---

## 11. Immediate next seven actions

1. Replace the highway clip with 3 pilot junction clips (even phone footage) and confirm YOLO detects two-wheelers and pedestrians at 1080p/1280 inference.
2. Install CVAT (Docker) or Label Studio and create the project with 5 classes + `helmet_status` attribute.
3. Write the one-page annotation guideline and label 10 frames by hand to expose ambiguities.
4. Fix `src/road_safety/tracking.py` path bug, delete the duplicate, and wire `configs/tracking.yaml` into a single CLI.
5. Add `model.track(..., tracker="bytetrack.yaml", persist=True)` and save an annotated output video with IDs.
6. Install `streamlit`, `pandas`, `scipy`, `motmetrics`, `lap`; pin versions in `requirements.txt`.
7. Create `docs/dataset-card.md` and `docs/ethics.md` skeletons so evidence accumulates from day one.

---

## Appendix A — Reproducible commands used in this audit

```powershell
# Environment check
.\.venv\Scripts\python.exe -c "import ultralytics, cv2, torch; print(ultralytics.__version__, cv2.__version__, torch.__version__)"

# Video metadata
.\.venv\Scripts\python.exe -c "import cv2; c=cv2.VideoCapture('data/raw/videos/traffic_sample_01.mp4'); print(c.get(3), c.get(4), c.get(5), c.get(7))"

# Detection smoke test (4 frames)
.\.venv\Scripts\python.exe -c "from ultralytics import YOLO; import cv2; m=YOLO('models/pretrained/yolov8n.pt'); c=cv2.VideoCapture('data/raw/videos/traffic_sample_01.mp4'); ..."
```

## Appendix B — PDF references

- Page 3, row 3: project specification (technique, data, tools, classes, minimum, deliverables, metrics, future use, ethics).
- Page 4: milestone table with outputs/evidence.

Both were extracted from the supplied `Data_Annotation_Projects.pdf` and cross-checked against the repository.
