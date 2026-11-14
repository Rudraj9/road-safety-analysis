# Public Dataset Options — No Self-Recording

**Constraint:** no self-recorded footage. Need enough video to cover 15 clips,
500+ annotated frames with track IDs, and the seven project labels
(`pedestrian`, `two_wheeler`, `rider-helmet`, `rider-no-helmet`, `car`,
`autorickshaw`, `bus_truck`).

**Bottom line:** no single public dataset covers all seven labels with gold
track IDs. The best plan is a **hybrid**:

> Download, extraction, and folder-placement commands are in
> [`dataset-download.md`](dataset-download.md).

1. **AI City Challenge 2023 Track 5** — two-wheelers, helmets, and **track IDs**.
2. **Indian Road Driving Dataset (ThirdEye Labs)** — autorickshaw, pedestrian,
   car, bus/truck, and track IDs (machine-generated).
3. **IDD** — human-labelled autorickshaw still images for detector training.
4. **MOT17** — pedestrian MOT ground truth to build and validate the metric
   harness.

---

## 1. At a glance

| Dataset | What you get | Project labels | Track IDs | Size | Access | Licence | Fit |
|---|---|---|---|---|---|---|---|
| **AI City 2023 Track 5** | 100 videos × 20 s @10 fps, 1920×1080; per-frame boxes | motorbike, driver/pillion helmet & no-helmet | **Yes** | 100 clips, ~33 min | Request form + password | Custom AI City agreement | **Best for two-wheeler + helmet tracking** |
| **AI City 2024 Track 5** | 100 videos × 20 s @10 fps, 1920×1080; up to 4 riders | motorbike, rider helmet/no-helmet | No | 100 clips | Request form + password | Custom AI City agreement | Fallback if 2023 unavailable |
| **Indian Road Driving (ThirdEye)** | 8,441 clips, 646,014 frames @1 fps, BDD100K JSON | person, rider, car, truck, bus, motorcycle, bicycle, **autorickshaw**, animal, traffic light/sign | **Yes** (`track_id`) | ~210 GB, 646 shards | Hugging Face gate (instant) | **CC BY 4.0** | **Best for autorickshaw + mixed traffic** |
| **RideSafe-400 (DashCop)** | 400 dashcam videos, 60–72 s @25 fps, 1080p; 6 classes | rider, motorcycle, helmet, no-helmet, R-M instance, plate | **Yes** | ~600K frames | **"Coming Soon"** — contact authors | TBD | Ideal if released |
| **IDD** | 46,588 detection images; 10,003 segmentation images | 34 classes incl. **autorickshaw**, rider, person, car, truck, bus, motorcycle | No (images) | Tens of GB | Login + licence | Displayed before download | Autorickshaw detector training |
| **MOT17** | 21 train sequences, 15,948 frames, 1,638 tracks | pedestrian only | **Yes** | 5.5 GB + 9.7 MB labels | Direct download | CC BY-NC-SA 3.0 | Metric/tracker validation |
| **MOTOR** (optional) | 1,629 clips, 25+ h, multi-view Indian two-wheeler rides | clip-level events only | No boxes | ~large | HF | CC BY-NC 4.0 | Raw video source only |
| **CAM** (optional) | 30 rides, 12 riders, 177 GB ego video, 1080p30 | YOLO labels for peds, autos, two-wheelers | No | 177 GB | IEEE account | CC BY 4.0 | Auto/pedestrian raw video |

---

## 2. AI City Challenge 2023 — Track 5 (primary)

**Links:** [2023 data & evaluation](https://www.aicitychallenge.org/2023-data-and-evaluation/) ·
[Track 5 download page](https://www.aicitychallenge.org/2023-track5-download/) ·
[2024 version](https://www.aicitychallenge.org/2024-data-and-evaluation/)

### What it contains

- **100 training videos**, each **20 seconds**, **10 fps**, **1920×1080**.
- Ground truth is one row per object per frame:
  `video_id, frame, track_id, bb_left, bb_top, bb_width, bb_height, class`
- Classes:
  1. `motorbike`
  2. `DHelmet` (driver with helmet)
  3. `DNoHelmet` (driver without helmet)
  4. `P1Helmet` / 5. `P1NoHelmet` (pillion 1)
  6. `P2Helmet` / 7. `P2NoHelmet` (pillion 2)
- **Track IDs are included** — this is the key advantage over the 2024 release.

### Why it fits

- Instantly exceeds the PDF minimum of **15 clips** and **500+ frames with track IDs**.
- Helmet is attached to the **rider**, which matches the recommended schema
  (two-wheeler as the tracked object, helmet as an attribute).
- 10 fps is workable for frame-by-frame annotation and interpolation.

### Access and licence

- Fill the **AI City Challenge Datasets Request Form** linked on the data page.
- The Track 5 download page is **password-protected**; the password is provided
  after approval. If there is no response, email the contact listed on the page.
- Acceptance of the **AI City dataset licence agreement** is required. Treat it
  as research-only and **do not redistribute** the videos or labels.
- Verify the current agreement text before publishing anything derived from it.

### Caveats

- Only two-wheeler/helmet classes. No pedestrian, car, autorickshaw, or bus/truck
  labels. Background vehicles are present but unlabelled.
- The 2024 release has up to four riders but **no `track_id`**, so prefer 2023
  for tracking.
- Scene diversity is limited to the challenge's capture locations.

### Use in this project

- Use 8–10 clips as the helmet/two-wheeler core.
- Convert `DHelmet`/`P1Helmet`/`P2Helmet` to `rider-helmet`; `DNoHelmet` etc. to
  `rider-no-helmet`.
- Keep `track_id` and the `motorbike` box as the `two_wheeler` track.
- Review/correct a 500-frame subset in CVAT rather than submitting the downloaded
  labels unchanged (the annotation process is the graded deliverable).

---

## 3. Indian Road Driving Dataset — ThirdEye Labs (primary for other classes)

**Link:** [thirdeyelabs/indian-road-dataset on Hugging Face](https://huggingface.co/datasets/thirdeyelabs/indian-road-dataset)

### What it contains

- **8,441 clips**, **646,014 annotated frames**, **6,896,202 detections**.
- Delhi NCR dashcam footage: day, night, dusk, rain.
- **12 classes:** person, rider, car, truck, bus, motorcycle, bicycle,
  **autorickshaw**, animal, vehicle fallback, traffic light, traffic sign.
- **BDD100K-format JSON** with `track_id` on each label.
- 646 WebDataset tar shards (~1,000 frames each), plus one 1.3 GB
  `annotations/detection.json` covering all frames.
- ~**210 GB** total; 1 frame per second keyframes (not full 25/30 fps video).

### Why it fits

- Directly supplies `autorickshaw`, `pedestrian`, `car`, and `bus/truck` labels,
  which AI City lacks.
- Indian mixed traffic, which is much closer to the project's setting than a
  German highway.
- Track IDs are already present; can be used as CVAT pre-annotations.
- **CC BY 4.0** — usable with attribution, including sharing derived work.

### Access

The dataset is gated but **access is granted immediately** after accepting the
conditions on Hugging Face.

1. Accept **Get access** on the dataset page.
2. Create a read token at <https://huggingface.co/settings/tokens>.
3. Download the annotations and two shards with the project script:

```powershell
$env:HF_TOKEN = "hf_xxxxxxxxxxxxxxxxxxxx"
.\.venv\Scripts\python.exe scripts\download_hf_subset.py --shards 2
Remove-Item Env:\HF_TOKEN
```

Do **not** clone all 210 GB. One or two shards plus the annotation JSON is enough
to sample clips and test the converter. See
[`dataset-download.md`](dataset-download.md) for the official CLI and browser
alternatives.

### Caveats

- **Machine-generated labels, not gold ground truth.** The dataset card states
  that about **33% of detections are below 0.5 confidence** and that a
  human-verified split is still in progress.
- Keyframes are **1 fps**, so tracking is coarse. Do not use it for high-speed
  tracker evaluation; use it for class coverage, pre-annotation, and static-scene
  annotation.
- Dashcam viewpoint and Delhi NCR domain differ from a fixed junction camera.
- 210 GB total; shard boundaries may not align with clips, so check clip IDs
  before selecting.

### Use in this project

- Filter `detection.json` for clips containing `autorickshaw`, `person`, `bus`,
  and `truck`; pick 5–7 clips.
- Use the existing labels as starting points in CVAT, then correct 200–300 frames.
- Treat the original labels as `prelabel`, not `ground truth`, in the report.

---

## 4. IDD — India Driving Dataset (autorickshaw stills)

**Link:** [idd.insaan.iiit.ac.in](https://idd.insaan.iiit.ac.in/) ·
[details](https://idd.insaan.iiit.ac.in/dataset/details/)

- **46,588 detection images** and **10,003 segmentation images** from **182 drive
  sequences** in Hyderabad and Bangalore.
- **34 classes**, including `autorickshaw`, `rider`, `person`, `car`, `truck`,
  `bus`, `motorcycle`, `bicycle`.
- Human-annotated, high quality; images only, no track IDs.
- Login required; the licence is displayed before download.
- Use for **fine-tuning the autorickshaw class** and for still-image detection
  evaluation. Do not count these images as tracking clips.

---

## 5. MOT17 — pedestrian tracking and metric harness

**Link:** [MOT17](https://motchallenge.net/data/MOT17/)

- **21 training sequences**, **15,948 frames**, **1,638 tracks**,
  **336,891 boxes**; 5.5 GB videos + 9.7 MB labels.
- MOTChallenge ground truth; standard MOTA/IDF1/ID-switch evaluation.
- Pedestrians only, no vehicles or helmets.
- **CC BY-NC-SA 3.0**, research/non-commercial; some sequences have third-party
  terms.
- Use it to:
  - validate the ByteTrack/BoT-SORT comparison and the metric code,
  - produce a reference MOTA/IDF1 number before custom data,
  - demonstrate that the evaluation harness works.
- Do **not** present MOT17 results as road-safety results.

---

## 6. RideSafe-400 / DashCop — watch list

**Links:** [project page](https://dash-cop.github.io/) ·
[paper](https://arxiv.org/abs/2503.00428) ·
[code](https://github.com/dash-cop/DashCop)

- 400 dashcam videos, 60–72 s, 25 fps, 1920×1080, ~600K frames.
- Six classes: R-M instance, rider, motorcycle, helmet, no-helmet, licence plate.
- Track IDs, group IDs, triple-riding labels, and plate numbers.
- **Download says "Coming Soon"** as of this check. Contact the authors listed on
  the project page if you need it.
- If released with a usable licence, this is the closest single dataset to the
  entire project (helmet + pillion/triple riding + plates + tracks).

---

## 7. Optional raw video sources

| Dataset | Use | Caveat |
|---|---|---|
| [MOTOR](https://huggingface.co/datasets/varunpaturkar/MOTOR) | 1,629 Indian two-wheeler clips, multi-view, faces/plates already blurred, CC BY-NC 4.0 | Clip-level event labels only; no boxes/tracks |
| [CAM](https://ieee-dataport.org/documents/cam-cognitive-attention-modeling-2-wheeler-riders) | 177 GB ego video from 12 riders, YOLO labels for pedestrians/autos/two-wheelers, CC BY 4.0 | Huge; no helmet labels; IEEE account |

---

## 8. Recommended 15-clip composition

| Source | Clips | Purpose |
|---|---|---|
| AI City 2023 Track 5 | 8 | Helmet/no-helmet, dense two-wheelers, gold track IDs |
| ThirdEye Indian Road | 7 | Autorickshaw, pedestrian, car, bus/truck, Indian mixed traffic |
| **Total** | **15** | Meets the PDF's 15-clip minimum |

Annotation target: **750–900 frames** (50–60 per clip), exceeding the 500-frame
minimum. Split **10 train / 3 val / 2 test by clip**.

If AI City access is delayed:

- Start with ThirdEye (instant access) and MOT17 (direct download).
- Use ThirdEye for all seven object classes, and train the helmet attribute on a
  small Kaggle/Roboflow helmet image set until AI City arrives.
- Note in the report that ThirdEye labels are machine-generated and were
  reviewed/corrected for the annotated subset.

---

## 9. Mapping to the project schema

| Source label | Project label | Notes |
|---|---|---|
| AI City `motorbike` | `two_wheeler` | One box per motorcycle |
| AI City `DHelmet`, `P1Helmet`, `P2Helmet` | `rider-helmet` | Attribute on the two-wheeler track |
| AI City `DNoHelmet`, `P1NoHelmet`, `P2NoHelmet` | `rider-no-helmet` | Attribute on the two-wheeler track |
| ThirdEye `person` | `pedestrian` | Exclude riders |
| ThirdEye `rider` | `two_wheeler` | Verify motorised vs cyclist |
| ThirdEye `autorickshaw` | `autorickshaw` | Direct |
| ThirdEye `car` | `car` | Direct |
| ThirdEye `truck`, `bus` | `bus_truck` | Merge |
| IDD `autorickshaw` | `autorickshaw` | Still images; detector training |
| MOT17 pedestrian | `pedestrian` | Evaluation only |

---

## 10. Licence and ethics summary

| Dataset | Licence | Redistribution |
|---|---|---|
| AI City 2023/2024 | Custom challenge agreement | Do not redistribute; request access per user |
| ThirdEye Indian Road | CC BY 4.0 | Allowed with attribution |
| IDD | Custom; shown at download | Follow the displayed licence |
| MOT17 | CC BY-NC-SA 3.0 | Non-commercial; share-alike |
| RideSafe-400 | TBD | Wait for the release terms |
| MOTOR | CC BY-NC 4.0 | Non-commercial |
| CAM | CC BY 4.0 | Allowed with attribution |

**Do not commit raw datasets to the GitHub repo.** Keep them in `data/raw/`
(gitignored) and publish only derived annotations, aggregate metrics, and
blurred/downscaled demo clips permitted by the licence. Even public datasets may
contain unblurred faces or plates; apply the project's blurring pass before
sharing.

---

## 11. Immediate actions

1. **Request AI City 2023 Track 5** via the dataset request form and email the
   contact if there is no response within a few days.
2. **Accept the ThirdEye gate**, download `annotations/detection.json` and one or
   two shards, and inspect the clip/class distribution.
3. **Download MOT17 labels** (and optionally the 5.5 GB videos) to build the
   metric harness.
4. **Filter ThirdEye** for clips containing autorickshaw, pedestrian, bus, and
   truck; choose 7.
5. **Pick 8 AI City clips** with a mix of helmet/no-helmet and rider counts.
6. **Create 15 CVAT tasks**, import prelabels, and run the 50-frame pilot.
7. **Update the dataset card** with sources, licences, and selection rules.

---

## 12. Sources checked

- AI City Challenge 2023 data and Track 5 pages (2026-11-13).
- AI City Challenge 2024 data page (4-rider variant, no track IDs).
- ThirdEye Labs Indian Road Driving Dataset card and Hugging Face metadata.
- IDD dataset details and download pages.
- MOT17 benchmark page and licence.
- DashCop / RideSafe-400 project page and paper.
- MOTOR and CAM dataset cards.
- BDD100K was considered but its official portal did not load during this check;
  it also lacks autorickshaw and helmet labels, so it is not recommended here.
