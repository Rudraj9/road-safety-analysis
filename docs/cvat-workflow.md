# CVAT Workflow — Setup, Annotation, Export

This is the operational companion to `annotation-schema.md`.

---

## 1. Install CVAT (Docker)

CVAT is the recommended tool because it has first-class **track mode** and
**interpolation**, both required by the PDF.

```powershell
git clone https://github.com/cvat-ai/cvat
cd cvat
docker compose up -d
```

Create an admin user (run once):

```powershell
docker exec -it cvat_server bash -ic "python3 ~/manage.py createsuperuser"
```

Open `http://localhost:8080` and log in.

**Fallback:** Label Studio is lighter to install and supports video tracking,
but its interpolation and track editing are weaker. Use CVAT if Docker is
available.

---

## 2. Create the project

1. **Projects → Create project**
   - Name: `Road Safety Tracking`
   - Labels: `pedestrian`, `two_wheeler`, `car`, `autorickshaw`, `bus_truck`
2. For `two_wheeler`, add attributes:
   | Attribute | Input type | Values | Mutable |
   |---|---|---|---|
   | `helmet_status` | Select | `helmet`, `no_helmet`, `unknown` | yes |
   | `pillion_helmet` | Select | `helmet`, `no_helmet`, `unknown`, `none` | yes |
   | `rider_count` | Select | `1`, `2`, `3+` | yes |
3. Set **Mutable** on all three so they can change along the track.

> If you annotate a pillion separately, do **not** create a second
> `two_wheeler` box on the same vehicle.

---

## 3. Prepare clips

Trim each clip to a 10–15 s segment, resample to 10 fps, and keep 1920 px width
for annotation detail:

```powershell
ffmpeg -i data/raw/videos/RS-01_source.mp4 -ss 00:00:05 -t 12 `
  -vf "fps=10,scale=1920:-2" -c:v libx264 -crf 18 -an `
  data/processed/clips/RS-01.mp4
```

Name files `RS-01.mp4` … `RS-15.mp4` and keep the original alongside them.

---

## 4. Create a task

1. **Tasks → Create task**
   - Name: `RS-01 junction`
   - Project: `Road Safety Tracking`
   - Upload: `RS-01.mp4`
   - Frame step: `1` (the clip is already 10 fps)
   - Segment size: `100` (or the full clip length)
   - Overlap: `0` for the pilot; `10` if two people annotate adjacent segments
2. Assign yourself (and the second annotator) under **Assignees**.
3. Open the task and switch the workspace to **Track** mode.

---

## 5. Annotation workflow

1. Draw a box around an object; CVAT creates a new track with an ID.
2. Set the label and, for `two_wheeler`, the three attributes.
3. Move forward to the next keyframe (every 5th frame) and adjust the box.
4. Use the track's **Interpolate** action to fill between keyframes.
5. Scan every interpolated frame; fix drift and clipping.
6. When an object is occluded, stop the track; start a new one on re-entry if
   the gap is longer than the schema allows.
7. Mark ambiguous objects with a CVAT **Issue** so the pilot review can discuss
   them.

**Do not** change labels after schema freeze without updating the schema
version and re-checking the pilot.

---

## 6. Export

For each task:

1. **Menu → Export task dataset**
2. Choose:
   - **MOT 1.1** → `data/annotations/mot/RS-01.txt` (boxes + IDs for metrics)
   - **Datumaro 1.0** → `data/annotations/cvat/RS-01/` (classes + attributes)
3. Rename the MOT file to the clip ID; MOT itself has no clip name.
4. Store the attribute sidecar as
   `data/annotations/attributes/RS-01.csv`.

MOT 1.1 rows are:

```text
frame, track_id, bb_left, bb_top, bb_width, bb_height, conf, -1, -1, -1
```

Convert Datumaro attribute JSON to the CSV sidecar with a small script once the
pilot export is available (add `scripts/export_attributes.py`).

---

## 7. Review and QC

1. Use **Review** mode for the 10% double-annotated frames.
2. Open an **Issue** for every disagreement; resolve and close it.
3. Export the review report (CVAT analytics) as evidence for milestone 5.
4. Re-export corrected annotations; keep the previous version under
   `data/annotations/archive/`.

---

## 8. Pilot checklist

- [ ] CVAT running and project created
- [ ] 3 clips trimmed, resampled, named `RS-P01` … `RS-P03`
- [ ] 50 frames annotated by 2+ people
- [ ] `helmet_status` Kappa computed
- [ ] Box IoU computed on matched objects
- [ ] Disagreements logged and schema updated to v1
- [ ] MOT + Datumaro exports verified by re-importing one file
