# Dataset Download and Placement

Two paths. Pick the one you can access first.

| Path | Dataset | Why | Size | Access |
|---|---|---|---|---|
| **A** | AI City 2023 Track 5 | Gold helmet + two-wheeler labels with track IDs | Unknown; expect several GB | Request form + password |
| **B** | ThirdEye Indian Road | Autorickshaw, pedestrian, car, bus/truck; track IDs | ~210 GB total; download ~2–3 GB | Instant HF gate |

**Best:** do both — 8 AI City clips + 7 ThirdEye clips.
**Start today:** use B while A is being approved.

---

## 0. Critical: this repo is inside OneDrive

The project path is:

```text
C:\Users\ASUS\OneDrive\Desktop\road-safety-analytics
```

Anything placed under `data/raw/` will be synced by OneDrive. A few GB may be
acceptable; **210 GB is not**. For large data:

1. Create a folder outside OneDrive, e.g. `C:\datasets\road-safety\`.
2. Put the dataset there.
3. Tell me the absolute path — the config will point to it. No need to copy it
   into the repo.

For AI City (smaller), `data/raw/public/` is fine if you pause OneDrive syncing
or mark the folder "Free up space" afterwards.

**Never** put raw videos in `data/raw/videos/` if they are large; that folder is
inside OneDrive and in the repo tree.

---

## 1. Folder layout

```text
road-safety-analytics/
└── data/
    └── raw/
        ├── videos/                       # existing small test clip
        └── public/                       # new; gitignored
            ├── aicity2023_track5/        # path A
            │   └── <extracted archive as-is>
            ├── indian-road/              # path B
            │   ├── annotations/
            │   │   ├── detection.json
            │   │   └── scene_attributes.json
            │   └── data/
            │       ├── train-00000-of-00646.tar
            │       └── train-00001-of-00646.tar
            └── mot17/                    # optional evaluation data
                └── <MOT17Labels extracted>
```

`data/raw/public/` is ignored by git (`.gitignore`: `data/raw/*`). Do not rename
files inside a dataset; the converter needs the original IDs and structure.

---

## 2. Path A — AI City 2023 Track 5

### 2.1 Request access

1. Open the [2023 data and evaluation page](https://www.aicitychallenge.org/2023-data-and-evaluation/).
2. Fill the **AI City Challenge Datasets Request Form** linked there.
3. Wait for approval and the download password. If there is no reply after a few
   days, email the contact listed on the page (`aicitychallenges@gmail.com`).
4. Open the [Track 5 download page](https://www.aicitychallenge.org/2023-track5-download/)
   and enter the password.

### 2.2 Download

Download the Track 5 archive(s). Keep the downloaded `.zip`/`.tar.gz` in
`Downloads` until extraction succeeds.

Check free space first:

```powershell
Get-PSDrive C | Select-Object Used, Free
```

### 2.3 Extract inside the project

```powershell
$zip = "$env:USERPROFILE\Downloads\AIC2023_Track5.zip"   # change to actual name
$dest = "data\raw\public\aicity2023_track5"

New-Item -ItemType Directory -Force -Path $dest | Out-Null
Expand-Archive -Path $zip -DestinationPath $dest -Force
```

If the archive is `.tar.gz`:

```powershell
tar -xzf "$env:USERPROFILE\Downloads\AIC2023_Track5.tar.gz" -C $dest
```

**Do not rearrange the extracted folders.** Keep the archive's structure. The
importer will adapt to it.

### 2.4 Verify

```powershell
Get-ChildItem data\raw\public\aicity2023_track5 -Recurse -Depth 2 |
  Select-Object FullName, Length |
  Format-Table -AutoSize
```

You are looking for:
- video files (`.mp4`) or per-video frame folders,
- annotation `.txt` files whose lines look like:
  `video_id, frame, track_id, bb_left, bb_top, bb_width, bb_height, class`.

If the structure differs, that is fine — just record the tree.

---

## 3. Path B — ThirdEye Indian Road Dataset

### 3.1 Accept the gate

1. Open the [dataset page](https://huggingface.co/datasets/thirdeyelabs/indian-road-dataset).
2. Log in / sign up for Hugging Face.
3. Click **Get access**, choose "Academic research", and submit. Access is granted
   immediately.

### 3.2 Create a read token

You do not need the `hf` CLI. The project includes a small downloader that uses
the `requests` package already installed in the venv.

1. Open <https://huggingface.co/settings/tokens>.
2. Click **Create new token** → type **Read** → create.
3. Copy the token (`hf_...`). Do not commit it or paste it into the repo.

### 3.3 Download with the project script

Do **not** download all 210 GB. Start with the annotations and two shards:

```powershell
$env:HF_TOKEN = "hf_xxxxxxxxxxxxxxxxxxxx"

.\.venv\Scripts\python.exe scripts\download_hf_subset.py --shards 2

Remove-Item Env:\HF_TOKEN
```

Default destination: `data/raw/public/indian-road`. This downloads:

```text
annotations/detection.json
annotations/scene_attributes.json
data/train-00000-of-00646.tar
data/train-00001-of-00646.tar
```

Need more clips? Raise the shard count:

```powershell
.\.venv\Scripts\python.exe scripts\download_hf_subset.py --shards 4
```

The script skips files that already exist, so adding shards later is safe. It
prints a progress line every ~10% and writes a `.part` file until complete.

If you see `401`, either the token is wrong or you have not clicked **Get access**
on the dataset page yet.

### 3.4 Optional — use the official CLI instead

Only if `pip` works on your machine:

```powershell
.\.venv\Scripts\python.exe -m pip install "huggingface_hub[cli]"
.\.venv\Scripts\hf.exe auth login

.\.venv\Scripts\hf.exe download thirdeyelabs/indian-road-dataset `
  --repo-type dataset `
  --include "annotations/*" `
  --include "data/train-00000-of-00646.tar" `
  --include "data/train-00001-of-00646.tar" `
  --local-dir data\raw\public\indian-road
```

If `hf.exe` is still missing after installation, use the script in 3.3.

### 3.5 Browser fallback

On the dataset page:

1. Open **Files and versions**.
2. Download `annotations/detection.json`, `annotations/scene_attributes.json`.
3. Open `data/` and download `train-00000-of-00646.tar` and
   `train-00001-of-00646.tar`.
4. Recreate the same `annotations/` and `data/` folders under
   `data/raw/public/indian-road/`.

### 3.6 Inspect one shard

```powershell
New-Item -ItemType Directory -Force -Path "data\raw\public\indian-road\shard_00000" | Out-Null
tar -xf "data\raw\public\indian-road\data\train-00000-of-00646.tar" `
  -C "data\raw\public\indian-road\shard_00000"

Get-ChildItem "data\raw\public\indian-road\shard_00000" -File |
  Group-Object Extension |
  Select-Object Name, Count
```

Each shard is WebDataset: `{clip_id}_{frame:04d}.jpg`, `.png`, `.json`.

If the shards are too large for the project drive, put `indian-road` in
`C:\datasets\road-safety\` instead and skip the copy.

---

## 4. Optional — MOT17 labels (evaluation harness)

Start with labels only (9.7 MB); the 5.5 GB videos are optional.

```powershell
New-Item -ItemType Directory -Force -Path "data\raw\public\mot17" | Out-Null
Invoke-WebRequest "https://motchallenge.net/data/MOT17Labels.zip" `
  -OutFile "$env:USERPROFILE\Downloads\MOT17Labels.zip"
Expand-Archive "$env:USERPROFILE\Downloads\MOT17Labels.zip" `
  -DestinationPath "data\raw\public\mot17" -Force
```

MOT17 is only for validating tracker metrics; it is pedestrian-only and
non-commercial (CC BY-NC-SA 3.0).

---

## 5. Verify nothing entered git

```powershell
git status --short
```

You should see only code/docs changes — **no** `.mp4`, `.tar`, `.zip`, or
`detection.json` under `data/`. If a dataset path appears, stop and fix
`.gitignore` before committing.

Also check total size:

```powershell
Get-ChildItem data\raw\public -Recurse -File |
  Measure-Object Length -Sum |
  Select-Object Count, @{n='GB';e={[math]::Round($_.Sum/1GB,2)}}
```

---

## 6. Send me the tree

Once the data is in place, run:

```powershell
Get-ChildItem data\raw\public -Recurse -Depth 2 |
  Select-Object FullName |
  Format-Table -AutoSize
```

Send that output (or confirm it matches the layout above). Then I will:

1. Write the importer for the actual structure.
2. Convert annotations to the project's MOT + attributes CSV.
3. Add `configs/data.yaml` with the 15-clip split.
4. Run the pipeline on one clip to verify end-to-end.

---

## 7. Select and extract ThirdEye clips

Once the shards are in place, rank the clips and extract a balanced subset:

```powershell
.\.venv\Scripts\python.exe scripts\select_thirdeye_clips.py --select 7 --extract
```

It scans the per-frame JSONs inside the tar shards (not the 1.2 GB global file),
requires at least 50 frames, 5 autorickshaw objects, and 10 two-wheeler objects,
then favours new scene types and times of day.

Outputs:

```text
outputs/reports/thirdeye_clip_candidates.csv   # all 26 clips + scores
outputs/reports/thirdeye_clip_candidates.json  # same, machine-readable
data/processed/clips/<clip_id>/                # jpg + json for selected clips
data/processed/clips/manifest.csv              # clip, frame, image, annotation
```

Review the CSV before annotating; swap clips by editing the `selected` flags or
re-running with different thresholds. ThirdEye has no helmet label, so helmet
attributes still come from AI City.
