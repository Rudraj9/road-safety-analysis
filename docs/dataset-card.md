# Dataset Card — Road-Safety Video Tracking

Status: **template — fill during data collection (milestone 2).**

---

## 1. Overview

| Field | Value |
|---|---|
| Dataset name | Road Safety Tracking — Bhubaneswar (working title) |
| Owner | _your name_ |
| Purpose | Track pedestrians, two-wheelers, cars, autorickshaws, and buses/trucks; measure helmet compliance |
| Collection period | _start – end_ |
| Location(s) | _city / junction types_ (avoid exact coordinates in shared copies) |
| Licence | _e.g. research-only, not for redistribution_ |
| Contact | _email_ |

## 2. Collection

| Field | Value |
|---|---|
| Device | _phone / dashcam model_ |
| Resolution | 1920×1080 (or 3840×2160) |
| Frame rate | 25 / 30 fps |
| Weather | _sunny / overcast / rain_ |
| Time of day | _morning / afternoon / evening / night_ |
| Camera position | _standing / tripod / vehicle mount_ |
| Consent / permission | _public place; no individual targeting; see ethics.md_ |

## 3. Clips

| Clip | Scene type | Location type | Time | Weather | Duration | Frames @10fps | Annotated frames | Split | Notes |
|---|---|---|---|---|---|---|---|---|---|
| RS-01 | Junction | Signalised 4-arm | | | | | | train | |
| RS-02 | Junction | Signalised 3-arm | | | | | | train | |
| RS-03 | Arterial | Mid-block | | | | | | train | |
| RS-04 | Arterial | Mid-block | | | | | | train | |
| RS-05 | Market | Pedestrian-heavy | | | | | | train | |
| RS-06 | School | Pedestrian-heavy | | | | | | train | |
| RS-07 | Junction | Unsignalised | | | | | | train | |
| RS-08 | Arterial | Bus stop | | | | | | train | |
| RS-09 | Junction | Night | | | | | | train | |
| RS-10 | Mixed | Rain / wet | | | | | | train | |
| RS-11 | Junction | Signalised | | | | | | val | |
| RS-12 | Market | Two-wheeler-heavy | | | | | | val | |
| RS-13 | Arterial | Night | | | | | | val | |
| RS-14 | Junction | Unsignalised | | | | | | test | |
| RS-15 | Highway | Negative / fast traffic | | | | | | test | |

## 4. Label distribution

_Fill after annotation._

| Class | Boxes | Tracks | Clips present |
|---|---|---|---|
| pedestrian | | | |
| two_wheeler | | | |
| car | | | |
| autorickshaw | | | |
| bus_truck | | | |

Helmet status (two-wheeler tracks):

| Status | Count |
|---|---|
| helmet | |
| no_helmet | |
| unknown | |

## 5. Splits

| Split | Clips | Frames | Rule |
|---|---|---|---|
| Train | RS-01 … RS-10 | | Clip-level split; no frame leakage |
| Val | RS-11 … RS-13 | | |
| Test | RS-14 … RS-15 | | |

## 6. Known gaps and biases

- Daylight-heavy; night and rain under-represented.
- Single city / camera height.
- Occlusion-heavy junctions may be under-sampled.
- Helmet labels depend on visibility; `unknown` is not a failure.
- Autorickshaw and e-rickshaw appearance varies widely.

## 7. Ethics and privacy

See `ethics.md`. Raw footage stays local; shared outputs are blurred and
downscaled. No face crops or plate reads are published.

## 8. Version history

| Version | Date | Change |
|---|---|---|
| v0 | 2026-11-13 | Template created |
