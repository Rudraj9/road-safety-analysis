# Annotation Schema v0 — Road-Safety Video Tracking

Status: **draft for pilot review**. Freeze after the 30–50 frame pilot.
Project: PDF row 3 — pedestrians, two-wheelers, helmets, autorickshaws.

---

## 1. Label set

Five object classes are tracked; helmet compliance is a track attribute, not a
separate box class.

| ID | Label | Definition | Includes | Excludes |
|---|---|---|---|---|
| 1 | `pedestrian` | Person walking, standing, or waiting on the road, verge, footpath, or crossing | People pushing a cycle; traffic police directing traffic | Riders on a two-wheeler; people inside a vehicle |
| 2 | `two_wheeler` | Motorcycle, scooter, moped, or e-scooter moving on the road | Rider(s) and the vehicle as **one box** | Bicycles (separate future class); parked two-wheelers with no rider |
| 3 | `car` | Passenger car, hatchback, sedan, SUV, jeep, taxi | Cars with roof carriers | Autorickshaw, van, bus, truck |
| 4 | `autorickshaw` | Three-wheeled passenger or commercial auto | Petrol/diesel/CNG autos, e-rickshaws (decision pending) | Cars, tempos, goods carriers |
| 5 | `bus_truck` | Bus, minibus, truck, lorry, tractor-trailer, large goods vehicle | School buses, container trucks | Pickups and small tempos (decide in pilot) |

**Box rule:** tight box around the visible extent. For a two-wheeler, include
the rider's helmet and body but not the shadow. If a rider's arm or leg extends
outside the vehicle silhouette, include it.

**Why not separate `rider-helmet` / `rider-no-helmet` boxes?**
One two-wheeler can carry a helmeted driver and a non-helmeted pillion. Two
boxes would duplicate the vehicle and break tracking. Attributes keep one
vehicle = one track while still producing the PDF's labels.

---

## 2. Attributes (on `two_wheeler` tracks)

| Attribute | Values | Rule |
|---|---|---|
| `helmet_status` | `helmet`, `no_helmet`, `unknown` | Driver only. `helmet` only when the helmet is worn on the head and fastened/covering the skull. |
| `pillion_helmet` | `helmet`, `no_helmet`, `unknown`, `none` | `none` when no pillion. If two or more pillions, use `unknown` and note in comments. |
| `rider_count` | `1`, `2`, `3+` | Count all people on the vehicle, including children. |

### Helmet edge cases

| Situation | Label |
|---|---|
| Helmet worn but chin strap not visible | `helmet` (do not guess fastening) |
| Helmet held in hand / hanging on mirror / in lap | `no_helmet` |
| Cap, scarf, turban, dupatta covering the head | `no_helmet` |
| Full-face helmet, visor up | `helmet` |
| Construction/cycling helmet worn on a motorised two-wheeler | `no_helmet` (not a motorcycle helmet) |
| Rider's head outside frame, behind a pillar, or too blurred | `unknown` |
| Night-time, headlight glare, motion blur | `unknown` if the head is not clearly visible |
| Child standing in front, no helmet | `pillion_helmet = no_helmet` (or `unknown` if head not visible) |

**Never guess.** `unknown` is a valid and useful label; it keeps the compliance
denominator honest.

---

## 3. Track lifecycle rules

1. **Start** a new track when an object first appears, even if only partially
   visible.
2. **Keep the same ID** through occlusions up to **20 frames** (~2 s at 10 fps).
3. **End** a track after the object is absent for more than 20 frames.
4. **Re-entry after more than 30 frames** gets a **new ID**. Do not reconnect
   unless the vehicle is unmistakable (e.g. unique colour + plate visible).
5. **Never swap IDs** between two similar objects, even if they overlap. If
   unsure, end one track and start another.
6. **Parked/stopped** vehicles are tracked only while they interact with the
   road scene; a vehicle parked for the whole clip may be annotated once and
   then ignored (note the rule in the task comments).
7. **Truncated at frame edge:** keep the track alive while any part is visible.
8. **Group of two-wheelers:** one box per vehicle. Do not draw a single box
   around a cluster.

### Interpolation policy

- Annotate a **keyframe every 5th frame** (0.5 s at 10 fps) for moving objects.
- Interpolate between keyframes in CVAT, then **correct every interpolated
  frame** where the box drifts, jumps, or clips the object.
- For stationary objects, two keyframes (start and end) are enough.
- Do not interpolate through an occlusion; break the track instead.

---

## 4. Sampling plan

| Item | Value |
|---|---|
| Clips | 15 (RS-01 … RS-15) |
| Segment per clip | 10–15 s |
| Annotation frame rate | 10 fps (resampled from 25/30 fps) |
| Frames per clip | 60–100 |
| Total annotated frames | 750–1,000 (PDF minimum: 500) |
| Pilot | 50 frames from 3 clips, 2+ annotators |

Clip-level split (not frame-level): **10 train / 3 val / 2 test**. Keep scene
types balanced across splits.

---

## 5. Pilot and quality control

### Pilot (before bulk annotation)

1. Annotate 50 frames from 3 clips with 2+ people using this schema.
2. Compute:
   - box IoU per matched object (target mean ≥ 0.75),
   - track-ID consistency (same ID on the same object across frames),
   - Cohen's Kappa for `helmet_status` (target ≥ 0.80).
3. List every disagreement, resolve it, and update this document.
4. Freeze schema v1 and do not change labels mid-annotation.

### Bulk QC

- Double-annotate **10%** of frames.
- Review every 10th frame plus all `unknown` helmet labels.
- Maintain an error taxonomy: missed VRU, false positive, box drift, ID
  switch, wrong helmet, missed pillion.
- Keep a corrected-labels log with frame, track ID, old value, new value,
  reason.

---

## 6. Export format

| Artifact | Format | Contents |
|---|---|---|
| Tracking ground truth | MOT 1.1 | `frame, id, x, y, w, h, conf, -1, -1, -1` |
| Classes + attributes | CSV sidecar | `frame, track_id, class_name, helmet_status, pillion_helmet, rider_count` |
| CVAT source | Datumaro JSON | Preserves attributes and review history |

**Known limitation:** MOT 1.1 has no class or attribute columns, so the class
and helmet labels must live in the CSV sidecar. This is expected; document it in
the report rather than trying to force attributes into MOT.

---

## 7. Open decisions (resolve in pilot)

- [ ] Do e-rickshaws count as `autorickshaw` or a separate class?
- [ ] Are pickups/small tempos `car` or `bus_truck`?
- [ ] Are bicycles annotated now or deferred? (PDF says two-wheeler; bicycles
      are COCO class 1 but not in the target label set.)
- [ ] Do we annotate parked two-wheelers with no rider?
- [ ] How do we treat riders whose head is visible only in reflection/shadow?
- [ ] Should pillion helmet be a separate attribute when `rider_count = 3+`?

Record every decision with a date and a one-line rationale.

---

## 8. Version history

| Version | Date | Change |
|---|---|---|
| v0 | 2026-11-13 | Initial draft for pilot |
