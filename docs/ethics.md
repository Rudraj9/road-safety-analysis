# Ethics and Privacy Note — Road-Safety Video Tracking

This project films public roads. The PDF requires: film only in public places,
never target individuals, blur faces/plates before sharing, and get permission
where needed. This note turns those rules into concrete practice.

---

## 1. Collection rules

- Film from public spaces or with the property owner's permission. Do not enter
  private property or restricted areas.
- Do not zoom, follow, or single out any person, vehicle, or number plate.
- Prefer wide, scene-level shots over close-ups of individuals.
- Do not film police checkpoints, protests, accidents with identifiable victims,
  or anything sensitive without explicit guidance from the mentor.
- Do not record audio.
- Stop recording if someone objects.

## 2. Consent and permission

- For shopkeepers, security guards, or campus staff who ask what you are doing,
  explain the academic purpose and show the project title.
- Keep a permission log: date, place, who gave permission, contact (optional).
- For public roads, individual consent is not practical; rely on public-space
  filming, minimal collection, and anonymisation before sharing.

## 3. Storage and access

- Raw footage stays on the project machine / external drive, not in the public
  git repository.
- Do not upload raw clips to public cloud drives, WhatsApp groups, or social
  media.
- Share only blurred, downscaled clips with the mentor or team.
- Delete raw footage after the project is graded unless the mentor asks to keep
  it.

## 4. Anonymisation before sharing

- Blur every visible face and number plate in any clip that leaves the local
  machine.
- Blurring must be applied to the whole frame, not just detected boxes, if the
  detector misses a face.
- Keep the unblurred copy local and clearly labelled `RAW — DO NOT SHARE`.
- Record the anonymisation method and date in the dataset card.

Practical approach (to implement in `scripts/anonymize.py`):

1. Run a face detector + plate detector per frame.
2. Expand each box by 15–20%.
3. Apply Gaussian blur or pixelation strong enough to make the identity
   unrecoverable.
4. Re-encode and verify a few random frames by eye.

## 5. Annotation and publication

- Annotators see raw footage; that is necessary for the task. Do not copy frames
  to personal devices.
- Do not publish crops of faces, plates, or identifiable individuals.
- Report aggregate statistics (counts, compliance rates), not individual
  trajectories tied to a person.
- If a clip shows a crash or injury, exclude it from the dataset and report it
  to the mentor.

## 6. Legal and institutional

- Follow your university's research-ethics guidance and the mentor's
  instructions.
- Review applicable Indian data-protection law (e.g. the DPDP Act, 2023) with
  the mentor before publishing any dataset.
- If a public dataset is used (e.g. MOT17), follow its licence — MOT17 is for
  research and is not a source of helmet or autorickshaw labels.

## 7. Incident handling

If identifiable footage is shared by mistake:

1. Remove the file from the shared location.
2. Notify the mentor and the affected team.
3. Record what happened and what was changed.
4. Re-share only the anonymised version.

---

_This note is a working document; update it as the project and its data flows
change._
