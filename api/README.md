# Morphometry API

FastAPI service for MRI torsion examinations: ingests DICOM (UI uploads or
Orthanc), runs the nnUNet segmentation + torsion docker pipeline as background
jobs, and serves results.

## Architecture

```
routers/        HTTP endpoints (examinations, volumes, uploads, jobs, health)
deps.py         DI providers (settings, DB session, store, queue, API-key auth)
ingest/         DICOM → .nii.gz + Examination row (single, multi-series, series-selection, orthanc)
serializers.py  DB rows → response schemas
schemas/        Pydantic request/response + docker-output validation models
db/             SQLModel tables (Examination, Job), WAL SQLite engine, repository
storage/        per-examination .nii.gz files + encoded-PNG cache + orthanc staging
tasks/          RQ queue, worker entrypoint, torsion job, orthanc finalize
domain/encode.py  slice → base64 PNG rendering (worker-side)
domain/masks.py   combine per-region hip/knee/ankle masks → one labelled volume
settings.py     env-driven config (MORPH_API_*)
runtime.py      cached settings/engine/store shared by API + worker
main.py         app factory + lifespan
```

Metadata/status/results live in **SQLite**; large images are **`.nii.gz` files**
on disk (never in the DB or RAM cache). Long jobs run in a **separate RQ worker**
(durable status, GPU serialized by a single worker); the API process never blocks
on docker.

The torsion UI (Cornerstone3D viewer) streams the NIfTI volumes directly from
`routers/volumes.py`:

- `GET /examinations/{id}/volume/image.nii.gz` → the stored `transformed.nii.gz`
  (the combined hip+knee+ankle volume), served via `FileResponse` with HTTP range
  support (206 partial content). The transformed volume is usually `float64`, which
  WebGL cannot texture (Cornerstone renders it black), so the endpoint serves a
  cached `float32` copy (`source/display.nii.gz`, same shape/affine) via
  `store.ensure_display_volume`; volumes already in a GPU-safe dtype are served
  as-is.
- `GET /examinations/{id}/volume/mask.nii.gz` → a single labelled mask (hip=1,
  knee=2, ankle=3) aligned to the image volume, built on demand from the
  per-region masks (`domain/masks.combine_region_masks`) and cached as
  `combined.nii.gz`.

The paths end in `.nii.gz` on purpose: the Cornerstone NIfTI loader decides
whether to gunzip the response solely from the URL pathname suffix (Content-Type
is ignored), so a bare `/volume/image` path makes it parse raw gzip bytes as a
NIfTI header and fail with "Array buffer allocation failed".

These two endpoints accept the API key from **either** the `X-API-Key` header
**or** an `?api_key=` query param (the Cornerstone NIfTI loader can't set
headers), and CORS exposes the range headers so cross-origin streaming works.

## Two-phase torsion upload (series selection)

A PACS export is a whole examination directory: several DICOM series plus junk
files. Instead of guessing which series to use, the UI uploads the whole folder
and lets the user pick, then processing auto-starts:

1. `POST /upload/torsion/series` — stages the upload, groups files by
   `SeriesInstanceUID` (discarding anything not parseable as an image DICOM),
   renders per-series preview slices, and records a **`pending_selection`**
   examination carrying the candidate-series metadata (`Examination.series`). No
   volumes are produced yet. Returns the pending detail (`type: "pending"`, its
   `series` list with `preview_count`). Raw DICOM is staged under
   `{id}/incoming/{series_uid}/` and previews under `{id}/previews/{series_uid}/`.
2. `GET /examinations/{id}/series/{series_uid}/preview/{index}.png` — one preview
   slice (served like the volumes: header-or-`?api_key=` auth, for `<img>` tags).
3. `POST /upload/torsion/select` — body `{examination_id, mode, ...}` with
   `mode: "whole_leg"` (`series_uid`; one series auto-split into hip/knee/ankle) or
   `mode: "regions"` (`hip`/`knee`/`ankle` UIDs; three already-split series).
   Materializes the chosen series from the staged DICOM (reusing the single/multi
   ingest paths), clears the staging, and **auto-dispatches the full pipeline**
   (`api.tasks.dispatch.dispatch_job`, `JobKind.FULL`) — returns `{job_id, ...}` to
   poll. The examination moves `pending_selection → unprocessed → running →
   processed`.

The single-series `POST /upload/` and Orthanc paths are unchanged (they still
ingest one series directly). `enumerate_torsion_series` / `materialize_torsion_
selection` live in `ingest/dicom.py`.

## Run

The API, worker, and Redis are separate processes sharing Redis + the SQLite DB +
the storage dir. Run from the repo root with the venv activated:

```bash
cp api/.env.example .env         # storage dir, redis/db URLs, image tags, API_KEYS, CORS

# Redis (no local redis-server binary — run it in Docker, publish 6379)
docker run -d --name morphometry-redis -p 6379:6379 redis:7   # first time
docker start morphometry-redis                                # subsequently

python -m api.tasks.worker       # one worker on the 'gpu' queue (runs docker images)

# auth off + allow the UI origin so the viewer can fetch volumes cross-origin
MORPH_API_CORS_ALLOW_ORIGINS=http://localhost:3000 \
  uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Auth: send `X-API-Key` when `MORPH_API_API_KEYS` is set (`/health` is exempt; the
volume endpoints also accept `?api_key=`). The worker is the only process that
needs the docker socket. See the repo root `README.md` → "Running the stack" for
the full walkthrough including the frontend.

## Test

```bash
pytest api/tests        # docker, queue, storage all mocked — no GPU/Redis needed
```

## Migrate legacy data

```bash
python scripts/migrate_pickles.py   # api/data/*.pkl → SQLite + .nii.gz files
```

## Repair upside-down volumes

`transform_coordinate_system` reorients to LPI from the DICOM affine alone, so a
series with bad orientation metadata (acquired/stored upside-down — the affine
says LPI while the voxels run ankle→knee→hip) is not corrected, and the
hip/knee/ankle split comes out reversed. Ingest now auto-detects this from the
anatomy (the proximal cross-section is much larger than the ankle) and reverses
z (`api/ingest/dicom.py::_orient_superoinferior`). Examinations ingested before
that fix can be repaired in place from their retained raw `original` volume:

```bash
python scripts/repair_torsion_orientation.py 0003420100      # one exam
python scripts/repair_torsion_orientation.py --all --dry-run # scan, report only
python scripts/repair_torsion_orientation.py --all           # fix all affected
```

It re-splits, overwrites the stored volumes + offsets, and resets the exam to
`unprocessed` (clearing the stale masks/landmarks/torsion) so you re-run
segmentation + torsion. Correctly-oriented exams are left untouched.
