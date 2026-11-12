# Morphometry Web App — API & Frontend

A self-contained guide to the **morphometry web application**: a FastAPI backend
(`api/`) and a Next.js frontend (`frontend/`) that turn an MRI leg-torsion
examination (DICOM) into **femoral and tibial torsion angles**, with an
interactive viewer in which a reader can review and correct the landmarks.

> **Not for clinical use.** This is a research tool. The automated values are
> not validated for diagnosis, and a failed measurement is shown as `0°`
> (see [Known limitations](#known-limitations)).

The measurement library itself (`morphometry/`) is documented in the root
[`README.md`](README.md) and [`CLAUDE.md`](CLAUDE.md). This file covers only the
web application around it.

---

## Contents

1. [What it does](#what-it-does)
2. [Architecture](#architecture)
3. [Intended workflow](#intended-workflow)
4. [Installation & first run](#installation--first-run)
5. [Configuration reference](#configuration-reference)
6. [REST API reference](#rest-api-reference)
7. [Processing pipeline internals](#processing-pipeline-internals)
8. [Data model & storage](#data-model--storage)
9. [Results & landmark format](#results--landmark-format)
10. [Frontend internals](#frontend-internals)
11. [Operations](#operations)
12. [Testing](#testing)
13. [Known limitations](#known-limitations)
14. [Where to look in the code](#where-to-look-in-the-code)

---

## What it does

For one MRI torsion examination (axial hip, knee and ankle stations, either as
one stacked "whole-leg" series or as three separate series), the app:

1. ingests the DICOM, converts it to NIfTI, reorients it to LPI and splits it
   into hip / knee / ankle sub-volumes;
2. segments the bones with an **nnU-Net** ensemble (GPU docker image);
3. computes, per leg, the **femoral torsion** by two methods (**Lee** and
   **Murphy**) and the **tibial torsion** (CPU docker image);
4. shows the volume, the segmentation overlay and the reference lines in a
   **Cornerstone3D** viewer, recomputes the angles live while a reader drags
   the landmarks, and saves the corrected landmarks.

---

## Architecture

```
            ┌──────────────────────┐        ┌──────────────────────┐
 Browser ──►│  Next.js UI :3000    │──────► │  FastAPI app :8000   │
            │  frontend/           │  REST  │  api/main.py         │
            └──────────────────────┘        └───┬──────────┬───────┘
                                                │          │ enqueue job
 Orthanc ── api/orthanc_plugin.py ──► /upload/orthanc      ▼
                                                │   ┌──────────────┐
                                SQLite (api.db) ◄┼──►│ Redis :6379  │
                                .nii.gz files   ◄┤   └──────┬───────┘
                                (storage_dir)    │          │ dequeue
                                                 │   ┌──────▼─────────────────┐
                                                 └──►│ RQ worker              │
                                                     │ python -m api.tasks.worker
                                                     │  ├ docker: nnunet_torsion (GPU)
                                                     │  └ docker: torsion (CPU)
                                                     └────────────────────────┘
```

Four cooperating processes share **Redis**, the **SQLite database** and the
**storage directory**:

| Process | Role | Needs |
|---|---|---|
| **Redis** | RQ broker + job queue | — |
| **FastAPI app** (`uvicorn api.main:app`) | HTTP API: ingest, metadata, results, volume streaming. Never runs models and never blocks on docker. | Python venv |
| **RQ worker** (`python -m api.tasks.worker`) | Runs the jobs: loads volumes, runs the two docker images, stores results. | Python venv, docker socket, NVIDIA GPU + container runtime |
| **Next.js UI** (`npm run dev:webpack` / `npm start`) | The only client of the API. | Node.js |

Design principles:

- **Metadata in SQLite, images on disk.** Rows hold paths; volumes are
  `.nii.gz` files under `MORPH_API_STORAGE_DIR`. Nothing large is held in RAM
  between requests.
- **Durable job state.** Every job has a DB row; `GET /jobs/{id}` survives API
  restarts. On startup the API marks jobs left `running` by a dead process as
  `failed` ("Interrupted by restart").
- **One worker on one queue** (`gpu`) serialises GPU jobs. Do not start a second
  worker on the same GPU.
- **Inference is isolated in docker images**, so the API environment does not
  need the nnU-Net / CUDA stack.

---

## Intended workflow

### From the reader's side (web UI)

```
 Sign in ─► Upload exam folder ─► Pick series ─► (auto) Processing ─► Review/correct ─► Save
            /upload                SeriesPicker   segmentation+torsion  viewer
```

1. **Sign in** at `/login` (only if the API enforces authentication; an open
   dev server needs no login).
2. **Upload** (`/upload`): select the *whole examination folder* as exported
   from the PACS — all series and any junk files. Nothing has to be pre-sorted.
   The API groups the files by `SeriesInstanceUID`, discards anything that is
   not an image DICOM, renders preview slices, and creates the examination in
   status **`pending_selection`**.
3. **Pick the series** (series picker, shown right after the upload or later
   from the examination list via *Select series*). Each candidate series is a
   card with scrollable preview slices. Choose a mode:
   - **Whole leg** — one series containing all three stations, which is split
     into hip / knee / ankle automatically;
   - **Separate hip / knee / ankle** — assign three series.

   Confirming materialises the selection and **starts processing
   automatically**.
4. **Wait for processing.** The UI polls the job until it is `finished` or
   `failed`, then opens the viewer. Segmentation plus torsion typically take
   minutes per exam; jobs queue behind each other.
5. **Review** (`/examinations/<accession>`): the viewer shows axial slices of the
   combined volume, the segmentation overlay (toggle *Show segmentation*) and
   one reference line per measurement axis, labelled with its anatomical
   endpoints and its signed proximal/distal angle. The torsion totals (Lee,
   Murphy, tibial; per leg) are shown in the top-right read-out.
6. **Correct**: drag a landmark endpoint to move it within its slice; scroll the
   mouse wheel *while dragging* to move the whole line to the neighbouring
   slice. The read-out updates live.
7. **Save changes** writes the corrected landmarks back to the API.

The examination list (`/examinations`) shows every exam with its status and the
actions available in that status:

| Status | Meaning | UI action |
|---|---|---|
| `pending_selection` | Uploaded, waiting for the series pick | *Select series* |
| `unprocessed` | Volumes ingested, no job run yet | *Start processing* (full pipeline) |
| `running` | A job is executing | live job status |
| `segmented` | Masks exist, torsion not (yet) computed | *Start processing* (torsion only) |
| `processed` | Masks, angles and landmarks available | *View* |
| `failed` | The last job failed (see `GET /jobs/{id}` → `error`, and the worker log) | *Delete* and re-upload, or re-run via the API |

Every row also has *Delete*, which removes the database row and all files.

### Automatic ingestion from Orthanc

For a PACS-fed deployment, `api/orthanc_plugin.py` runs **inside an Orthanc
server** (Orthanc Python plugin). Every stored instance is POSTed to
`MORPH_API_UPLOAD_URL` (default `http://localhost:8000/upload/orthanc`) with the
`MORPH_API_API_KEY` header. The API accepts an instance only if its tags match a
rule in `api/filter_rules.json` (by default: study description
`MRT Beinachsenmessung` and series description `T2 TSE ax 3 Stacks`), stages it
under `storage_dir/_incoming/<accession>/`, and schedules a debounced
*finalize* job. Once no new instance has arrived for
`MORPH_API_ORTHANC_DEBOUNCE_SECONDS`, the finalize job ingests the staged series
(as a single whole-leg series) and runs the full pipeline, so the exam appears
in the list as `processed` without anyone touching it. **See the scheduler
caveat under [Known limitations](#known-limitations): as shipped, the worker
does not run scheduled jobs.**

### Machine clients

Scripts can drive the same workflow over REST with an `X-API-Key`
(see [REST API reference](#rest-api-reference) for a curl walk-through).

---

## Installation & first run

### Prerequisites

| Component | Version / note |
|---|---|
| Python | 3.10, the repo venv from `requirements.txt` (shared with the library) |
| Node.js + npm | a version supported by Next.js 15 (Node ≥ 18.18) |
| Docker | the worker shells out to `docker run`; the worker's user needs access to the docker socket |
| NVIDIA GPU | + NVIDIA Container Toolkit (`--runtime=nvidia --gpus all`), for the segmentation image |
| Redis | 7, conveniently as a container |
| nnU-Net checkpoints | for building the segmentation image; gitignored, must be present locally |

### 1. Python and Node dependencies

```bash
# repo root
python3.10 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

cd frontend && npm install && cd ..
```

### 2. Build (or pull) the two model images

```bash
# GPU segmentation image; checkpoints must exist under
# api/docker/nnunet_torsion/checkpoints/{hip,knee,ankle}/ before building
cd api/docker/nnunet_torsion && docker build -t swestfechtel/nnunet_torsion:latest . && cd -

# CPU torsion image; the build context is the morphometry/ package directory
cd morphometry && docker build -t swestfechtel/torsion:latest . && cd -
```

The tags are configurable (`MORPH_API_NNUNET_IMAGE`, `MORPH_API_TORSION_IMAGE`).
Rebuild the torsion image whenever the measurement code in `morphometry/`
changes — the worker runs whatever is in the image, not the checkout.

### 3. Configure

```bash
cp api/.env.example api/.env                    # MORPH_API_* settings
cp frontend/.env.example frontend/.env.local    # optional, usually left empty
```

`api/.env` and a repo-root `.env` are both read (`api/.env` wins; real
environment variables win over both). See
[Configuration reference](#configuration-reference).

### 4. Start the stack (four terminals, repo root, venv active)

```bash
# 1. Redis
docker run -d --name morphometry-redis -p 6379:6379 redis:7   # first time
docker start morphometry-redis                                # afterwards

# 2. Worker (the only process that runs docker)
python -m api.tasks.worker

# 3. API
MORPH_API_CORS_ALLOW_ORIGINS=http://localhost:3000 \
  uvicorn api.main:app --host 0.0.0.0 --port 8000

# 4. UI
cd frontend && npm run dev:webpack       # http://localhost:3000
```

Use **`npm run dev:webpack`**, not `npm run dev`: the plain script uses
Turbopack, which skips the webpack hook the Cornerstone viewer needs (web
workers / WASM), and the viewer then silently fails to load volumes.
`npm run build && npm start` always uses webpack.

Check that the API is up: `curl http://localhost:8000/health` → `{"status":"ok"}`.
Interactive OpenAPI docs are at `http://localhost:8000/docs`.

### 5. Users (when authentication is on)

```bash
python -m api.users create alice          # prompts for a password
python -m api.users passwd alice          # change password (invalidates tokens)
python -m api.users list
python -m api.users deactivate alice      # block login, keep the account
python -m api.users delete alice
```

The CLI uses the same `MORPH_API_*` settings, hence the same database, as the API.

### Production notes

- Set `MORPH_API_AUTH_REQUIRED=true`, an explicit `MORPH_API_SECRET_KEY`
  (shared by all API replicas) and explicit `MORPH_API_CORS_ALLOW_ORIGINS`
  (not `*`).
- Serve the UI with `npm run build && npm start`. `NEXT_PUBLIC_*` variables are
  inlined at build time — rebuild after changing them.
- Keep exactly one worker per GPU.
- Back up `storage_dir` (it holds both `api.db` and the volumes).

---

## Configuration reference

### API (`MORPH_API_*`, `api/settings.py`)

| Variable | Default | Meaning |
|---|---|---|
| `MORPH_API_STORAGE_DIR` | `api/data` | Root for `api.db` and per-examination files |
| `MORPH_API_DATABASE_URL` | `sqlite:///<storage_dir>/api.db` | SQLAlchemy URL |
| `MORPH_API_LOG_DIR` | `api/logs` | Rotating logs |
| `MORPH_API_LOG_LEVEL` | `INFO` | |
| `MORPH_API_ON_DUPLICATE` | `reject` | Upload of an existing accession number: `reject` (HTTP error) or `replace` (delete the old exam). A *pending* exam is always replaced. |
| `MORPH_API_REDIS_URL` | `redis://localhost:6379/0` | |
| `MORPH_API_GPU_QUEUE_NAME` | `gpu` | RQ queue name |
| `MORPH_API_JOB_TIMEOUT` | `3600` | Seconds, per RQ job and per `docker run` |
| `MORPH_API_ORTHANC_DEBOUNCE_SECONDS` | `10` | Quiet period before an Orthanc series is finalised |
| `MORPH_API_NNUNET_IMAGE` | `swestfechtel/nnunet_torsion:latest` | Segmentation image |
| `MORPH_API_TORSION_IMAGE` | `swestfechtel/torsion:latest` | Torsion image |
| `MORPH_API_DOCKER_SHM_SIZE` | `32G` | `--shm-size` for both containers |
| `MORPH_API_ENCODE_POOL_SIZE` | OS default | Worker pool for legacy PNG slice rendering |
| `MORPH_API_API_KEYS` | empty | Comma-separated machine keys (`X-API-Key`) |
| `MORPH_API_AUTH_REQUIRED` | `false` | Require a user token or API key on protected endpoints |
| `MORPH_API_SECRET_KEY` | auto | Token signing key; if unset, generated once and stored in `<storage_dir>/.secret_key` |
| `MORPH_API_SESSION_TTL_SECONDS` | `604800` (7 d) | Login token lifetime |
| `MORPH_API_CORS_ALLOW_ORIGINS` | `*` | Comma-separated origins; set to the UI origin |

Authentication is **enforced when `AUTH_REQUIRED=true` or any API key is set**;
otherwise the API is open.

Orthanc plugin (environment of the Orthanc process): `MORPH_API_UPLOAD_URL`,
`MORPH_API_API_KEY`.

### Frontend (`frontend/.env.local`)

| Variable | Default | Meaning |
|---|---|---|
| `NEXT_PUBLIC_MODEL_API` | unset | Explicit API base URL. Set **only** when the API is on a different host than the UI. |
| `NEXT_PUBLIC_MODEL_API_PORT` | `8000` | Port used when the API URL is derived from the page's hostname |
| `NEXT_PUBLIC_API_KEY` | unset | Optional machine key sent alongside the user token (normally unset) |

API URL resolution (`frontend/app/server_config.ts`): explicit
`NEXT_PUBLIC_MODEL_API` → in the browser, *same hostname as the page* on the API
port → during server-side rendering, `http://localhost:<port>`. Deriving the
hostname is what lets the UI work from both `http://localhost:3000` and
`http://<lan-ip>:3000`; a hard-coded `localhost` would be blocked by Chrome's
Private Network Access policy when the page is opened via the LAN IP.

---

## REST API reference

All responses are JSON unless noted. Errors use one envelope:
`{"detail": "<message>", "code": "<machine code>"}` (validation errors:
`code: "validation_error"`, HTTP 422).

### Authentication

| Credential | How to send | For |
|---|---|---|
| User token | `Authorization: Bearer <token>` (media URLs: `?token=`) | humans via the UI |
| API key | `X-API-Key: <key>` (media URLs: `?api_key=`) | scripts, Orthanc |

Tokens are signed (itsdangerous), expire after `SESSION_TTL_SECONDS`, and embed
the user's `token_version`, so changing a password or deactivating a user
invalidates all outstanding tokens. Passwords are argon2-hashed. The query
parameter fallback exists only on the media endpoints, because `<img>` tags and
the Cornerstone NIfTI loader cannot always set headers.

### Endpoints

| Method & path | Auth | Purpose | Response |
|---|---|---|---|
| `GET /health` | open | Liveness | `{"status":"ok"}` |
| `POST /auth/login` | open | Body `{username, password}` | `{access_token, username, expires_in, …}` |
| `GET /auth/me` | user token | Current user | `{username, is_active}` |
| `GET /examinations/` | ✓ | List all examinations | `{"examinations": [Summary…]}` |
| `GET /examinations/{id}` | ✓ | Detail (discriminated by `type`) | `TorsionDetail` or `PendingDetail` |
| `PATCH /examinations/{id}` | ✓ | Body `{landmarks?, status?}` | `{"status":"updated"}` |
| `DELETE /examinations/{id}` | ✓ | Delete row + files | 205, empty body |
| `POST /upload/torsion/series` | ✓ | **Phase 1**: multipart `files` = whole exam folder | 201, `PendingDetail` |
| `POST /upload/torsion/select` | ✓ | **Phase 2**: body `SeriesSelection`; materialises and starts the full job | 202, `{job_id, examination_id}` |
| `POST /upload/` | ✓ | Legacy single-series upload: form `examination_type=torsion` + `files` | 201, `{examination_id}` (no job started) |
| `POST /upload/torsion/multi` | ✓ | Legacy three-series upload: `hip_files`, `knee_files`, `ankle_files` | 201, `{examination_id}` (no job started) |
| `POST /upload/orthanc` | ✓ | One instance from the Orthanc plugin: multipart `file` + `metadata` (simplified JSON) | 202 |
| `POST /model/segmentation/{id}` | ✓ | Run segmentation only | 202, `{job_id, examination_id}` |
| `POST /model/torsion/{id}` | ✓ | Full pipeline if `unprocessed`, else torsion only | 202, `{job_id, examination_id}` |
| `GET /jobs/{job_id}` | ✓ | Job status | `{job_id, examination_id, kind, status, error}` |
| `GET /examinations/{id}/volume/image.nii.gz` | media | Combined hip+knee+ankle volume (float32, byte-range capable) | `application/gzip` |
| `GET /examinations/{id}/volume/mask.nii.gz` | media | Combined label mask (hip=1, knee=2, ankle=3), aligned to the image | `application/gzip` |
| `GET /examinations/{id}/series/{uid}/preview/{i}.png` | media | Preview slice *i* of a candidate series (pending exams) | `image/png` |

The examination id **is the DICOM accession number** (a random one is generated
if the DICOM has none). The volume paths end in `.nii.gz` on purpose: the
Cornerstone loader decides whether to gunzip from the URL suffix alone.

### Schemas

```jsonc
// SeriesSelection (POST /upload/torsion/select)
{ "examination_id": "0003420100", "mode": "whole_leg", "series_uid": "1.2.840…" }
{ "examination_id": "0003420100", "mode": "regions",
  "hip": "1.2.…", "knee": "1.2.…", "ankle": "1.2.…" }

// ExaminationSummary (list entries; also the base of both detail types)
{ "accession_number": "0003420100", "patient_name": "Anonymised",
  "study_date": "…", "study_time": "…", "study_description": "…",
  "status": "processed", "active_job_id": null }

// PendingDetail
{ …summary, "type": "pending",
  "series": [{ "uid": "…", "description": "T2 TSE ax 3 Stacks", "modality": "MR",
               "instances": 120, "rows": 512, "cols": 512, "preview_count": 16 }] }

// TorsionDetail
{ …summary, "type": "torsion",
  "shape": [x, y, z], "knee_offset": 40, "ankle_offset": 80,
  "torsion": { "femoral_torsion_left": 0, "femoral_torsion_right": 0,
               "femoral_torsion_left_murphy": 0, "femoral_torsion_right_murphy": 0,
               "tibial_torsion_left": 0, "tibial_torsion_right": 0 },
  "landmarks": { … see Results & landmark format … },
  "image": [], "segmentation": [] }   // legacy base64 PNG slices
```

Enumerations (`api/schemas/enums.py`):

- examination `status`: `pending_selection`, `unprocessed`, `running`,
  `segmented`, `processed`, `failed`
- job `kind`: `full`, `segmentation`, `torsion`, `orthanc_finalize`
- job `status`: `queued`, `running`, `finished`, `failed`

### Scripted example

```bash
API=http://localhost:8000; KEY='X-API-Key: <key>'

# Phase 1: upload a whole exam folder
args=(); while IFS= read -r -d '' f; do args+=(-F "files=@$f"); done \
  < <(find /path/to/exam -type f -print0)
curl -s -H "$KEY" "${args[@]}" $API/upload/torsion/series > pending.json
ID=$(jq -r .accession_number pending.json); jq '.series[] | {uid, description, instances}' pending.json

# Phase 2: choose one whole-leg series → job starts automatically
JOB=$(curl -s -H "$KEY" -H 'Content-Type: application/json' \
  -d "{\"examination_id\":\"$ID\",\"mode\":\"whole_leg\",\"series_uid\":\"<uid>\"}" \
  $API/upload/torsion/select | jq -r .job_id)

# Poll, then read the results
curl -s -H "$KEY" $API/jobs/$JOB
curl -s -H "$KEY" $API/examinations/$ID | jq '{status, torsion}'
```

---

## Processing pipeline internals

### Status lifecycle

```
upload/torsion/series ─► pending_selection ─(select)─► unprocessed ─(job)─► running
                                                                            │
                     segmented ◄── segmentation stage done ─────────────────┤
                         │                                                  │
                         └─(torsion stage)─► processed      any exception ─► failed
```

The job function is `api/tasks/torsion.py::run_torsion(examination_id, job_id, mode)`
with `mode ∈ {full, segmentation, torsion}`. `api/tasks/dispatch.py` commits the
`Job` row *before* enqueueing, so the worker always finds it.

### 1. Ingest (`api/ingest/dicom.py`, runs in the API process)

- **Series enumeration** (phase 1): files are grouped by `SeriesInstanceUID`;
  unparseable / non-image files are dropped; up to 16 preview PNGs per series
  are rendered. Raw DICOM is staged under `{id}/incoming/{uid}/`, previews
  under `{id}/previews/{uid}/`.
- **Whole-leg series**: a torsion series is often three stations (hip, knee,
  ankle) in one series with large gaps between them. `_detect_slabs` orders the
  slices by position along the slice normal and cuts at the gaps; if it finds
  exactly three slabs, each is converted separately so SimpleITK keeps its true
  z-spacing (converting the gapped whole would average the gaps into a ~2×
  inflated spacing and corrupt every superior–inferior distance). Otherwise the
  whole stack is converted and split at changepoints of the per-slice
  foreground footprint (`_split_volume`).
- **Orientation**: volumes are reoriented to LPI; a series whose affine is
  correct but whose voxels run ankle → hip (upside-down acquisition) is
  detected from the anatomy and flipped (`_orient_superoinferior`).
- **Three-series mode**: the three volumes are concatenated along z.
- **Persisted**: `original` (as acquired), `transformed` (LPI, combined,
  float32 so WebGL can texture it), `hip`, `knee`, `ankle`, and the slice
  offsets `knee_offset` (= hip slice count) and `ankle_offset` (= hip + knee).

### 2. Segmentation stage (worker → `nnunet_torsion` image)

Per region, `{tmp}/{region}/input/{region}_0000.nii.gz` is written, and the
container runs with `{tmp}` mounted at `/app/mnt`:

```
docker run --rm --runtime=nvidia --gpus all --shm-size $SHM --group-add root \
  -v {tmp}:/app/mnt:rw,Z  $MORPH_API_NNUNET_IMAGE
```

It runs nnU-Net v2 (ResEncUNet-XL plans, datasets 004 hip / 006 knee / 007
ankle, 5-fold ensemble) and writes `{tmp}/{region}/output/{region}.nii.gz`. The
worker reorients the masks to LPI, stores them under `{id}/masks/`, builds the
combined label mask for the viewer (`masks/combined.nii.gz`) and the legacy
encoded PNG slices, and sets the exam to `segmented`.

### 3. Torsion stage (worker → `torsion` image)

The three masks are written as `{tmp}/{hip,knee,ankle}_segmentation.nii.gz` and
the container (`morphometry/docker/compute_torsion.py`) runs with `{tmp}`
mounted at `/app/temp`. It splits each mask into left/right image halves and
computes, per leg, femoral torsion (Lee, Murphy) and tibial torsion. Each
measurement is wrapped individually: a failure is recorded in `errors.json`
and that value becomes `NaN`, while the others still complete. Outputs:

| File | Content | Used by the API |
|---|---|---|
| `results.json` | the six angles (+ `*_murphy_qc_flags`) | angles validated (`ResultsModel`) and stored; QC flags **dropped** |
| `landmarks.json` | nested landmark tree | stored as-is |
| `errors.json` | `{"errors": [traceback, …]}` | logged at ERROR in the worker log only |

The exam is then `processed`. Note: the job is `finished` even if every single
measurement failed — check the values and the worker log.

### Orthanc finalize (`api/tasks/orthanc.py`)

Each received instance re-schedules `finalize_orthanc(accession)` after the
debounce window. A finalize that finds an instance newer than the window
returns early (a later one does the work); otherwise it ingests the staged
series, clears the staging, creates a `full` job and runs it inline in the same
worker job.

---

## Data model & storage

### Database (SQLModel, SQLite in WAL mode; `api/db/models.py`)

| Table | Key columns |
|---|---|
| `examinations` | `id` (= accession), `examination_type`, `status`, patient/study fields, `dicom_metadata`, `series` (candidate series, pending only), `torsion_values`, `landmarks`, `source_paths`, `mask_paths`, `encoded_paths`, `shape`, `knee_offset`, `ankle_offset`, timestamps |
| `jobs` | `id` (uuid4), `examination_id`, `kind`, `status`, `rq_job_id`, `error`, `created_at` / `started_at` / `finished_at` |
| `users` | `username`, `password_hash` (argon2), `is_active`, `token_version` |

Tables are created on startup; there is no migration framework.

### Files (`api/storage/store.py`)

```
<storage_dir>/
├── api.db                    # SQLite database
├── .secret_key               # auto-generated token key (if SECRET_KEY unset)
├── _incoming/<accession>/    # Orthanc instances being staged
└── <accession>/
    ├── source/  original.nii.gz  transformed.nii.gz  hip|knee|ankle.nii.gz
    │            display.nii.gz   # cached float32 copy served to the viewer, if needed
    ├── masks/   hip|knee|ankle.nii.gz  combined.nii.gz
    ├── encoded/ image_###.png  seg_###.png      # legacy slice renderings
    ├── incoming/<series_uid>/####.dcm           # pending exams only
    └── previews/<series_uid>/####.png           # pending exams only
```

`original.nii.gz` is kept so an exam can be re-split later (see
`scripts/repair_torsion_orientation.py`).

---

## Results & landmark format

**Side convention.** The result keys `…_left` / `…_right` are the **patient's**
side. In LPI the image-left half is the patient's right leg, so
`femoral_torsion_right` is measured on the image-left half.

**Sign convention.** Femoral torsion: positive = antetorsion, negative =
retrotorsion. Tibial torsion: positive = external, negative = internal. Angles
are folded into (−90°, 90°].

**Landmarks** (`landmarks` in `TorsionDetail`, keys again by patient side):

```jsonc
{
  "femur": {
    "Lee":    { "left":  { "hip_start": [x,y,z], "hip_end": [x,y,z],
                           "knee_start": [x,y,z], "knee_end": [x,y,z] },
                "right": { … } },
    "Murphy": { "left": { … }, "right": { … } }
  },
  "tibia":    { "left":  { "knee_start": […], "knee_end": […],
                           "ankle_start": […], "ankle_end": […] },
                "right": { … } }
}
```

Coordinates are voxel indices: x/y in the full-width (both legs) volume,
z relative to the region's own sub-volume (hip, knee or ankle). The viewer adds
`knee_offset` / `ankle_offset` to place them in the combined volume
(`frontend/app/components/cornerstone/landmark-mapping.ts`). Each `*_start` /
`*_end` pair is one reference line and lies on one axial slice.

**Edited landmarks vs. stored angles.** `PATCH` stores only the landmarks. The
`torsion` block in the API response keeps the **automated** values; the angles
shown in the viewer are recomputed client-side from the current landmarks
(`frontend/app/utils.tsx`). Anyone exporting results from the API has to
recompute the angles from the landmarks to obtain the reader-corrected values.

---

## Frontend internals

Next.js 15 (App Router), React 19, TypeScript 5, Tailwind CSS 4 + Flowbite,
next-intl 4, Cornerstone3D 5. Local `useState` only, with no global state
library. More detail in [`frontend/CLAUDE.md`](frontend/CLAUDE.md).

| Route | File | Notes |
|---|---|---|
| `/` | `app/page.tsx` | Landing page |
| `/about` | `app/about/page.tsx` | What it measures, disclaimer |
| `/login` | `app/login/page.tsx` | Username/password, redirects to `?next=` |
| `/upload` | `app/upload/page.tsx` | Folder upload → `SeriesPicker` |
| `/examinations` | `app/examinations/page.tsx` | Server-rendered list |
| `/examinations/<accession>` | `app/examinations/[...slug]/page.tsx` | Viewer for `type: "torsion"`, series picker for `type: "pending"` |

Key components (`app/components/`):

- `series-picker.tsx`: series cards with previews, whole-leg vs. regions mode,
  POSTs `/upload/torsion/select`.
- `polling-component.tsx`: polls `/jobs/{id}`.
- `list-component.tsx` / `filter-component.tsx`: examination table, filters
  and actions.
- `torsion-examination-component.tsx`: owns the landmark state and the save,
  and lays out the details panel and the viewer.
- `cornerstone/`: the viewer. It shows one **axial StackViewport** of native
  slices rather than a 3-plane MPR, because the torsion MRI is highly
  anisotropic (≈0.6 mm in-plane, several mm between slices). It uses a custom
  `TorsionReferenceLine` annotation tool (subclass of `LengthTool`) and draws a
  per-slice derived labelmap for the segmentation overlay.

**Auth in the UI.** The login token is stored in the `morph_token` cookie (not
localStorage) so that server components can forward it during SSR
(`app/server-auth.ts`). Client calls go through `apiFetch` (`app/auth.ts`),
which attaches the token and redirects to `/login` on a 401. Media URLs get
`?token=`.

**i18n.** English and German, locale in the `NEXT_LOCALE` cookie (no locale in
the URL), catalogs in `frontend/messages/{en,de}.json`, which must keep
identical key trees.

**Commands.**

```bash
npm run dev:webpack   # development (use this one)
npm run build         # production build (webpack)
npm start             # serve the build
npm run lint
```

---

## Operations

- **Logs**: `MORPH_API_LOG_DIR` (default `api/logs/`), rotating, shared by API
  and worker. Container output is logged in full when a container exits
  non-zero, and per-measurement torsion errors are logged at ERROR.
- **Restart safety**: job status lives in the DB. A job that was `running` when
  the API restarted is marked `failed` ("Interrupted by restart"); re-dispatch it
  with `POST /model/torsion/{id}`.
- **Re-running**: `POST /model/segmentation/{id}` re-segments;
  `POST /model/torsion/{id}` recomputes torsion from the stored masks (full
  pipeline if the exam is still `unprocessed`). Recomputing torsion
  **overwrites the landmarks**, including any reader corrections.
- **Duplicates**: re-uploading an accession number is rejected unless
  `MORPH_API_ON_DUPLICATE=replace`. A pending (not yet selected) upload is
  always replaced.
- **Repair upside-down exams** ingested before orientation detection existed:
  `python scripts/repair_torsion_orientation.py <id> | --all [--dry-run]`.
  This re-splits from `original.nii.gz` and resets the exam to `unprocessed`.
- **Multi-slab exams ingested before slab detection** carry an inflated
  z-spacing and must be re-ingested from the original DICOM.
- **Legacy data**: `python scripts/migrate_pickles.py` imports old
  `api/data/*.pkl` exams.
- **Model updates**: rebuild and re-tag the docker image. The worker picks up
  the new image on its next job, but existing results are not recomputed.

---

## Testing

```bash
pytest api/tests          # API: docker, queue and storage mocked; no GPU/Redis needed
cd frontend && npm run lint
```

`api/tests` uses an in-process `EagerQueue` and a temporary DB/store. The
measurement code inside the torsion image is covered by the library test suite
(`pytest tests/`, see the root `CLAUDE.md`), including
`tests/test_compute_torsion_pipeline.py`, which mirrors the container
end-to-end. `api/tests.py` / `api/tests.ipynb` are manual smoke tests against a
running API. The frontend has no automated tests.

---

## Known limitations

- **Orthanc finalize needs an RQ scheduler.** `/upload/orthanc` schedules the
  finalize with `enqueue_in`, and RQ only runs scheduled jobs on a worker started
  *with* its scheduler. `api/tasks/worker.py` calls `Worker.work()` with the
  default `with_scheduler=False`, so as shipped the staged instances are never
  finalised. Until the entrypoint is changed, run the worker as
  `rq worker --with-scheduler -u $MORPH_API_REDIS_URL gpu` when using Orthanc.
- **A failed measurement reads as 0°.** NaN angles and NaN landmark
  coordinates are replaced by `0` for display (`api/schemas/docker_io.py::_sanitize`),
  and failed tibial landmarks are written as `[0, 0, 0]`. A value of exactly
  `0` should be treated as "not measured".
- **Murphy QC flags are discarded.** The torsion container emits
  `femoral_torsion_{left,right}_murphy_qc_flags` (degenerate femoral neck), but
  the API's result schema ignores extra keys, so the flags are not visible in
  the API or UI.
- **Stored angles are not updated by landmark edits** (see
  [Results & landmark format](#results--landmark-format)).
- **Non-square in-plane pixels are refused.** The torsion composition requires
  equal in-plane voxel spacing. A series with rectangular pixels produces an
  error for every measurement rather than a value.
- **One worker, one GPU.** Jobs run strictly one after another.
- **SQLite** is fine for a single host. For multiple API hosts, point
  `MORPH_API_DATABASE_URL` at a server database and share `storage_dir`.
- The `ExaminationType` / `ModelJob` fields of `filter_rules.json` are matched
  but not otherwise used: every matched instance is treated as a torsion exam.

---

## Where to look in the code

```
api/
  main.py              app factory, CORS, error handlers, router wiring
  settings.py          all MORPH_API_* settings
  deps.py              DI: DB session, store, queue, auth checks
  routers/             auth, examinations, uploads, jobs, volumes, health
  ingest/dicom.py      DICOM → NIfTI, slab detection, orientation, split, series staging
  ingest/orthanc.py    filter-rule matching
  tasks/               queue, dispatch, worker entrypoint, torsion job, docker wrappers, orthanc finalize
  db/                  models, engine, repository
  storage/store.py     file layout
  schemas/             request/response + container-output models
  auth/ users.py       tokens, passwords, user CLI
  orthanc_plugin.py    runs inside Orthanc
  docker/nnunet_torsion/  segmentation image
morphometry/
  Dockerfile, docker/compute_torsion.py   torsion image
frontend/
  app/server_config.ts   API URL resolution
  app/auth.ts            apiFetch, session cookie
  app/components/        list, upload/series picker, polling, viewer
  messages/              i18n catalogs
```
