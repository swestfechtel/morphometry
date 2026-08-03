# morphometry

A Python library for computing orthopaedic morphometric measurements from
segmentation masks of the lower extremity (hip, knee, ankle), for both MRI and
whole-leg CT, plus a FastAPI service and batch-processing scripts.

## Layout

- **`morphometry/`** — the analysis library.
  - **`morphometry/measurements/`** — all public `calculate_*` measurement
    functions, one submodule per region (`hip`, `femur`, `knee`, `tibia`,
    `ankle`, `whole_leg`, `cartilage`). This is the public surface:

    ```python
    from morphometry.measurements.femur import calculate_femoral_torsion
    from morphometry.measurements.hip import calculate_ccd, calculate_center_edge_angle
    ```
  - Region modules at the package root (`hip.py`, `femur.py`, `knee.py`,
    `tibia.py`, `ankle.py`, `whole_leg.py`, `cartilage/`) hold the landmark /
    reference-line helpers (`get_*`) the measurements build on.
  - Infrastructure: `image_io.py` (the `Image` / `Segmentation` wrapper),
    `geometry.py`, `constants.py`, `utils.py`, `bresenham.py`.
- **`api/`** — a FastAPI service that ingests DICOM uploads / Orthanc callbacks
  and dispatches inference + measurement jobs to docker images.
- **`scripts/`** — self-contained batch-processing scripts (hard-coded data
  paths; adapt before running elsewhere).
- **`frontend/`** — the Next.js 15 / React 19 web UI for the API (DICOM upload,
  job dispatch, and interactive landmark editing of results). The torsion viewer
  uses **Cornerstone3D** to render the NIfTI volumes directly (see "Running the
  stack" below).

Measurements operate on masks placed in **LPI** orientation; load with
`Segmentation(...).read_image(path)` then call `transform_coordinate_system()`.
The `side` argument everywhere refers to the **image** side, not the patient
side. See `CLAUDE.md` for the full architecture and conventions, and
`docs/reader_measurement_guide.md` for anatomic descriptions of each measurement.

## Tests

```bash
pytest -m "not needs_ct"          # fast characterization + unit tests
pytest                            # include CT golden tests (slow; needs CT data)
MORPH_UPDATE_GOLDEN=1 pytest      # (re)capture golden baselines
```

Tests skip cleanly when sample data is absent. Data paths can be overridden via
`MORPH_AUGSBURG_PA000001`, `MORPH_NAKO_SAMPLE_DIR`, `MORPH_CT_SAMPLE`.

## Running the stack

The full system is four cooperating processes: **Redis** (broker + durable job
state), the **RQ worker** (loads volumes, runs the docker model images — needs
the docker socket + the morphometry venv), the **FastAPI app**, and the
**Next.js UI**. The worker is the only process that touches docker; the API does
not.

All Python commands run from the repo root with the venv activated. Copy the
example env files first:

```bash
cp api/.env.example .env                       # MORPH_API_* settings
cp frontend/.env.example frontend/.env.local   # NEXT_PUBLIC_MODEL_API (optional)
```

### 1. Redis (via Docker)

There is no local `redis-server` binary; run the broker in a container. The
default `MORPH_API_REDIS_URL` is `redis://localhost:6379/0`, so publish 6379:

```bash
# first time: create and start the container
docker run -d --name morphometry-redis -p 6379:6379 redis:7

# afterwards: just start the existing container
docker start morphometry-redis

# verify it answers
docker exec morphometry-redis redis-cli ping     # -> PONG

# stop it when done (state persists in the container)
docker stop morphometry-redis
```

### 2. RQ worker

```bash
python -m api.tasks.worker      # one worker on the 'gpu' queue (serializes GPU jobs)
```

### 3. FastAPI app

```bash
# auth off (MORPH_API_API_KEYS unset) + allow the UI origin for the volume fetches
MORPH_API_CORS_ALLOW_ORIGINS=http://localhost:3000 \
  uvicorn api.main:app --host 0.0.0.0 --port 8000
```

`GET /health` is open; other endpoints require an `X-API-Key` header only when
`MORPH_API_API_KEYS` is set. The torsion volume endpoints
(`GET /examinations/{id}/volume/{image,mask}`) also accept the key as an
`?api_key=` query param so the Cornerstone NIfTI loader can fetch them.

### 4. Next.js UI

```bash
cd frontend
npm install
npm run dev:webpack    # dev server on http://localhost:3000 (NOT `npm run dev`)
```

Use `npm run dev:webpack`, **not** `npm run dev`: the Cornerstone torsion viewer
needs web workers / WASM wired up by the `webpack()` hook in `next.config.ts`,
which `next dev --turbopack` skips. `npm run build` always uses webpack, so
production builds are fine. Point the UI at a non-default API with
`NEXT_PUBLIC_MODEL_API` (inlined at build time — rebuild after changing it).
