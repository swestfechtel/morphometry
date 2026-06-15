# Morphometry API

FastAPI service for torsion/x-ray examinations: ingests DICOM (UI uploads or
Orthanc), runs the nnUNet segmentation + torsion docker pipeline as background
jobs, and serves results.

## Architecture

```
routers/        HTTP endpoints (examinations, uploads, jobs, health)
deps.py         DI providers (settings, DB session, store, queue, API-key auth)
ingest/         DICOM → .nii.gz + Examination row (single, multi-series, orthanc, x-ray)
serializers.py  DB rows → response schemas
schemas/        Pydantic request/response + docker-output validation models
db/             SQLModel tables (Examination, Job), WAL SQLite engine, repository
storage/        per-examination .nii.gz files + encoded-PNG cache + orthanc staging
tasks/          RQ queue, worker entrypoint, torsion job, orthanc finalize
domain/encode.py  slice → base64 PNG rendering (worker-side)
settings.py     env-driven config (MORPH_API_*)
runtime.py      cached settings/engine/store shared by API + worker
main.py         app factory + lifespan
```

Metadata/status/results live in **SQLite**; large images are **`.nii.gz` files**
on disk (never in the DB or RAM cache). Long jobs run in a **separate RQ worker**
(durable status, GPU serialized by a single worker); the API process never blocks
on docker.

## Run

```bash
cp api/.env.example .env        # set storage dir, redis/db URLs, image tags, API_KEYS, CORS
redis-server
python -m api.tasks.worker      # one worker on the 'gpu' queue
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Auth: send `X-API-Key` when `MORPH_API_API_KEYS` is set (`/health` is exempt).

## Test

```bash
pytest api/tests        # docker, queue, storage all mocked — no GPU/Redis needed
```

## Migrate legacy data

```bash
python scripts/migrate_pickles.py   # api/data/*.pkl → SQLite + .nii.gz files
```
