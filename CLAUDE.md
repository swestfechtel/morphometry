# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository layout

The repo has four loosely-coupled parts; the three Python parts share a single virtualenv and the `morphometry` package, and `frontend/` is a self-contained Node/Next.js app:

- `morphometry/` — the analysis library, layered as an acyclic dependency graph:
  - **`morphometry/measurements/`** — all public `calculate_*` measurement functions, one submodule per region (`hip.py`, `femur.py`, `knee.py`, `tibia.py`, `ankle.py`, `whole_leg.py`, `cartilage.py`). This is the public surface (re-exported from `morphometry.measurements`). Import measurements from here, e.g. `from morphometry.measurements.femur import calculate_femoral_torsion`.
  - **Region modules** at the package root (`hip.py`, `knee.py`, `ankle.py`, `femur.py`, `tibia.py`, `whole_leg.py`, `cartilage/`) hold the landmark / reference-line helpers (`get_*`) that the measurements build on. They must NOT import from `measurements/`.
  - **Leaf infrastructure**: `image_io.py` (the `Image` / `Segmentation` wrapper every function operates through), `geometry.py` (shared pure helpers: `fold_to_acute`/`fold_to_obtuse`, `split_left_right`, `slice_centroid_to_point`, `mirror_sagittal_coordinate`), `constants.py` (modality-scoped `SegmentationLabels` + tunable thresholds), `utils.py`, `bresenham.py`. These import no region/measurement code.
  - Dependency direction (enforced by `tests/test_import_graph.py`): `measurements/* → region get_* → geometry/utils/constants/image_io/bresenham`.
  - MRI vs CT: torsion / CCD / neck-center / bone-length have separate `_ct` functions (modality differs in landmark acquisition); the acetabulum / CEA / subchondral / offset family take a `ct=` flag. Shared `_core` helpers isolate the modality differences.
  - Torsion `calculate_*` return only the angle; landmarks come from `get_femoral_torsion_landmarks` / `get_tibial_torsion_landmarks`. Measurement functions never change their return arity based on `plot` — pass a matplotlib `Axes` / PyVista `Plotter` to `plot=` to draw overlays.
- `api/` — a FastAPI service (`api/main.py`) that ingests DICOM uploads or Orthanc callbacks, stores examination metadata in SQLite + images as `.nii.gz` files (`api/db`, `api/storage`), and dispatches model jobs to a **Redis/RQ worker** (`api/tasks`) that shells out to the docker images (see below). Layered: `routers` → `ingest`/`serializers`/`deps` → `db`/`storage`/`tasks` → `settings`/`runtime`. Config is env-driven via `api/settings.py` (`MORPH_API_*` / `.env`). `api/examination.py` is legacy, kept only for `scripts/migrate_pickles.py`. Torsion uploads are **two-phase**: `POST /upload/torsion/series` enumerates the DICOM series in a whole examination folder (grouping by SeriesInstanceUID, discarding junk, rendering previews) into a `pending_selection` examination; the user picks a series (whole-leg → auto-split, or three hip/knee/ankle series) via `POST /upload/torsion/select`, which materializes it and auto-starts the full pipeline. A whole-leg series is frequently a **gapped 3-station acquisition** (hip/knee/ankle slabs); `api/ingest/dicom.py::_detect_slabs` splits it at the position gaps and converts each slab on its own so SimpleITK doesn't average the gaps into a ~2×-inflated z-spacing (which would silently corrupt all superior-inferior distances) — see `api/README.md` "Multi-slab acquisitions". **Auth** (`api/auth/`): username/password login (`POST /auth/login` → signed Bearer token, argon2 hashes) coexisting with `X-API-Key` for machine clients; users are managed with `python -m api.users`. See `api/README.md`.
- `scripts/` — one-off batch processing scripts (`process_augsburg_*.py`, `process_nako*.py`, `combine_series.py`, etc.). Each script is self-contained and typically prepends `sys.path.append('/home/simon/Work/morphometry')` and hard-codes absolute data paths (e.g. `/home/simon/Data/...`); update paths when running elsewhere. A few are proper `argparse` CLIs — e.g. `compute_knee_cartilage_thickness.py -i <seg.nii.gz ...|dir> [--femur-label 3 --tibia-label 4 --method knn -o summary.csv --dump-json maps.json --visualize both --screenshot <dir> --html <dir>]` computes per-subregion knee cartilage thickness and optionally renders the 3D segment/heatmap views (via `plot_knee_segments`/`plot_knee_thickness`) as PNGs (`--screenshot`) and/or self-contained interactive HTML (`--html`). `-i` accepts multiple files and/or directories (searched recursively for `*.nii`/`*.nii.gz`); multiple inputs batch into one combined CSV (leading `subject` column) + one combined JSON (keyed by subject). All visualisation files land flat in the `--screenshot`/`--html` dirs named `<subject>_knee_<view>.<ext>`. Output paths ending without an extension are treated as directories (`summary.csv`/`thickness.json` written inside). `analyse_thinnest_point_flexion.py -i <dir> [-o <dir> --screenshot/--html <dir> --visualize]` groups `K<subject>_<angle>_<load>.nii.gz` segmentations by `(subject, load)` and tests whether the thinnest point migrates posteriorly with flexion — measured as the thinnest point's arc-length fraction along the femoral articular surface (frame-invariant across the separately-acquired flexion scans), summarised per series (Spearman ρ + slope) and cohort-wide (binomial sign test).
- `frontend/` — the **Next.js 15 / React 19 / TypeScript** web UI for the API (Tailwind 4 + Flowbite, npm). It is the only client of the API and uses its exact contract: the `{examinations: [...]}` list envelope, base64 image/segmentation layers, and the `/examinations`, `/upload`, `/model/{segmentation,torsion}/{id}`, `/jobs/{id}` endpoints. App Router code lives under `frontend/app/`; the API base URL is `frontend/app/server_config.ts` (env `NEXT_PUBLIC_MODEL_API`, default `http://localhost:8000`). Node toolchain is independent of the Python venv. See `frontend/CLAUDE.md` for UI architecture.

## Core data flow

1. **Images** — every analysis function expects a `morphometry.image_io.Image` or `Segmentation`, which wraps either a `nibabel.Nifti1Image` or a `SimpleITK.Image`. Construct with `Image.from_nibabel(...)` / `Segmentation.from_nibabel(...)` or `Image('nibabel').read_image(path)`.
2. **Coordinate system** — masks must be placed in **LPI** orientation before measurement. Always call `seg.transform_coordinate_system()` right after loading; `remove_outliers()` is also commonly applied before measurement.
3. **Left/right splitting** — pipelines split a full-body mask at `array.shape[0] // 2` into `left_*` / `right_*` halves. Note: "left" / "right" refers to the *image* side, not the patient side. Functions take a `side='left'|'right'` argument and document this convention explicitly.
4. **Torsion landmarks** — `compute_torsion.py` now produces femoral landmarks nested by method: `landmarks['femur']['Lee'|'Murphy'][side]`. Tibia landmarks stay flat at `landmarks['tibia'][side]`. The REST layer (`api/serializers.py`, served via `api/routers/examinations.py`) handles both shapes.
5. **Whole-leg CT** — `morphometry/whole_leg.py` and the `*_ct` variants in `femur.py` / `hip.py` use `split_ct_image` (in `image_io.py`) to derive hip/knee/ankle sub-volumes from a single whole-leg segmentation with labels: femur=1, tibia=2, fibula=3, patella=5, hip=7 (defaults). `calculate_mikulicz_deviation` was renamed to `calculate_mechanical_axis_deviation`.

## Docker images

The API does not run inference in-process — the RQ worker invokes two external docker images built from this repo. Both are pulled/tagged as `swestfechtel/*:latest` and their tags are configured via `api/settings.py` (`MORPH_API_NNUNET_IMAGE` / `MORPH_API_TORSION_IMAGE`), used by `api/tasks/`.

- **`swestfechtel/nnunet_torsion:latest`** — built from `api/docker/nnunet_torsion/`. Runs nnUNetv2 predictions on hip/knee/ankle using the ResEncUNetXL plans (datasets 004, 006, 007) with a 5-fold ensemble. Requires CUDA (`--runtime=nvidia --gpus all`) and expects checkpoints under `api/docker/nnunet_torsion/checkpoints/{hip,knee,ankle}/` at build time. These checkpoints are gitignored; they must be present locally before `docker build`.
- **`swestfechtel/torsion:latest`** — built from `morphometry/Dockerfile` with entrypoint `morphometry/docker/compute_torsion.py`. Pure-Python torsion computation from segmentation masks; no GPU needed.

Rebuild commands (run from each Dockerfile's directory):

```bash
# nnUNet segmentation image
cd api/docker/nnunet_torsion && docker build -t swestfechtel/nnunet_torsion:latest .

# Torsion computation image (build context must be the morphometry/ package dir)
cd morphometry && docker build -t swestfechtel/torsion:latest .
```

## Running the API

The API and the model worker are separate processes sharing Redis, the SQLite DB,
and the storage dir. Copy `api/.env.example` to `api/.env` (or a repo-root `.env` —
both are loaded, `api/.env` wins) and adjust (`MORPH_API_*`: storage dir, redis/db
URLs, docker image tags, `API_KEYS`, `AUTH_REQUIRED`, `CORS_ALLOW_ORIGINS`).

```bash
# from repo root, with venv activated
redis-server                                   # broker + job state
python -m api.tasks.worker                      # RQ worker (one, on the 'gpu' queue → serializes GPU jobs)
uvicorn api.main:app --host 0.0.0.0 --port 8000 # the web app
```

The worker is the only process that loads volumes and runs docker (needs docker
socket access + the morphometry venv); the API process does not touch docker. Job
status is durable in the DB (`GET /jobs/{id}` survives restarts). Endpoints
require an `X-API-Key` header when `MORPH_API_API_KEYS` is set (`/health` is open).
Logs rotate in `api/logs/` (see `api/logging_config.py`). `api/orthanc_plugin.py`
runs **inside an Orthanc process** and forwards stored instances to
`MORPH_API_UPLOAD_URL` (default `http://localhost:8000/upload/orthanc`) with the
`MORPH_API_API_KEY` header. Migrate legacy `api/data/*.pkl` with
`python scripts/migrate_pickles.py`.

## Running the UI

The web client lives in `frontend/` and runs on its own Node toolchain (separate
from the Python venv):

```bash
cd frontend
npm install
npm run dev        # dev server on http://localhost:3000
```

Point it at the API with `NEXT_PUBLIC_MODEL_API` (default `http://localhost:8000`;
copy `frontend/.env.example` to `frontend/.env.local` to override — `NEXT_PUBLIC_*`
is inlined at build time, so rebuild after changing). For local dev, allow the UI
origin in the API's CORS: `MORPH_API_CORS_ALLOW_ORIGINS=http://localhost:3000`.
For an open dev server leave `MORPH_API_API_KEYS` unset and `MORPH_API_AUTH_REQUIRED`
false (the UI then works without signing in). To require login, set
`MORPH_API_AUTH_REQUIRED=true`, create a user with `python -m api.users create <name>`,
and sign in at the UI's `/login`.

## Tests

A pytest harness lives in `tests/` (config in `pyproject.toml`; the runtime env is still pinned by `requirements.txt`). It is primarily **characterization / golden tests** that lock the numeric output of the measurement functions so refactors can be proven not to change results:

- `tests/golden/*.json` hold committed reference values. Compare with `pytest`; (re)capture with `MORPH_UPDATE_GOLDEN=1 pytest`. A golden delta must be intentional and re-captured deliberately.
- Data-locating fixtures in `tests/conftest.py` **skip** when sample data is absent. Markers: `needs_augsburg`, `needs_nako`, `needs_ct`, `stochastic`. Override data paths with `MORPH_AUGSBURG_PA000001` / `MORPH_NAKO_SAMPLE_DIR` / `MORPH_CT_SAMPLE`. CT tests read large volumes over sshfs (~minutes) — run `pytest -m "not needs_ct"` for a fast loop.
- `tests/test_compute_torsion_pipeline.py` mirrors `morphometry/docker/compute_torsion.py` end-to-end (the production oracle); `tests/test_import_graph.py` enforces the package dependency DAG; `tests/unit/test_geometry.py` covers the shared geometry helpers.

`api/tests.py` remains a separate collection of manual HTTP smoke tests against a running API (it assumes a specific local data layout); `api/tests.ipynb` and `notebooks/augsburg_test.ipynb` serve the same exploratory purpose.

## Dependencies

`requirements.txt` pins the full environment (Python 3.10, `torch==2.5.1`, `nnunetv2==2.5.1`, `numpy==1.26.4`, `SimpleITK`, `nibabel`, `pyvista`, `pydicom`, `fastapi`, `TotalSegmentator`, `pingouin`, etc.). When bumping `torch` or `nnunetv2`, also update the pins inside `api/docker/nnunet_torsion/Dockerfile` — they are installed independently in the image.

## Conventions worth preserving

- Docstrings are verbose: one-line purpose, `:param:` / `:return:` for each parameter. New public functions should follow this style (see e.g. `morphometry/measurements/hip.py::calculate_ccd`).
- Measurement functions commonly accept a `plot: bool | plt.Axes` argument — passing an `Axes` draws overlays instead of creating a new figure. Preserve this pattern when adding new measurements.
- `Tibia`/`Femur` clean their extracted cartilage mask before any processing (`_remove_mask_outliers`, `outlier_ratio=0.1` ctor arg): per-label 26-connectivity components smaller than the ratio of the total cartilage volume are dropped, removing mislabelled blobs / detached clusters that otherwise corrupt the Delaunay meshes. Legit cartilage (femoral horseshoe, the two tibial plateaus) is far larger than the threshold and retained; pass `outlier_ratio=0` to disable. This benefits every consumer (thickness, thinnest-area, the scripts, the API).
- `find_thinnest_area` (re-exported; wrapper `calculate_thinnest_cartilage_area`) locates the thinnest *area* of the tibiofemoral cartilage robustly: it sums the tibial+femoral thickness maps over their overlapping axial `(x,y)` coords, rasterises to a dense grid, NaN-aware median-smooths (kernel `k` derived from `neighbourhood_mm`/in-plane spacing, forced odd) to reject single-voxel outliers, argmins for the neighbourhood centre, then refines to the actual thinnest voxel within a radius disc. Returns `{point, center, center_thickness, combined_thickness, tibial_thickness, femoral_thickness, kernel_size}` (2D voxel indices — z is discarded by the thickness maps; `center_thickness` is the robust smoothed value, `combined_thickness` the raw value at `point`). `plot_thinnest_point` (re-exported; also `calculate_thinnest_cartilage_area(..., plot=pv.Plotter)`) renders the cartilage in 3D (`base='thickness'` heatmap or `base='segments'` voxels) and points a red arrow at the thinnest point — its 3D depth is recovered as the femur/tibia contact-surface midpoint at that `(x,y)` column (`_thinnest_point_physical`).
- Femoral central weight-bearing zone (CWBZ) **and posterior-zone** extraction restrict to the tibia-facing superior-inferior cluster (`_restrict_to_contact_cluster`): the A-P/L-R window from the tibial plateau alone would, at high knee flexion, also capture the anterior trochlea folded into the same A-P range, so only the S-I cluster nearest the tibial contact (CWBZ: tibial-contact z; posterior zone: CWBZ mean z) is kept. The split uses a spacing-scaled **absolute** S-I gap (`Femur._si_gap()` ≈ 1.5 mm in voxels), not a span-relative one — a distant fold inflates the span and would push a relative threshold above the real (small) fold-vs-contact valley. The anterior zone is left unrestricted (it genuinely is the trochlea). Preserve this when touching `extract_central_weightbearing_zone` / `extract_anterior_posterior_zones`.
- Knee cartilage thickness has 3D visualisation (`morphometry/cartilage/knee.py`): `Tibia`/`Femur` each expose `plot_segments()` (raw subregion voxels as distinctly-coloured cube glyphs — the `SEGMENT_COLORS`/`*_SUBREGIONS` tables) and `plot_thickness(thicknesses)` (articular surface as a thickness heatmap with a shared colour bar). `plot_knee_segments`/`plot_knee_thickness` render both bones in one PyVista scene (re-exported from `measurements.cartilage`); the `calculate_*_cartilage_thickness` wrappers take `plot: bool | pv.Plotter` to draw the heatmap. All accept an optional `plotter=` and only `.show()` when they created it. Femoral posterior zones reconstruct their surface with a y/z swap to match how `calculate_thickness` extracted them.
- Failures inside batch scripts and `compute_torsion.py` are caught per-measurement (`except (RuntimeError, AssertionError, ValueError)`) and written out as `np.nan` so a single bad patient does not abort a cohort run. Keep this behavior when extending those scripts.