# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## How to read this file

`CLAUDE.md` holds what is **true now**: the layout, the contracts, the shipped constructions, the
decisions, and — most load-bearing of all — the **do-not-pool warnings** that say which rows of
which cohort table may no longer be compared with which. The measurement evidence *behind* those
decisions (cohort surveys, probe runs, rejected constructions, blast-radius tables, and the
long-form paragraphs this file used to carry inline) sits in `docs/evidence/`, moved there
**verbatim**. Where a bullet rests on such a trail it links to it; nothing was summarised away
there. **Before changing a landmark or measurement, read its evidence file** — the summary here says
*what* is true and *what must not be pooled*; the file says by how much and why.

| evidence file | what it holds |
|---|---|
| [`femoral_head_slab.md`](docs/evidence/femoral_head_slab.md) | the ~1.2 mm medial slab drag, the six arbiter routes closed by measurement, the four rejected constructions, the CT/MRI blast-radius tables, the `zscore` guard's removal |
| [`alpha_angle.md`](docs/evidence/alpha_angle.md) | the reslice invariance probes, the 3000-case NaKo survey, the radial-position sweep built and withdrawn |
| [`condylar_line.md`](docs/evidence/condylar_line.md) | the no-notch fallback's prevalence and cues, the NMDID cohort re-measurement and its fold-seam tail, the Augsburg line-length survey |
| [`lee_neck_axis.md`](docs/evidence/lee_neck_axis.md) | the two-rotation defect, the padding exposure, the three crop residuals closed one at a time |
| [`qc_gates.md`](docs/evidence/qc_gates.md) | the cohort surveys behind all three QC gates and their non-mutating verification |
| [`in_plane_spacing.md`](docs/evidence/in_plane_spacing.md) | the four-cohort spacing surveys behind `validate_in_plane_isotropy` |
| [`torsion_image_rebuilds.md`](docs/evidence/torsion_image_rebuilds.md) | every rebuild of the torsion image, what moved, how it was verified |
| [`golden_recaptures.md`](docs/evidence/golden_recaptures.md) | which golden moved when, and what moved it |
| [`ccd_neck_shell_x_ratio.md`](docs/evidence/ccd_neck_shell_x_ratio.md) | why `x_ratio` is a conditioning parameter and why the millimetre reformulation is worse |
| [`cartilage_cohort_figures.md`](docs/evidence/cartilage_cohort_figures.md) | the `dfg_knees` subregion, thickness and flexion-reproducibility figures |
| [`orientation_and_splitting.md`](docs/evidence/orientation_and_splitting.md) | LPI reorientation and its failure cases, the left/right split guarantee |
| [`mirror_equivariance.md`](docs/evidence/mirror_equivariance.md) | how femoral torsion, CCD, tibial torsion, the head centre, CEA, shaft axis and offset were made mirror-equivariant, with every before/after figure |
| [`axial_angle_composition.md`](docs/evidence/axial_angle_composition.md) | the signed femoral-torsion and knee-rotation compositions, the TFR sign convention and its literature basis |
| [`hip_landmarks.md`](docs/evidence/hip_landmarks.md) | subchondral cone axis, MRI head consensus fit and `x_ratio`, Murphy's distal landmark, three repaired broken paths, `get_contour`'s border artefact |
| [`knee_cartilage.md`](docs/evidence/knee_cartilage.md) | the full Wirth & Eckstein subregion, thickness, surface and deformation-map conventions |
| [`pipeline_conventions.md`](docs/evidence/pipeline_conventions.md) | data-layout traps, test-fixture behaviour, the per-script exception tuples |
| [`docs/scripts.md`](docs/scripts.md) | full CLI / output reference of the batch scripts (not evidence — reference) |

Four habits this file keeps, and that are worth keeping:

- **A correction stays visible.** Where a claim here was once wrong, both the wrong version and the
  correction are recorded, because the wrong one is what a reader would otherwise re-derive.
- **A figure that must not be quoted bare says so in the sentence that gives it** — a fold-seam sd,
  a maximum that is one artefactual leg, a p-value that is consistent rather than large.
- **A decision records what would reopen it**, not only what was decided.
- **A parameter is never chosen against the data it will then be quoted against.** Where that was
  done anyway (`CEA_UPPER_MARGIN_FACTOR`), it is named as a mistake and cited as the precedent.

**Keep this file short.** It must stay well under Claude Code's 150k-character limit (it was cut from
247k to its current size on 2026-11-12). New measurement evidence, before/after tables and long
do-not-pool paragraphs go into `docs/evidence/`; this file gets a one- or two-sentence summary and a link.

## Open items

Everything else here is shipped, closed by measurement, or a deliberate non-action. Three items are
genuinely open, and each is blocked on data or people that do not exist in this repo — not on work:

1. **The CT femoral-head-radius arbiter.** 17 of 92 verification halves still fit a head radius
   outside the 18–26 mm adult range, and six candidate arbiters are closed by measurement, so
   `C.FEMORAL_HEAD_SLAB_RADII` stays at 2.0. Reopening it needs **cadaveric ground truth, or a
   cohort whose acetabulum is labelled separately** the way NaKo's is — not another cue derived from
   this image family. Evidence: [`femoral_head_slab.md`](docs/evidence/femoral_head_slab.md).
2. **Asking reader Jonathan which convention the angles were measured in.** The recorded signs track
   an image-frame convention, which is a hypothesis these tables cannot settle; an answer would also
   sharpen the Lee/Murphy reader gap (task #2). Needs the reader, so it cannot be closed from here.
   Evidence: [`condylar_line.md`](docs/evidence/condylar_line.md).
3. **The non-LPI orientation guard for the torsion pipelines.** Closed as *unreachable* rather than
   done: 7745 files across four cohorts reorient to LPI with no exception, and the only non-LPI
   scans known anywhere are three `dfg_knees` cartilage scans no torsion pipeline can reach. Worth
   adding the day a cohort arrives that is not uniformly one raw orientation, and the spacing survey
   is the thing to re-run then. Evidence: [`in_plane_spacing.md`](docs/evidence/in_plane_spacing.md).

## Repository layout

The repo has four loosely-coupled parts; the three Python parts share a single virtualenv and the `morphometry` package, and `frontend/` is a self-contained Node/Next.js app:

- `morphometry/` — the analysis library, layered as an acyclic dependency graph:
  - **`morphometry/measurements/`** — all public `calculate_*` measurement functions, one submodule per region (`hip.py`, `femur.py`, `knee.py`, `tibia.py`, `ankle.py`, `whole_leg.py`, `cartilage.py`). This is the public surface (re-exported from `morphometry.measurements`). Import measurements from here, e.g. `from morphometry.measurements.femur import calculate_femoral_torsion`.
  - **Region modules** at the package root (`hip.py`, `knee.py`, `ankle.py`, `femur.py`, `tibia.py`, `whole_leg.py`, `cartilage/`) hold the landmark / reference-line helpers (`get_*`) that the measurements build on. They must NOT import from `measurements/`.
  - **Leaf infrastructure**: `image_io.py` (the `Image` / `Segmentation` wrapper every function operates through), `geometry.py` (shared pure helpers: `fold_to_acute`/`fold_to_obtuse`, `split_left_right`, `slice_centroid_to_point`, `mirror_sagittal_coordinate`, `signed_axial_angle`, `validate_in_plane_isotropy`, `mirror_index_x`, `order_condylar_points`), `constants.py` (modality-scoped `SegmentationLabels` + tunable thresholds), `utils.py`, `bresenham.py`. These import no region/measurement code.
  - Dependency direction (enforced by `tests/test_import_graph.py`): `measurements/* → region get_* → geometry/utils/constants/image_io/bresenham`.
  - MRI vs CT: torsion / CCD / neck-center / bone-length have separate `_ct` functions (modality differs in landmark acquisition); the acetabulum / CEA / subchondral / offset family take a `ct=` flag. Shared `_core` helpers isolate the modality differences.
  - **Whole hip bone vs acetabulum**: `calculate_center_edge_angle` / `calculate_acetabular_anteversion` were written for masks that label the acetabulum separately (MRI/NaKo, isotropic). Whole-leg CT label maps have one label for the entire hip bone, so pass `acetabulum_radius_factor` (~1.4) to route them through `get_acetabular_surface` (in `hip.py`), which keeps only the bone *facing* the femoral head — per ray out of the head centre, the first bone voxel — so the iliac wing is shadowed rather than mistaken for the acetabular rim. That flag also switches CEA's angle into millimetres: the default index-space arithmetic is only valid for isotropic voxels, and on 0.8×0.8×3.0 mm CT it inflates a 25° angle to ~60°. Default `0` keeps the legacy behaviour, so the MRI goldens are untouched.
  - Torsion `calculate_*` return only the angle; landmarks come from `get_femoral_torsion_landmarks` / `get_tibial_torsion_landmarks`. Measurement functions never change their return arity based on `plot` — pass a matplotlib `Axes` / PyVista `Plotter` to `plot=` to draw overlays.
- `api/` — a FastAPI service (`api/main.py`) that ingests DICOM uploads or Orthanc callbacks, stores examination metadata in SQLite + images as `.nii.gz` files (`api/db`, `api/storage`), and dispatches model jobs to a **Redis/RQ worker** (`api/tasks`) that shells out to the docker images (see below). Layered: `routers` → `ingest`/`serializers`/`deps` → `db`/`storage`/`tasks` → `settings`/`runtime`. Config is env-driven via `api/settings.py` (`MORPH_API_*` / `.env`). `api/examination.py` is legacy, kept only for `scripts/migrate_pickles.py`. Torsion uploads are **two-phase**: `POST /upload/torsion/series` enumerates the DICOM series in a whole examination folder (grouping by SeriesInstanceUID, discarding junk, rendering previews) into a `pending_selection` examination; the user picks a series (whole-leg → auto-split, or three hip/knee/ankle series) via `POST /upload/torsion/select`, which materializes it and auto-starts the full pipeline. A whole-leg series is frequently a **gapped 3-station acquisition** (hip/knee/ankle slabs); `api/ingest/dicom.py::_detect_slabs` splits it at the position gaps and converts each slab on its own so SimpleITK doesn't average the gaps into a ~2×-inflated z-spacing (which would silently corrupt all superior-inferior distances) — see `api/README.md` "Multi-slab acquisitions". **Auth** (`api/auth/`): username/password login (`POST /auth/login` → signed Bearer token, argon2 hashes) coexisting with `X-API-Key` for machine clients; users are managed with `python -m api.users`. See `api/README.md`.
- `scripts/` — one-off batch processing scripts (`process_augsburg_*.py`, `process_nako*.py`, `process_nmdid.py`, `combine_series.py`, …). Each is self-contained, typically prepends `sys.path.append('/home/simon/Work/morphometry')` and hard-codes absolute data paths (`/home/simon/Data/...`). The cohort runners write a trailing `qc_flags` column (the QC gates below; Side in the Augsburg/NaKo tables is the *patient* side, so the image-left half lands on 'right'). **The full CLI and output reference — `process_nmdid.py` (SLURM sharding, `clean_leg`), `process_nako_large.py` (chunks, resume, suspects, `--merge`), the knee-cartilage scripts (`compute_knee_cartilage_thickness.py`, `analyse_thinnest_point_flexion.py`, `report_knee_cartilage_cohort.py`, `analyse_cartilage_deformation.py`) and the analysis/survey scripts — is in [`docs/scripts.md`](docs/scripts.md).** Points worth knowing without opening it:
  - The knee-cartilage scripts have **no `--method`** (passing one fails loudly). `analyse_cartilage_deformation.py` reads the **current** 113-scan `dfg_knees` label set (`/home/simon/Downloads/dfg_knees_segmentations/dfg_knees/segmentations`); the knee goldens and `report_knee_cartilage_cohort.py`'s quoted tables still use the older set under `/home/simon/Data/dfg_knees/monai_segmentations/labels/` — never mix them; re-pinning on the current set is open follow-up work.
  - **Every earlier `analyse_thinnest_point_flexion.py` run is superseded**, and its per-compartment sign tests are not robust to the search restrictions — quote them only with that caveat.
  - `analyse_reader_bias.py` → `docs/augsburg_reader_bias.md`, which carries a **2026-11-04 partial retraction** (task #2: the Lee−Murphy separation is anatomy). `analyse_ccd_x_ratio.py` → `docs/ccd_x_ratio.md`; it deliberately picks no parameter. `survey_ct_spacing.py` / `survey_ct_murphy_qc.py` survey the whole-leg CT cohort (→ `docs/ct_murphy_qc.md`).
  - `process_nako_large.sbatch` refuses a dirty checkout; the checkout must not change until the array finishes. Cohort is 30,407 cases.
- `frontend/` — the **Next.js 15 / React 19 / TypeScript** web UI for the API (Tailwind 4 + Flowbite, npm). It is the only client of the API and uses its exact contract: the `{examinations: [...]}` list envelope, base64 image/segmentation layers, and the `/examinations`, `/upload`, `/model/{segmentation,torsion}/{id}`, `/jobs/{id}` endpoints. App Router code lives under `frontend/app/`; the API base URL is `frontend/app/server_config.ts` (env `NEXT_PUBLIC_MODEL_API`, default `http://localhost:8000`). Node toolchain is independent of the Python venv. See `frontend/CLAUDE.md` for UI architecture. `README_WEBAPP.md` is the standalone user/operator guide to the API + frontend together (workflow, endpoint reference, config tables, known limitations); keep it in sync when endpoints, settings or the pipeline change.

## Core data flow

1. **Images** — every analysis function expects a `morphometry.image_io.Image` or `Segmentation`, which wraps either a `nibabel.Nifti1Image` or a `SimpleITK.Image`. Construct with `Image.from_nibabel(...)` / `Segmentation.from_nibabel(...)` or `Image('nibabel').read_image(path)`.
2. **Coordinate system** — masks must be placed in **LPI** before measurement: call `seg.transform_coordinate_system()` right after loading (`remove_outliers()` is also commonly applied). The reorientation only permutes/flips axes, so a strongly oblique scan can come out non-LPI; `Image.achieved_axcodes` / `Image.orientation_ok` record it (with one WARNING), behaviour unchanged. Known cases: 3 of 96 `dfg_knees` scans (K7/K8/K11_90_belastet), attributed by the user to a scanner misconfiguration and flagged in the knee scripts' `qc_flags`, not fixed. K11_90_belastet is additionally a broken 1 mm isotropic scan. `orientation_ok` is only meaningful on the reoriented image: a `mirror_image_left_right` copy reads `False`. Details: [`orientation_and_splitting.md`](docs/evidence/orientation_and_splitting.md).
3. **Left/right splitting** — pipelines split at `array.shape[0] // 2` into `left_*` / `right_*`; "left"/"right" is the **image** side, not the patient side, and functions taking `side=` document this. Whenever a half is used with geometry, split with `image_io.split_image_left_right` — never wrap `array[half:]` in the parent's affine (that put every right-side physical coordinate `half * spacing_x` off). Left half keeps the parent geometry; right-half index `i` ↔ parent `i + [half, 0, 0]`, so index-space landmarks from a right half still need `+ half` on x. `image_io.crop_image` / `split_ct_image` / the API's `_split_volume` share the same affine-advance (`_crop_axis`). Two flips, not interchangeable: `mirror_image_left_right` relabels indices (physical positions kept), `reflect_image_left_right` reverses voxels and keeps geometry (the physical *mirror twin*). `geometry.split_left_right` is the geometry-free array split. Details: [`orientation_and_splitting.md`](docs/evidence/orientation_and_splitting.md).

   **Mirror equivariance — the rule.** Femoral torsion (MRI Lee/Murphy + CT), the MRI and CT CCD, tibial torsion, the knee rotation angle, the femoral head centre (MRI both paths + CT), the NaKo CEA and the MRI shaft axis / NaKo femoral offset all measure a **right image half as the reflection (or, for the MRI CCD, the mirror) of a left one**, run left-side code, and map landmarks back with `geometry.mirror_index_x` / `geometry.order_condylar_points` (CT head centre: `hip.py::_mirror_physical_point_x`). **Don't reintroduce `side == 'right'` branches into landmark code — add left-side logic and let the reflection handle the right.** The MRI CCD needs `mirror_image_left_right` (it spans two images with different origins); the CT landmarks need `reflect_image_left_right` (they read physical directions). Left-side values were bit-identical through all of this. Equivariance is consistency, not accuracy. Pinned by the `tests/characterization/test_*_equivariance.py` family. **Do not pool right-side rows produced before and after these changes** (CT CCD, tibial torsion, CT head-centre consumers, NaKo CEA, femoral offset, CCD) — and the two both-sides measurements (CEA, acetabular version) move on their *left* element too, because they use both head centres. `CEA_UPPER_MARGIN_FACTOR` = 1.1 was **chosen against the 60 NaKo reader cases** (calibration on the evaluation set — the precedent named above); do not pool NaKo CE rows at 1.5 with rows at 1.1. On the non-isotropic path `calculate_femoral_offset` still calls `get_femoral_head_center(side='right')` directly and is not equivariant. Figures: [`mirror_equivariance.md`](docs/evidence/mirror_equivariance.md).

   **Signed axial angles.** `geometry.signed_axial_angle` composes femoral torsion (`measurements/femur.py::_femoral_torsion_angle`: posterior condylar line → neck axis; **positive = antetorsion**, negative = retrotorsion), tibial torsion (`measurements/tibia.py::_tibial_torsion_angle`: posterior tibial condylar line → tibia→fibula centroid line; **positive = external**) and the knee rotation angle (`measurements/knee.py::_knee_rotation_angle`: femoral → tibial posterior condylar line; this is the literature's **tibiofemoral rotation (TFR)** angle, positive = tibia externally rotated, `KNEE_ROTATION_POSITIVE_SIGN = +1` per Anzillotti et al. 2026 / Diederichs et al. 2013 — only the sign carries over from the literature, never the magnitude). All fold into (−90, 90], are rotation-invariant, and `side='right'` negates x components. The former fold-to-acute compositions flipped sign on image-frame coordinate signs. Pinned by `tests/unit/test_{femoral_torsion,tibial_torsion,knee_rotation}_composition.py`. Flipping `KNEE_ROTATION_POSITIVE_SIGN` is one line plus re-capturing `knee_rotation_angle_left`. Details: [`axial_angle_composition.md`](docs/evidence/axial_angle_composition.md).

   **Posterior condylar line** (`knee.py::get_knee_reference_line`). When `find_notch` fails the divider is the section's centre of mass (`get_condylar_divider`) and the frame is left alone (LPI gives posterior); the old `rotate_tibia` fallback flipped the frame ~180° and read the anterior cortex — removed. The endpoints are found by **rotating coordinates about the `int16` divider, never the image** (`get_condylar_tangent_points`), which makes the line exactly translation-invariant (200/200 under pad/crop) and puts both endpoints on actual mask voxels; `thresh`/`step_size` are obsolete and raise on non-default values. Endpoint order is canonicalised by `geometry.order_condylar_points` at all call sites. **Do not pool** femoral/tibial torsion or knee-rotation rows across task #4 (fallback) or task #5 (read-out, 2026-11-04; median ~0.5–0.9° moves on Augsburg, CT goldens moved). The mirror *bearing* asymmetry is now ~0.33° median but the line's *length* differs far more between direct and reflected readings, so anything thresholding length must measure in the consumer's frame. On NMDID CT the no-notch fallback fires on ~62% of legs; tibial torsion there has sd ~17° (≈2.5× published) and poor within-case L/R agreement — don't quote it as validation, and don't quote raw maxima (fold-seam artefacts). Details: [`condylar_line.md`](docs/evidence/condylar_line.md).

   **In-plane isotropy (task #10).** All axial-plane angles are composed in **index space**, valid only for equal in-plane spacing. `geometry.validate_in_plane_isotropy(spacing, source)` raises `ValueError` (only ever `ValueError`, so batch handlers catch it) when `|sx − sy| > C.IN_PLANE_SPACING_RTOL (1e-3) × max`. Measurements check wherever the spacing is known; entry points that took bare arrays now accept an `Image` or an array (`image_io.mask_and_spacing`), with explicit `spacing=` winning. No warn-when-absent, by design. `dfg_knees` violates the precondition (0.75 × 0.3125 mm in plane after LPI) and is refused rather than physically re-derived (a flexed knee has no canonical axial plane). Same class, unfixed and **unreachable from production**: `calculate_joint_line_convergence_angle` (no script caller) and `calculate_pma_angle` (dead import; its plafond line is a dummy `[0,0,0]→[0,0,1]`, so its golden ~89.7° is by construction). Details: [`in_plane_spacing.md`](docs/evidence/in_plane_spacing.md).
4. **Torsion landmarks** — `compute_torsion.py` produces femoral landmarks nested by method: `landmarks['femur']['Lee'|'Murphy'][side]`; tibia landmarks stay flat at `landmarks['tibia'][side]`. The REST layer (`api/serializers.py`, via `api/routers/examinations.py`) handles both shapes. `results.json` also carries eight QC-flag strings (Murphy ×2, Lee ×2, condylar ×4); as of 2026-11-06 the API **keeps** them — declared on `api/schemas/docker_io.py::ResultsModel` and `api/schemas/examination.py::TorsionValues`, stored in the JSON column `Examination.torsion_values` (no DB change), surfaced on the examination endpoints (not `GET /jobs/{id}`). They default to `""`, so "QC didn't run" and "QC found nothing" are indistinguishable. Container keys are **patient** side, the library's image side. Pinned by `tests/unit/test_torsion_qc_flags_schema.py`. Details: [`qc_gates.md`](docs/evidence/qc_gates.md).
5. **Whole-leg CT** — `morphometry/whole_leg.py` and the `*_ct` variants in `femur.py` / `hip.py` use `split_ct_image` (in `image_io.py`) to derive hip/knee/ankle sub-volumes from a single whole-leg segmentation with labels: femur=1, tibia=2, fibula=3, patella=5, hip=7 (defaults). `calculate_mikulicz_deviation` was renamed to `calculate_mechanical_axis_deviation`.

## Docker images

The API does not run inference in-process — the RQ worker invokes two external docker images built from this repo. Both are pulled/tagged as `swestfechtel/*:latest` and their tags are configured via `api/settings.py` (`MORPH_API_NNUNET_IMAGE` / `MORPH_API_TORSION_IMAGE`), used by `api/tasks/`.

- **`swestfechtel/nnunet_torsion:latest`** — built from `api/docker/nnunet_torsion/`. Runs nnUNetv2 predictions on hip/knee/ankle using the ResEncUNetXL plans (datasets 004, 006, 007) with a 5-fold ensemble. Requires CUDA (`--runtime=nvidia --gpus all`) and expects checkpoints under `api/docker/nnunet_torsion/checkpoints/{hip,knee,ankle}/` at build time. These checkpoints are gitignored; they must be present locally before `docker build`.
- **`swestfechtel/torsion:latest`** — built from `morphometry/Dockerfile` with entrypoint `morphometry/docker/compute_torsion.py`. Pure-Python torsion computation from masks; no GPU. **It carries a copy of the library, so it must be rebuilt after every landmark / measurement change in `morphometry/`** — otherwise the API silently serves the numbers of whatever commit it was built from. Preserve the outgoing image as `swestfechtel/torsion:pre-<sha>` when rebuilding. **The recorded sha is not evidence of what an image contains — run it, or hash its `/app/morphometry` against HEAD.** Current: rebuilt at `2d1fc91` (2026-11-12), all 33 `.py` files identical to HEAD, pushed as `swestfechtel/torsion@sha256:bb619140abec79ceb9f22a5393884ea78105c18cf0014cf7fe2d4f7b6eabe9eb`. Base pinned to `python:3.10.16-bookworm` (the venv's interpreter; pinned by tag, not digest); deps from `morphometry/requirements.txt` installed `--no-deps` + `pip check`; system libs `libgl1 libxrender1` only (needed because importing the measurements pulls in `pyvista` → `vtk` → `libGL`; nothing renders). Discriminating fixtures: PA000001 and PA001127 (the latter has non-empty Murphy flags). Rebuild records and why the pins matter numerically: [`torsion_image_rebuilds.md`](docs/evidence/torsion_image_rebuilds.md).

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

- `tests/golden/*.json` hold committed reference values. Compare with `pytest`; (re)capture with `MORPH_UPDATE_GOLDEN=1 pytest`. A golden delta must be intentional and re-captured deliberately. **The re-capture log — which golden moved when, and what moved it — is in [`docs/evidence/golden_recaptures.md`](docs/evidence/golden_recaptures.md).**
  - The 14 knee cartilage goldens (`tests/characterization/test_knee_cartilage.py`, marker `needs_dfg_knees`) pin `K1_0_unbelastet` fully and `K1_90_unbelastet` partly, on the **older** `dfg_knees` label set. `knee_rotation_angle_left` must be re-captured if `KNEE_ROTATION_POSITIVE_SIGN` is ever flipped. Details: [`golden_recaptures.md`](docs/evidence/golden_recaptures.md).
  - `MORPH_UPDATE_GOLDEN=1` rewrites **every golden it evaluates** — re-capture by running only the named failing node ids.
- **Data-layout traps that silently change numbers.** In `reader_study_augsburg/<patient>/` only `<region>_seg_test.nii.gz` is production (the identically-named `<region>_seg.nii.gz` is stale); in `preprocessed/`, `PA000031/` also holds `*_seg_clean.nii.gz`, so a `knee_seg*` glob measures it twice. Always take paths from `tests/data_paths.py` or the script, and match production filenames exactly, never a prefix. Details: [`pipeline_conventions.md`](docs/evidence/pipeline_conventions.md).
- Data-locating fixtures in `tests/conftest.py` **skip** when sample data is absent (an unreachable sshfs mount counts as absent). Markers: `needs_augsburg`, `needs_augsburg_reader_study`, `needs_nako`, `needs_ct`, `needs_dfg_knees`, `stochastic`. Overrides: `MORPH_AUGSBURG_PA000001` / `MORPH_AUGSBURG_READER_STUDY` / `MORPH_NAKO_SAMPLE_DIR` / `MORPH_CT_SAMPLE` / `MORPH_DFG_KNEES_DIR` (older set, knee goldens) / `MORPH_DFG_KNEES_CURRENT_DIR` (current 113-scan set, only `test_cartilage_deformation.py`). CT tests read over sshfs (~minutes) — `pytest -m "not needs_ct"` for a fast loop. The `dfg_knees` fixtures load knees as the knee scripts do (LPI, **no** `remove_outliers()`).
- `tests/test_compute_torsion_pipeline.py` mirrors `morphometry/docker/compute_torsion.py` end-to-end (the production oracle); `tests/test_import_graph.py` enforces the package dependency DAG; `tests/unit/test_geometry.py` covers the shared geometry helpers. The batch scripts' pure halves are unit-tested without data: `tests/unit/test_knee_thickness_script.py`, `test_thinnest_flexion_script.py`, `test_process_nako_large_script.py`, `test_knee_cohort_report_script.py`.
- Cohort numbers quoted in this file and in `docs/wirth_eckstein_2008_plan.md` about the `dfg_knees` knees are regenerated by `scripts/report_knee_cartilage_cohort.py` (see the `scripts/` section), not by pytest: a golden pins one scan, the report re-derives the tables. The step-5 run over all 96 reproduced every figure to the decimal quoted bar five, which were corrected in the plan doc.

`api/tests.py` remains a separate collection of manual HTTP smoke tests against a running API (it assumes a specific local data layout); `api/tests.ipynb` and `notebooks/augsburg_test.ipynb` serve the same exploratory purpose.

## Dependencies

`requirements.txt` pins the full environment (Python 3.10, `torch==2.5.1`, `nnunetv2==2.5.1`, `numpy==1.26.4`, `SimpleITK`, `nibabel`, `pyvista`, `pydicom`, `fastapi`, `TotalSegmentator`, `pingouin`, etc.). When bumping `torch` or `nnunetv2`, also update the pins inside `api/docker/nnunet_torsion/Dockerfile` — they are installed independently in the image.

`morphometry/requirements.txt` is the **torsion image's** pinned set: the ten direct dependencies of `compute_torsion.py`'s import closure plus their transitive closure (38 packages), each at exactly the repo-root `requirements.txt` version — **every version in the two files must stay equal**. It is installed `--no-deps`, so a missing package fails the build instead of resolving unpinned. `scikit-learn`, `pandas` and `pyvista` are still needed (module-level imports in `hip.py`/`knee.py`). Details: [`torsion_image_rebuilds.md`](docs/evidence/torsion_image_rebuilds.md).

## Conventions worth preserving

- Docstrings are verbose: one-line purpose, `:param:` / `:return:` for each parameter. New public functions should follow this style (see e.g. `morphometry/measurements/hip.py::calculate_ccd`).
- Measurement functions commonly accept a `plot: bool | plt.Axes` argument — passing an `Axes` draws overlays instead of creating a new figure. Preserve this pattern when adding new measurements.
- Physical (mm) coordinates stay floats; only *index* coordinates are rounded, and only where an array is indexed (casting mm to `int` makes a measurement depend on scanner position). Both shaft axes (`get_femoral_shaft_axis{,_ct}`), the Murphy neck base and the CT head centre return unrounded values.
- **The package has no image-rotating landmark transport left** (no `skimage.transform.rotate`): the condylar line, the alpha-angle reslice and the Lee neck axis all rotate coordinates or sample a plane anchored to anatomy. Keep it that way.
- Failures inside batch scripts and `compute_torsion.py` are caught per measurement and written as `np.nan`, so one bad patient never aborts a cohort run — keep this when extending them. **The caught tuple differs per file**: `compute_torsion.py` `(RuntimeError, AssertionError, ValueError)`; `process_augsburg_*` `(ValueError, IndexError, RuntimeError)` (so `AssertionError` aborts there); `process_nako{,_large}.py` bare `except Exception`; `process_nmdid.py` wraps its QC itself. New guards should raise `ValueError` (caught everywhere). Details: [`pipeline_conventions.md`](docs/evidence/pipeline_conventions.md).

### Hip

- **Subchondral distance** (`calculate_subchondral_distance_ray_tracing{,_ct}`): `cone_axis` defaults to `'superior'` (world +z, the weight-bearing dome); `'acetabulum'` is legacy opt-in (its centroid aims ~half the rays into the acetabular fossa). Ray loss is logged above `min_ray_fraction` (0.7).
- **MRI femoral-head centre** (`hip.py::get_femoral_head_center`, non-isotropic path) is a seeded consensus sphere fit in true mm (`_fit_femoral_head_consensus`; seed depth `C.FEMORAL_HEAD_SEED_DEPTH_MM`, tolerance `C.FEMORAL_HEAD_CONSENSUS_TOLERANCE_MM`). Pass `spacing=` (every `Image`-holding caller does); array-only callers fall back to `(1, 1, x_ratio)` voxels, latent and unrealised in production. The isotropic (NaKo) path is separate. **`x_ratio` is misnamed and deliberately not "corrected"** (task #6): it is inert everywhere except the MRI CCD's neck-shell sampling, where the dimensionally correct value is measurably worse; the CCD's ~138° level is a property of the neck landmark, inside the in-vivo 114–140° range, and is a documented property with no action indicated. Details: [`hip_landmarks.md`](docs/evidence/hip_landmarks.md), [`ccd_neck_shell_x_ratio.md`](docs/evidence/ccd_neck_shell_x_ratio.md).
- **Femoral-head slab** — `C.FEMORAL_HEAD_SLAB_RADII` stays **2.0 (decided and closed 2026-11-11)**, so the ~1.2 mm medial drag on the isotropic path and the inflated CT radius are documented, deliberately retained properties; reopen only with new data (Open item 1). Do not argue it down against the NaKo reader cases. The MRI slab walk's `zscore` guard was **removed** (2026-11-05; bit-identical on 320 production-frame probes); correcting it instead would be a different measurement. **CT head centre (shipped 2026-11-12):** `get_femoral_head_center_ct` trims its slab with a **two-sided box** about the top-slice centroid, half-widths from the head-dominated `superior_half`; the operative element is the newly added anterior bound in y. Seven CT goldens moved, four (bone lengths, tibial torsions) did not; no torsion-image rebuild needed (`compute_torsion.py` makes no `_ct` call). **Do not pool CT hip rows across this change.** Details: [`femoral_head_slab.md`](docs/evidence/femoral_head_slab.md).
- **Alpha angle** (`calculate_alpha_angle`) samples the oblique plane directly on a grid anchored to the head centre (`map_coordinates(order=0)`, window margin 1), so it is invariant under pad/crop (bitwise on 120/120 halves; probabilistic, not structural). It refuses anisotropic spacing and raises `ValueError` for a failed anterior/posterior search (formerly a silent angle to the window origin) and for any angle outside (0, 90). **Do not pool alpha-angle rows across this change.** It reads ~31–35° against a published ~54° (different plane): **do not tune `tol`, the `0.6·r` band or the `1.2·radius` ceiling toward the literature.** The radial-position sweep question is closed negatively; the `plane_rotation` mechanism stays. A head-radius plausibility gate waits on the slab decision. Two look-like-defects to leave alone: the unsigned 3-D rotation driver and the overwritten `sort_points_clockwise`. Details: [`alpha_angle.md`](docs/evidence/alpha_angle.md).
- **Murphy's distal landmark** (`femur.py::get_femoral_neck_base`) is the femur centroid on the first axial slice above the lesser trochanter's superior extent (continuous index, paired with the unrounded head centre); a level less than `C.MURPHY_NECK_BASE_MIN_HEAD_DISTANCE_MM` (10) below the head centre raises. A fixed 10 mm offset was rejected on the NaKo raters. Details: [`hip_landmarks.md`](docs/evidence/hip_landmarks.md).
- **"Lee" is Schock et al. (Sci Rep 2021)'s method**, and its gap with Murphy is anatomy, not error (task #2 closed as premise refuted): a proximal neck landmark reads lower; ~21% retrotorted legs is arithmetic of the method. **Do not "fix" it against the Augsburg readers**, whose Lee−Murphy gap is what fails to reconcile. The Lee neck axis (`get_femoral_neck_center_lee` / `contour_femoral_neck`) now accumulates its angle on coordinates and samples the notch raster once on a grid anchored at `rint(head centre)` (`mode='grid-constant'`, deliberately not `'constant'`), giving exact translation invariance (100/100). Preserved, not endorsed: the endpoint rounding (`_NECK_ENDPOINT_ROUNDED`, bound ≤0.716°), the one-voxel band asymmetry and the three-voxel shear. Don't raise `find_notch`'s `thresh` to mimic the old dilated raster; don't "simplify" the frame-relative `np.floor` back to `int()`. `PA000346` left (35.8° → 3.2°) is the whole tail and is caught by the Lee QC gate. **Do not pool Lee rows across this change** (four Lee goldens moved). Any future change to the MRI head fit will move every Lee value again. Details: [`lee_neck_axis.md`](docs/evidence/lee_neck_axis.md).
- `method='tomczak'` works now (it always raised before) but its values are **unvalidated** — a regression anchor only. `utils.get_contour` reports the array border as contour; that reaches no measurement and is deliberately left alone. Details: [`hip_landmarks.md`](docs/evidence/hip_landmarks.md).

### QC gates — flag, never repair

A flagged leg keeps its value; flags are `;`-joined `name:value` strings in the `qc_flags` column (Murphy, then Lee, then condylar, via `_join_flags`). All thresholds print and compare in **in-plane voxels**, and all three refuse in-plane anisotropic spacing. The wrappers (`get_murphy_neck_qc{,_ct}`, `get_lee_neck_qc`, `get_condylar_line_qc`) **never raise** — they catch `Exception` and report `*_landmark_failed:<reason>` — and that contract must not be narrowed: in both Augsburg scripts the QC calls sit outside any `try`, so an escape would abort the cohort run. All three are pinned non-mutating at runtime by `tests/characterization/test_qc_gates_non_mutating.py` (byte-equal arrays, `np.array_equal` values). Surveys and verification: [`qc_gates.md`](docs/evidence/qc_gates.md).

- **Murphy** (`femur.py::murphy_neck_base_qc`): `murphy_axis_short:<voxels>` (axis under `1/tan(C.MURPHY_AXIS_ANGLE_TOLERANCE_DEG=4°)` = 14.30 voxels — threshold placed on the cohorts' own distribution, so their flag rate is not unbiased), `murphy_base_in_head:<mm>` (3-D, inside the fitted sphere), `murphy_base_outside_bone`, `murphy_base_slice_multipart`. An axis under one voxel raises. 5 of 1887 MRI halves flagged. **`case-105094` image-left — the CT fixture every `*_ct_left` golden uses — is flagged** (`murphy_axis_short:13.73`); don't quote it as a well-conditioned reference. Don't pool `qc_flags` strings across the 2026-11-06 mm→voxel unit change.
- **Lee** (`femur.py::lee_neck_axis_qc`): `lee_ring_within_head_radius:<a>` when `a < C.LEE_MIN_NOTCH_DISTANCE_FACTOR` (1.0) — named for the radius because a true containment reading is false. 4 of 1896 Augsburg-large halves, 1 of 100 reader-study, 0 of 120 NaKo.
- **Condylar line** (`knee.py::knee_condylar_line_qc`): `femoral_condylar_line_short:<voxels>` / `tibial_condylar_line_short:<voxels>`, threshold `C.CONDYLAR_LINE_ANGLE_TOLERANCE_DEG = 4.0` (a separate literal, deliberately not an alias of the Murphy one). It reflects a right half first, as the consumers do. The femoral gate sits in a gap; the tibial one rests on conditioning alone. Flagged on Augsburg: `PA000877` femur 5.39, `PA000155`/`PA000571`/`PA000435` tibia (all image-right).

### Knee cartilage (Wirth & Eckstein 2008)

The full conventions are in [`knee_cartilage.md`](docs/evidence/knee_cartilage.md) and the cohort figures in [`cartilage_cohort_figures.md`](docs/evidence/cartilage_cohort_figures.md); the plan is `docs/wirth_eckstein_2008_plan.md`. In brief:

- **Surfaces** (`cartilage/surface.py`): bone meshes in world mm, **RAS for both backends**; cAB = triangles within `0.6 ×` voxel diagonal of that bone's cartilage; tAB = cAB + enclosed holes; exact numpy mesh splitting; all shares by triangle area, never voxel counts.
- **Tibia** (`tibia_regions.py::subdivide_tibia`) needs the tibial **bone** label (2); plates named by **patient side from RAS world x**; per plate central ellipse (20% of tAB) plus four ±45° wedges. **Femur** (`femur_regions.py::subdivide_femur`) needs the femoral bone label (1); bone-only femoral frame (metaphyseal shaft band — chosen on these same 96 scans, so its invariance figures are not unbiased), trochlear notch read off the bone, cF to `weightbearing_fraction` 0.6, thirds by normalised arc length, plus trochlea and posterior-condyle extension zones. Region keys are **patient-side** (`clt`/`crt` …); medial/lateral names only via `laterality=`, and `split_tibia_thickness_by_compartment` refuses `*MT` keys.
- **Thickness** (`cartilage/thickness.py`): area-weighted exact surface distance both ways, `aMe = (cMe + bMe)/2` is the number to quote; two families (cAB bare names, `tab_*` with denuded bone at 0) must not be mixed. Maps keyed in **one shared in-plane frame, the tibial plateau plane** (`KneeFrame`); a femur measured without a tibia gets a femoral frame that `find_thinnest_area` refuses to pair. `method=` rejects every value.
- **Thinnest area** (`find_thinnest_area`, `find_thinnest_area_by_compartment`): read the function's warning — the global argmin is almost always a rim cell; full-window restrictions apply to both the centre search and the refinement; `edge_margin_mm` / `min_thickness_mm` are off by default because the data cannot choose them.
- **Deformation maps** (`cartilage/{registration,deformation,deformation_plot,robust_search}.py`): bone-only rigid ICP onto the unloaded-30° reference; femoral map = `UnrolledFemur` (absolute arc length from the notch plane over the whole reference bone); hotspot search keeps `DEFAULT_HOTSPOT_EDGE_MARGIN_MM` (2.0) from each footprint edge — edge crowding is moved, not solved; enclosed label cavities read as near-zero spots (open caveat). Map units (one-sided cMe) differ from table units (aMe).
- **Do not pool** tibial or femoral subregion rows, or thickness rows, across the Wirth & Eckstein rewrites, nor Δ maps/hotspots across the edge-margin / detour / gap-cap / domain changes (per-subregion Δ rows are unaffected). Loaded/unloaded and flexion spreads are **spread, not precision**.
- 3D visualisation: `Tibia`/`Femur.plot_segments()` / `plot_regions()` / `plot_thickness()`, `plot_knee_segments` / `plot_knee_thickness`, `plot_thinnest_point`; all accept `plotter=` and only `.show()` what they created. `docs/illustrations/plot_knee_zones.py` renders the didactic figures off the library.
