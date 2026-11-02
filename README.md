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
    `cartilage/` holds the knee cartilage subregions and thickness after Wirth &
    Eckstein 2008 (`surface.py` = bone-surface geometry in world mm,
    `tibia_regions.py` and `femur_regions.py` = the per-bone subdivisions,
    `thickness.py` = the area-weighted surface thickness and the in-plane frame
    both bones' maps are keyed in, `knee.py` = the `Tibia` / `Femur` classes).
    Thickness is the area-weighted Euclidean distance between each bone's
    subchondral bone area and its articular cartilage surface, in both
    directions; there is no choice of "method" any more (the old voxel-column
    `mesh` / `knn` values raise), and rows measured with the old definition must
    not be pooled with the new ones.
  - Infrastructure: `image_io.py` (the `Image` / `Segmentation` wrapper),
    `geometry.py`, `constants.py`, `utils.py`, `bresenham.py`.
- **`api/`** — a FastAPI service that ingests DICOM uploads / Orthanc callbacks
  and dispatches inference + measurement jobs to docker images.
- **`scripts/`** — self-contained batch-processing scripts (hard-coded data
  paths; adapt before running elsewhere). A few are proper CLIs, e.g.
  `compute_knee_cartilage_thickness.py` (per-subregion knee cartilage thickness
  + 3D renderings) and `report_knee_cartilage_cohort.py`, which regenerates the
  `dfg_knees` cohort tables quoted in `CLAUDE.md` and
  `docs/wirth_eckstein_2008_plan.md` (Wirth & Eckstein Table I / II analogues,
  region areas, thickness per subregion, loaded-vs-unloaded and across-flexion
  spread, QC summary) into CSVs plus a markdown report and a `.meta.json`
  recording the commit it was produced at. `analyse_cartilage_deformation.py`
  maps cartilage thickness change per knee against its unloaded 30° scan
  (bone-registered Δ maps, per-subregion Δ, largest-thinning hotspots, figures,
  and a cohort report with uncorrected Wilcoxon p-values) on the current
  `dfg_knees` label set (`/home/simon/Downloads/dfg_knees_segmentations/dfg_knees/segmentations`):

  ```bash
  python scripts/analyse_cartilage_deformation.py \
      -i /home/simon/Downloads/dfg_knees_segmentations/dfg_knees/segmentations \
      -o /home/simon/Data/dfg_knees/deformation_<short sha> --workers 16
  ```

  How it works: each configuration's **bone** surface (never the cartilage) is
  rigidly ICP-registered onto the reference's, cropped to the articular end; on
  the current run (`/home/simon/Data/dfg_knees/deformation_0d993a3`) the residual
  RMS is median 0.121 / max 0.208 mm (femur, n = 98) and 0.091 / 0.142 mm (tibia,
  n = 88), far below the 1.0 mm QC threshold (`REGISTRATION_MAX_RMS_MM`, kept
  because the residuals show no gap). The tibial map is the reference's plateau
  plane; the femoral map is *unrolled*: medial-lateral position against signed arc
  length (mm) along the sagittal bone profile, measured from the trochlear-notch
  plane (a bone landmark), so trochlea and posterior condyle each appear once.
  Sagittal levels at the outer condyle edges whose profile folds onto the side
  wall are not used, which leaves 0.1-7.0% of a femur's covered area unmapped
  (median 3.1%). The largest-thinning hotspot is searched only more than 2 mm
  (`--edge-margin-mm`) from each scan's map edge, against the near-zero rim band
  and the edge mismatch between acquisitions; many hotspots still sit just past
  that margin, and single-voxel cavities in some cartilage labels (all of K10)
  read as thin spots that can capture a hotspot — `report.md` lists both caveats.

  Outputs: `subregion_delta.csv`, `hotspots.csv`, `registration.csv`,
  `failures.csv`, `<knee>_{tibia,femur}_delta.png`, `{bone}_hotspots_cohort.png`,
  `cohort_summary.csv`, `report.md`, `report.meta.json`. Without
  `--laterality-table` all knees are pooled by patient side (assumed same side).
  The existing knee goldens and `report_knee_cartilage_cohort.py`'s quoted tables
  still use the older label set under `/home/simon/Data/dfg_knees/monai_segmentations`
  (follow-up).
- **`frontend/`** — the Next.js 15 / React 19 web UI for the API (DICOM upload,
  job dispatch, and interactive landmark editing of results). The torsion viewer
  uses **Cornerstone3D** to render the NIfTI volumes directly (see "Running the
  stack" below).

Measurements operate on masks placed in **LPI** orientation; load with
`Segmentation(...).read_image(path)` then call `transform_coordinate_system()`.
The `side` argument everywhere refers to the **image** side, not the patient
side. The axial-plane angles (femoral torsion, tibial torsion, knee rotation, femoral
anteversion) are composed from voxel indices and therefore need **equal in-plane
x/y voxel spacing**; that precondition is checked wherever the function holds the
image geometry. The knee/ankle entry points that historically took a bare
`np.ndarray` accept an `Image` too (preferred — the spacing then comes for free),
or take it explicitly via `spacing=` / `knee_spacing=` / `ankle_spacing=`. An
in-plane anisotropic mask raises `ValueError` instead of returning a sheared angle.
See `CLAUDE.md` for the full architecture and conventions, and
`docs/reader_measurement_guide.md` for anatomic descriptions of each measurement.

## Tests

```bash
pytest -m "not needs_ct"          # fast characterization + unit tests
pytest                            # include CT golden tests (slow; needs CT data)
MORPH_UPDATE_GOLDEN=1 pytest      # (re)capture golden baselines
```

Tests skip cleanly when sample data is absent. Data paths can be overridden via
`MORPH_AUGSBURG_PA000001`, `MORPH_AUGSBURG_READER_STUDY`, `MORPH_NAKO_SAMPLE_DIR`,
`MORPH_CT_SAMPLE`, `MORPH_DFG_KNEES_DIR` (the knee cartilage goldens; marker
`needs_dfg_knees`), `MORPH_DFG_KNEES_CURRENT_DIR` (the current label set, for the
cartilage deformation characterization). The data-requiring markers are `needs_augsburg`,
`needs_augsburg_reader_study`, `needs_nako`, `needs_ct` and `needs_dfg_knees`.

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
