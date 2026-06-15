# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository layout

The repo has three loosely-coupled parts that share a single virtualenv and the `morphometry` package:

- `morphometry/` — the analysis library, layered as an acyclic dependency graph:
  - **`morphometry/measurements/`** — all public `calculate_*` measurement functions, one submodule per region (`hip.py`, `femur.py`, `knee.py`, `tibia.py`, `ankle.py`, `whole_leg.py`, `cartilage.py`). This is the public surface (re-exported from `morphometry.measurements`). Import measurements from here, e.g. `from morphometry.measurements.femur import calculate_femoral_torsion`.
  - **Region modules** at the package root (`hip.py`, `knee.py`, `ankle.py`, `femur.py`, `tibia.py`, `whole_leg.py`, `cartilage/`) hold the landmark / reference-line helpers (`get_*`) that the measurements build on. They must NOT import from `measurements/`.
  - **Leaf infrastructure**: `image_io.py` (the `Image` / `Segmentation` wrapper every function operates through), `geometry.py` (shared pure helpers: `fold_to_acute`/`fold_to_obtuse`, `split_left_right`, `slice_centroid_to_point`, `mirror_sagittal_coordinate`), `constants.py` (modality-scoped `SegmentationLabels` + tunable thresholds), `utils.py`, `bresenham.py`. These import no region/measurement code.
  - Dependency direction (enforced by `tests/test_import_graph.py`): `measurements/* → region get_* → geometry/utils/constants/image_io/bresenham`.
  - MRI vs CT: torsion / CCD / neck-center / bone-length have separate `_ct` functions (modality differs in landmark acquisition); the acetabulum / CEA / subchondral / offset family take a `ct=` flag. Shared `_core` helpers isolate the modality differences.
  - Torsion `calculate_*` return only the angle; landmarks come from `get_femoral_torsion_landmarks` / `get_tibial_torsion_landmarks`. Measurement functions never change their return arity based on `plot` — pass a matplotlib `Axes` / PyVista `Plotter` to `plot=` to draw overlays.
- `api/` — a FastAPI service (`api/rest_api.py`) that ingests DICOM uploads or Orthanc callbacks, persists `Examination` objects via `FileController`, and dispatches jobs through `ModelController`. Jobs shell out to docker images (see below).
- `scripts/` — one-off batch processing scripts (`process_augsburg_*.py`, `process_nako*.py`, `combine_series.py`, etc.). Each script is self-contained and typically prepends `sys.path.append('/home/simon/Work/morphometry')` and hard-codes absolute data paths (e.g. `/home/simon/Data/...`); update paths when running elsewhere.

## Core data flow

1. **Images** — every analysis function expects a `morphometry.image_io.Image` or `Segmentation`, which wraps either a `nibabel.Nifti1Image` or a `SimpleITK.Image`. Construct with `Image.from_nibabel(...)` / `Segmentation.from_nibabel(...)` or `Image('nibabel').read_image(path)`.
2. **Coordinate system** — masks must be placed in **LPI** orientation before measurement. Always call `seg.transform_coordinate_system()` right after loading; `remove_outliers()` is also commonly applied before measurement.
3. **Left/right splitting** — pipelines split a full-body mask at `array.shape[0] // 2` into `left_*` / `right_*` halves. Note: "left" / "right" refers to the *image* side, not the patient side. Functions take a `side='left'|'right'` argument and document this convention explicitly.
4. **Torsion landmarks** — `compute_torsion.py` now produces femoral landmarks nested by method: `landmarks['femur']['Lee'|'Murphy'][side]`. Tibia landmarks stay flat at `landmarks['tibia'][side]`. The REST layer (`rest_api.get_examination_by_id`) handles both shapes.
5. **Whole-leg CT** — `morphometry/whole_leg.py` and the `*_ct` variants in `femur.py` / `hip.py` use `split_ct_image` (in `image_io.py`) to derive hip/knee/ankle sub-volumes from a single whole-leg segmentation with labels: femur=1, tibia=2, fibula=3, patella=5, hip=7 (defaults). `calculate_mikulicz_deviation` was renamed to `calculate_mechanical_axis_deviation`.

## Docker images

The API does not run inference in-process — it invokes two external docker images built from this repo. Both are pulled/tagged as `swestfechtel/*:latest` and are referenced by name in `api/model_controller.py`.

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

```bash
# from repo root, with venv activated
uvicorn api.rest_api:app --host 0.0.0.0 --port 8000
```

The API writes logs to `api/logs/` (file handler, see `api/utils.py::init_logger`). `api/orthanc_plugin.py` is loaded **inside an Orthanc server process**, not by the FastAPI app — it forwards stored DICOM instances to `http://localhost:8000/upload/orthanc`.

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
- Failures inside batch scripts and `compute_torsion.py` are caught per-measurement (`except (RuntimeError, AssertionError, ValueError)`) and written out as `np.nan` so a single bad patient does not abort a cohort run. Keep this behavior when extending those scripts.