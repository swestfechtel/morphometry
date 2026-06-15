# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository layout

The repo has three loosely-coupled parts that share a single virtualenv and the `morphometry` package:

- `morphometry/` — the analysis library. One module per anatomic region (`hip.py`, `knee.py`, `ankle.py`, `femur.py`, `tibia.py`, `whole_leg.py`, `cartilage/`). Every measurement function operates on segmentation masks through the `Image` / `Segmentation` wrapper in `morphometry/image_io.py`. Utilities live in `morphometry/utils.py`.
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

There is no pytest / unittest runner configured. `api/tests.py` is a collection of manual smoke tests that hit a running API over HTTP (`requests.post('http://localhost:8000/...')`) and assume a specific local data layout. Invoke individual functions from a REPL or adapt the paths before running. `api/tests.ipynb` and `notebooks/augsburg_test.ipynb` serve the same exploratory purpose.

## Dependencies

`requirements.txt` pins the full environment (Python 3.10, `torch==2.5.1`, `nnunetv2==2.5.1`, `numpy==1.26.4`, `SimpleITK`, `nibabel`, `pyvista`, `pydicom`, `fastapi`, `TotalSegmentator`, `pingouin`, etc.). When bumping `torch` or `nnunetv2`, also update the pins inside `api/docker/nnunet_torsion/Dockerfile` — they are installed independently in the image.

## Conventions worth preserving

- Docstrings are verbose: one-line purpose, `:param:` / `:return:` for each parameter. New public functions should follow this style (see e.g. `morphometry/hip.py::calculate_ccd`).
- Measurement functions commonly accept a `plot: bool | plt.Axes` argument — passing an `Axes` draws overlays instead of creating a new figure. Preserve this pattern when adding new measurements.
- Failures inside batch scripts and `compute_torsion.py` are caught per-measurement (`except (RuntimeError, AssertionError, ValueError)`) and written out as `np.nan` so a single bad patient does not abort a cohort run. Keep this behavior when extending those scripts.