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
