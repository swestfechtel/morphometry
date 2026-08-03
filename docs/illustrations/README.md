# Parameter illustrations

Schematic SVG diagrams for each morphometric parameter in
[`../reader_measurement_reference_table.md`](../reader_measurement_reference_table.md).
Each shows a simplified bone outline plus the reference lines, landmarks and the
measured angle / distance — a didactic companion to the definitions, **not**
anatomically exact drawings and **not for clinical use**.

## Visual key

| Element | Meaning |
|---|---|
| Blue line | proximal / first reference line or axis |
| Red line | distal / second reference line or axis |
| Dashed grey | construction line (reference direction, perpendicular, superimposition) |
| Emerald arc + value | the measured **angle** |
| Purple double-arrow + value | the measured **distance** (mm) |
| Dark dots | anatomical landmarks |

Reference-range values shown in the diagrams (e.g. `≈127°`, `≈500 mm`) are
representative normal-adult figures from the reference table; the JLCA and
knee-rotation angles are drawn deliberately exaggerated because the real values
are near-parallel.

## Files

| Parameter | File |
|---|---|
| Femoral torsion (anteversion) | [`femoral_torsion.svg`](femoral_torsion.svg) |
| Tibial torsion | [`tibial_torsion.svg`](tibial_torsion.svg) |
| Knee rotation angle | [`knee_rotation.svg`](knee_rotation.svg) |
| CCD (neck–shaft) angle | [`ccd_angle.svg`](ccd_angle.svg) |
| Bone length | [`bone_length.svg`](bone_length.svg) |
| Acetabular version | [`acetabular_version.svg`](acetabular_version.svg) |
| Center-edge (CE) angle | [`ce_angle.svg`](ce_angle.svg) |
| Femoral offset | [`femoral_offset.svg`](femoral_offset.svg) |
| Hip–knee–ankle (HKA) angle | [`hka_angle.svg`](hka_angle.svg) |
| Joint-line convergence angle (JLCA) | [`jlca.svg`](jlca.svg) |
| Mechanical-axis deviation (MAD) | [`mad.svg`](mad.svg) |
| Subchondral distance (hip joint space) | [`subchondral_distance.svg`](subchondral_distance.svg) |

## Regenerating

The SVGs are produced by a generator so the drawn geometry (arc sweeps,
perpendicular feet) matches the anatomy and the whole set stays visually
consistent. Edit and re-run:

```bash
python docs/illustrations/generate_illustrations.py
```

## 3D render (real data)

`subchondral_distance.svg` is a schematic; `subchondral_distance_3d_{left,right}.png`
are **real** renders of the ray tracing on a NaKo hip segmentation. The script runs
the actual `measurements/hip.py::calculate_subchondral_distance_ray_tracing` with a
PyVista `Plotter`, so the rays (red), femoral-head exit points (blue) and acetabulum
hit points (green) are drawn by the measurement itself; it adds the femur and
acetabulum surfaces (marching cubes) as translucent context. Needs the sample data
and a PyVista off-screen backend:

```bash
MORPH_NAKO_SAMPLE_DIR=/path/to/segmentations \
    python docs/illustrations/plot_subchondral_3d.py --side left  --html
MORPH_NAKO_SAMPLE_DIR=/path/to/segmentations \
    python docs/illustrations/plot_subchondral_3d.py --side right --html
```

The render also shows the **peri-acetabular crop** that aims the cone: the pelvis
label (MRI acetabulum / CT hip-bone) includes the iliac wing, whose mass would pull
the cone axis superiorly and over-sample the roof, so the measurement restricts the
label to voxels within `acetabulum_radius_factor` femoral-head radii of the FHC
(default **1.5**, which isolates the acetabular articular surface while retaining
essentially the full cone of rays). The faint mesh is the full label; the dark-gold
cap is the crop actually used.

`--html` also writes a self-contained, rotatable `…_{side}.html` (open in a browser;
~7–8 MB each — regenerate rather than commit if you want to keep the repo lean).
`--interactive` opens a live window; `--radius-factor` sets the crop (≤0 disables it
and reverts to the full-label behaviour); `--out`, `--n-rays`, `--cone-angle`
override the rest.

## LaTeX / PDF

`../reader_measurement_reference_table.tex` embeds these as an "Illustrations"
grid via `\includegraphics`, using **vector** PDFs in [`pdf/`](pdf/). Those are
converted from the SVGs with headless Chrome (the SVGs style themselves with a
CSS `<style>` block that rsvg/MuPDF ignore, so a browser engine is required):

```bash
python docs/illustrations/build_pdf.py     # SVG -> pdf/*.pdf (needs google-chrome/chromium)
cd docs && pdflatex reader_measurement_reference_table.tex
```

Re-run `build_pdf.py` after changing any SVG, then recompile the LaTeX.
