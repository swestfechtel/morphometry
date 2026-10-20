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

### Cone axis (`--axis`)

`--axis` mirrors the measurement's `cone_axis` argument and controls where the ray
cone points. Every render draws the axis itself as an arrow from the FHC, one colour
per mode (acetabulum **purple**, superior **cyan**); in `compare` each panel draws
*both* arrows — its own solid, the other thin and translucent — and prints the angle
between them. The camera is placed perpendicular to the plane the two axes span
(roughly an anterior view, superior up) so the tilt and the cone opening are visible
rather than seen end-on, and the peri-acetabular crop is clipped to the far side of
the FHC so it backs the rays instead of hiding them.

| mode | axis | note |
| --- | --- | --- |
| `superior` (**default**) | straight up (world `+z`, which is Superior in both RAS+ and LPS) | fixed and standardised — samples the weight-bearing acetabular dome in every subject — but ignores individual acetabular orientation/version |
| `acetabulum` (legacy) | FHC → peri-acetabular centroid | follows the individual acetabulum, but the centroid depends on how much ischium/pubis/ilium the label includes and on the crop radius, and it sits ~46° medial of superior, spraying ~half its rays into the non-articular acetabular fossa |
| `compare` | both, side by side | one linked-camera scene, `subchondral_axis_comparison_{side}.png` |

`--axis` defaults to whatever the library's `cone_axis` default is (read off the
signature at import), currently `superior`; the render for that mode keeps the plain
`subchondral_distance_3d_{side}.png` name and any other mode is suffixed
`…_{side}_{axis}.png`, so the unsuffixed file always shows the default behaviour.

```bash
MORPH_NAKO_SAMPLE_DIR=/path/to/segmentations \
    python docs/illustrations/plot_subchondral_3d.py --side left --axis compare --html
```

On the NaKo sample the two axes differ by ~40–55° (mean ≈46° over 8 cases, both
hips), so with the default 45° half-angle the sampled regions only partially
overlap and the superior axis reads systematically lower (see below). That gap is
not a tie: the centroid axis is dragged medially by the ischium/pubis in the label
and points roughly at the **acetabular fossa**, the non-articular depression where
the femoral head touches no bone. Roughly half its rays cross the fossa and only
stop at its far wall, which inflates the mean by ~1.5 mm and roughly triples the
spread. Near the axis, where both modes sample true lunate surface, they agree
(3.65–3.77 mm vs 3.78–3.94 mm over the innermost 0–20°). This is why `superior` is
the library default; `acetabulum` is retained only to reproduce results captured
before the default changed.

`--table N` skips rendering, runs both axis modes over the first `N` sample cases ×
both image sides, prints the comparison and writes it as CSV (`--table-csv`).
Per-case failures are recorded as `NaN` rather than aborting the sweep. Over the
first 8 cases the superior axis gave a mean 1.1–2.3 mm *smaller* distance with a
markedly smaller spread (std ≈0.9 vs ≈2.2 mm), since it samples the roof only and
avoids the deep/peripheral rays the tilted axis picks up.

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

### 2D slices: `plot_subchondral_2d.py`

The same measurement as three orthogonal slices of the PD MRI — sagittal, coronal and
axial — all through the femoral head centre the measurement uses, with the femur,
peri-acetabular crop and full pelvis label outlined, the head centre and radius, the
cone axis, and the rays. The dashed cone edges are the actual angular extent of the
rays in that slice, not the nominal ±45° half-angle, so they come out narrower and can
be asymmetric: on case 100186 the coronal slice shows 23° laterally against 40°
medially, because lateral rays clear the acetabular rim and are dropped.

```bash
MORPH_NAKO_SAMPLE_DIR=/path/to/segmentations \
    python docs/illustrations/plot_subchondral_2d.py --case 0 --side left
```

Rays are 3D, so each panel draws rays whose head exit *and* acetabulum hit lie within
`--slab` voxels (default 1.5) of the slice in bold, and a faint projection of all
surviving rays. With the default superior axis the axial slice through the head centre
is perpendicular to the cone axis, so it contains **no** rays: that panel shows only
the projected footprint of the sampled roof and marks the axis as pointing out of the
page. The MRI is read from `MORPH_NAKO_IMAGE_DIR` (default
`/home/simon/Data/NaKo_sample/nifti`, `<stem>_0000.nii.gz`); without it the panels fall
back to the segmentation. Writes `subchondral_distance_2d_{side}.png` (gitignored).

### Golden-angle ray layout: `plot_golden_angle.py`

Explains how the subchondral cone spreads its 200 rays, using the measurement's own
`_sample_cone_directions`. Needs no sample data.

```bash
python docs/illustrations/plot_golden_angle.py
```

Writes `golden_angle.png` and `golden_angle.gif`, both viewed straight down the cone axis.
The rays are placed by two independent rules: each is **turned** a further 137.5° (the
golden angle) around the axis, and **tilted** a little further from it, evenly in area
from ~2° out to the 45° rim. The four panels show the turn alone, the real layout, the
same tilts with a round 90° turn (which stacks into four spokes and leaves wedges
unsampled), and a slice through the axis — only 18 of 200 rays fall in it, reaching
40.2° / 44.9°, which is why cone edges drawn from in-slice rays in
`plot_subchondral_2d.py` look narrower than 45° even with no rays lost. The GIF places
the rays one at a time.

## Knee cartilage subregions (real data): `plot_knee_zones.py`

The Wirth & Eckstein 2008 subregions (`docs/wirth_eckstein_2008_plan.md` steps 1-5)
rendered on real `dfg_knees` segmentations, using the library's own plotting API
(`Tibia.plot_regions` / `Femur.plot_regions` / `plot_knee_segments` /
`plot_knee_thickness` / `plot_thinnest_point`) and its own colour tables
(`TIBIA_SUBREGIONS` / `FEMUR_SUBREGIONS`), so a figure cannot disagree with the
numbers. Every tunable the figures depend on — the central share, the
weight-bearing fraction, the bone labels — is read off the library signatures.

```bash
python docs/illustrations/plot_knee_zones.py            # all figures -> out/knee_zones/
python docs/illustrations/plot_knee_zones.py --figures tibia femur --no-html
python docs/illustrations/plot_knee_zones.py --scan K2_0_unbelastet --flexion-scan K2_90_unbelastet
python docs/illustrations/plot_knee_zones.py --animate   # + turntable GIF/MP4 per figure
```

`--data-dir` (or `MORPH_DFG_KNEES_DIR`) points at the segmentations, `-o/--out`
at the output directory (default `out/knee_zones`, which is gitignored — these are
regenerable renders, like the subchondral ones above), `--size W H` sets the PNG
size (default 1600x1200 for the whole figure), `--no-html` skips the interactive
exports and `--allow-flagged` permits the three QC-flagged scans
(`K7_90_belastet`, `K8_90_belastet`, `K11_90_belastet`), which are refused by
default because a figure drawn from one of them illustrates the defect.

| `--figures` | shows |
| --- | --- |
| `tibia` | the tibial tAB from superior (paper Fig. 5A), both plates, with the central ellipse, the two ±45° wedge planes, the internal (blue) / anterior (red) directions and one label per region |
| `femur` | the femoral tAB from inferior (paper Fig. 6A) and from posterior, with the femoral frame's axes, the trochlear notch, the notch plane and each condyle's 60% posterior border |
| `both` | both bones in one scene: the labelled tAB next to the cartilage voxels the labels were transferred to |
| `thickness` | each tAB face-on as a thickness heatmap on one shared 0–max mm scale, plus a second figure painting the 0 mm (denuded / rim) triangles magenta |
| `flexion` | the same knee at 0° and 90°: the raw pose, then both subdivisions in their own bone frames with the region shares printed — the visual proof that the regions no longer move with flexion |
| `thinnest` | the thinnest-point arrows for the whole knee and both compartments |

Every panel carries the RAS orientation marker (R = patient right), and the
region keys are the **patient-side** ones the code uses (`clt` … `prt`, `iclf` …
`prf`). They are not relabelled medial/lateral: the `dfg_knees` laterality is
unrecorded (plan open issue 3). `--laterality left|right` passes an explicit
assumption to the library, which then adds the paper's medial/lateral names in
brackets; the figure titles mark it as an assumption.

The script also prints the numbers the figures must agree with — each tibial
central region is exactly 20% of its plate's tAB, each femoral third ~33.3% of
its condyle's weight-bearing tAB, and the patient-left plate is the one with the
smaller RAS world x — plus the region-transfer QC and `scan_qc_flags`.

Scenes that are a single renderer are also written as self-contained rotatable
HTML (same mechanism as `compute_knee_cartilage_thickness.py --html`). PyVista's
HTML exporter carries one renderer, so a multi-panel figure cannot be exported as
one interactive scene; the two comparison figures instead re-draw each panel on
its own (`flexion_*_<scan>.html`, at the panel's shared magnification) and write a
`flexion_*.html` that puts them side by side in iframes. Every run also writes an
**`index.html`** gallery of that run — still, turntable and links per figure — so
the output is one page to open rather than a directory listing.

`--animate` additionally films a turntable of every figure drawn: each panel spins
a full revolution **about its own anatomical up vector**, not about a shared world
axis, because the panels of a comparison figure are posed in their own bone frames
and a common axis would swing them out of step. The focal point, distance and
parallel scale are held, so the anatomy neither drifts nor changes size as it
turns — and the 0° / 90° comparison then holds up through the whole rotation, not
just at the one viewpoint the still was taken from. `--frames` (default 48) and
`--fps` (default 16) set the revolution; each figure gets a `.gif` (half linear
size, since a full-size 48-frame GIF runs to tens of MB for no extra legibility)
and, where `ffmpeg` is on `PATH`, a full-size inter-frame-compressed `.mp4`.
`ffmpeg` is called as a subprocess rather than through `imageio-ffmpeg`, which is
not in `requirements.txt`. A full `--animate` run is ~12 min and ~160 MB.
