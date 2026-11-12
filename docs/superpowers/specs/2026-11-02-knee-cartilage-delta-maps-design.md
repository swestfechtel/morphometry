# Knee cartilage deformation: delta maps against unloaded 30°

Date: 2026-11-02 · Branch / worktree: `knee-cartilage-delta-maps` (`../morphometry-delta-maps`)

## Intent

The `dfg_knees` cohort (current label set: 15 knees, see "Input data") has, per knee, up to eight MRI segmentations:
flexion 0/30/60/90° × unloaded (`unbelastet`) / loaded (`belastet`), named
`K<subject>_<angle>_<load>.nii.gz`. The earlier work (`scripts/analyse_thinnest_point_flexion.py`)
asked where the thinnest point sits and how it moves with flexion. This work asks how the
cartilage **deforms**: how its thickness changes, subregion by subregion and position by
position, relative to one reference configuration.

**Reference: unloaded 30°** (`K<s>_30_unbelastet`). Every other configuration of the same knee
is compared against it — seven comparisons per knee:

| | 0° | 30° | 60° | 90° |
|---|---|---|---|---|
| unloaded | Δ | **reference** | Δ | Δ |
| loaded | Δ | Δ | Δ | Δ |

Unloaded 30° vs loaded 30° isolates load; the unloaded row isolates flexion; the loaded 0/60/90°
cells mix both. The user chose to include all seven.

The work is used in two stages: **inspect knee by knee first, then cohort statistics.**

## Goals

1. **Per-subregion change.** For each knee, each of the seven comparisons and each Wirth &
   Eckstein subregion (the 10 tibial and 10 femoral keys of `TIBIA_REGION_KEYS` /
   `FEMUR_REGION_KEYS`), the change of `aMe` (ThC.cAB.aMe) against the reference, in mm and in %.
   Each configuration is subdivided on its own (step 1 of the user's pipeline: "determine the
   Wirth-Eckstein subregions for all configurations"), so subregion `alt` of `K3_0_belastet` is
   compared with subregion `alt` of `K3_30_unbelastet`.
2. **Per-position delta maps** — thickness of the configuration minus thickness of the
   reference at the **same anatomical spot of the bone** (the user explicitly chose this over
   "whatever cartilage is in the contact area at that angle"). Δ in mm, negative = thinner than
   the reference.
3. **Per-knee figures**, two per knee (tibia, femur), each the 2 × 4 grid above: the reference
   cell shows the reference's absolute thickness, the seven other cells show Δ on **one
   zero-centred diverging colour scale shared by the seven panels** of that figure. Subregion
   borders (of the reference) are drawn over every panel. Accompanied by the per-subregion
   table for that knee.
   - **Tibia**: top-down, looking along the plateau normal (the reference's `KneeFrame` from
     `knee_frame_from_tibia`).
   - **Femur**: **unrolled** (user chose option B over a view along the shaft): horizontal =
     medial–lateral, vertical = position along the condyle's sagittal profile from the trochlea
     (top) round the distal condyle to the posterior condyle (bottom). Every patch of femoral
     cartilage appears exactly once, so the 0° (distal) and 90° (posterior) contact zones are
     both visible.
4. **Largest-thinning hotspot** per knee × comparison × compartment: the centre of the area
   that thinned most, found on the smoothed Δ map, with its map coordinates, the (reference)
   subregion it lies in, and its smoothed and raw Δ. "Largest deformation" means **largest
   thinning** (most negative Δ), per the user — not largest absolute change.
5. **Cohort statistics** per comparison:
   - per subregion: n, mean, SD, median, min, max of Δ (mm and %), and a Wilcoxon signed-rank
     test of Δ against 0, reported **uncorrected** and labelled as such (n ≤ 15, n ≤ 8 for 90° loaded; 20 subregions ×
     7 comparisons are many tests);
   - hotspots: how often the hotspot lands in each subregion, and a scatter of all knees'
     hotspots over the unrolled femur and over the plateau (on a reference knee's outline or a
     neutral grid — builder's call), to show whether they cluster.

## Non-goals

- No change to the thickness measurement itself (`morphometry/cartilage/thickness.py` as
  defined in step 4 of `docs/wirth_eckstein_2008_plan.md`), to the subdivisions, or to any
  existing script's output.
- No precision / test-retest claim. These are separate acquisitions of a repositioned knee;
  the Δ values contain acquisition-to-acquisition spread (on the outdated label set, loaded vs
  unloaded at the same angle differed by a median |Δ aMe| 0.085 mm per subregion; not
  re-measured on the current set), and the report should say so next to the Δ values rather
  than present them as measurement precision.
- No medial/lateral naming *by default*: laterality of these knees is unrecorded, so
  compartments and subregions stay **patient-left / patient-right**, as everywhere else in the
  knee code, unless a laterality table is supplied (see "Laterality").
- No multiple-testing correction beyond labelling the p-values uncorrected (can be added later).

## Anti-goals (would make this a failure even if it runs)

- **Alignment that depends on the cartilage labels.** If the transform that puts two scans on
  top of each other is derived from where cartilage is labelled (plate centres from cAB, the
  arc fraction normalised to the edge of the trochlear cartilage), load and flexion leak into
  the alignment and the delta map measures its own frame. Alignment must come from **bone**.
- **A hotspot that is a rim artefact.** A cartilage map stops rather than fades, and the edge
  band of each map is near-zero thickness (see the warning on `find_thinnest_area`); a Δ map
  between two scans whose edges do not coincide will have its most negative cells at the rim.
  The hotspot search must stay off it.
- Pooling these numbers with anything produced by the superseded `knn`/`mesh` thickness.

## Decided

### Correspondence: rigid bone registration to the reference (my call, user did not object)

For each knee and each bone (femur, tibia separately — the joint moves between scans), every
configuration's **bone surface** is rigidly registered to the reference configuration's bone
surface of the same knee. Bone does not deform, so a rigid fit is the right model; the result
is a transform per (scan, bone) mapping that scan's world coordinates into the reference's.

- Initialise from the frames the subdivisions already compute (femoral frame:
  shaft / posterior-condylar / A-P axes; tibial plate frame), then refine with ICP on the bone
  surfaces (`surface.bone_surface_mesh` of the bone label). The initial frames may use
  cartilage (tibial plate centres); the **refinement must not** — it uses bone only.
- The knee FOV cuts the femoral and tibial shafts at different heights in different
  acquisitions (see the femoral shaft-axis note in `CLAUDE.md`), so the registration must be
  restricted to, or robust to, the part of the bone both scans contain (e.g. a band around the
  articular region, trimmed ICP / outlier rejection). How is the builder's call.
- **QC**: report the registration's residual (RMS / median / p95 surface distance after
  alignment, in mm) per (scan, bone) and flag a scan whose residual exceeds a threshold in the
  output's `qc_flags`. The threshold is the builder's to propose from the data; it must not be
  tuned on the Δ results.

### Delta maps are built in the reference's coordinates

Each configuration's per-triangle thickness (`ThicknessMaps.triangle_thickness`, the
`cAB → AC` distance on its own subdivision mesh; denuded triangles are excluded from the map,
not counted as 0) is carried into the reference's world frame by that scan's transform, then
binned into the **reference's** map coordinates:

- **Tibia**: the reference's `KneeFrame` (tibial plateau plane), cells of `grid_mm`.
- **Femur**: an unrolled 2-D coordinate computed **once per knee on the reference's femoral
  bone**, in **mm**, anchored to a **bone** landmark — not the existing 0–1
  `articular_arc_fraction`, which is normalised to the edge of the labelled trochlear cartilage
  and is documented as not reproducible across flexion. Suggested: horizontal = M-L coordinate
  of the femoral frame; vertical = signed arc length along the sagittal contour of the
  reference femur's articular bone surface, zero at the trochlear notch plane (a bone landmark,
  `find_trochlear_notch`), positive posteriorly. Because every scan is mapped into the
  reference's space first, the same coordinate function serves all eight configurations of the
  knee.

Δ per cell = configuration − reference, defined only where **both** have a measured cell;
cells measured in one of the two stay empty (drawn blank). Cell size may be coarser than the
thickness maps' 0.5 mm if the Δ maps are too sparse — builder's call, stated in the output.

The per-subregion Δ (goal 1) does **not** go through the registration: it is `aMe` of the
configuration's own subregion minus `aMe` of the reference's, exactly as each subdivision
reports it.

### Hotspot search

Reuse the thinnest-area machinery's logic rather than invent a new one: NaN-aware median
smoothing of the Δ map (kernel from a `neighbourhood_mm`), centre search restricted to cells
whose whole window lies inside the **both-scans** footprint, argmin of the smoothed Δ, then
refinement within a disc under the same restriction (the refinement-disc fix that
`find_thinnest_area` needed applies here too). Compartments: tibia patient-left plate,
patient-right plate; femur patient-left side, patient-right side (each side including its
trochlear and posterior zones, separated by the same plane that separates the condyles); plus
whole bone. If nothing survives the restriction, fall back to the whole footprint with a
WARNING, as `find_thinnest_area` does.

### Input data

- **`/home/simon/Downloads/dfg_knees_segmentations/dfg_knees/segmentations`** — the current
  label set (user, 2026-11-02). 113 scans of **15 knees** (K1–K11, K14, K15, K23, K25), uint16,
  labels 0–5, 0.3125 × 0.3125 × 0.75 mm. Every knee has its unloaded 30° reference. **Seven
  knees have no 90° loaded scan** (K7, K8, K10, K11, K14, K15, K25), so their `90° loaded`
  comparison is absent, not failed: the figure leaves that panel blank and labelled "no scan",
  and the CSVs omit the row (or mark it `missing`, builder's call) rather than writing NaN as if
  measurement had failed.
- **Outdated sets, not to be used or mixed in**: `/home/simon/Data/dfg_knees/monai_segmentations/
  labels/original` (96 scans; what `tests/data_paths.py::DFG_KNEES_DIR`, the committed knee
  goldens and every `dfg_knees` cohort number in `CLAUDE.md` / `docs/wirth_eckstein_2008_plan.md`
  were produced from) and its sibling `labels/final`. All 84 scans shared with the current set
  differ voxel-wise (0.02–1.1% of voxels). So **no figure quoted in `CLAUDE.md` for `dfg_knees`
  — shares, `aMe` means, the 0.085 mm loaded-vs-unloaded spread, the K7/K8/K11 QC flags —
  is known to hold for the current set**; the report must recompute any such baseline it quotes.
  A Δ map subtracts two label maps of one knee, so all eight configurations of a knee must come
  from the same directory.
- Default `-i` points at the current set's directory only if the builder wants a default; the
  script must accept any directory. Moving `tests/data_paths.py` / the goldens to the new set is
  **out of scope** (it would move every committed knee golden) — note it as follow-up.
- Knees and configurations are discovered generically from `K<subject>_<angle>_<load>` file
  names (my call), so further knees can be added without code changes; nothing hard-codes the
  knee count or assumes all eight configurations exist.

### Laterality (user: add support, assume same side for now)

The cohort statistics pool subregion keys across knees; `clt` is the medial plate of a right
knee and the lateral plate of a left one. The user believes the knees are all the same side but
has to confirm, so:

- **Default (no table): pool as-is**, assuming all knees are the same side, and say so in
  `report.md` next to the cohort tables.
- **`--laterality-table CSV(subject,laterality)`** (same format as
  `compute_knee_cartilage_thickness.py`'s): when given, the cohort stage maps every knee to one
  anatomical convention before pooling — subregion keys renamed to medial/lateral names
  (`rename_tibia_regions` / `rename_femur_regions`), compartments to medial/lateral, and the
  hotspot scatter mirrored so medial is on one side for every knee. Per-knee figures and the
  per-knee CSV rows keep patient-side keys and gain an `anatomical_subregion` column. A knee
  missing from a supplied table is excluded from the cohort statistics and listed.

### Cohort scope and QC

- 15 knees at present; the `90° loaded` comparison has n = 8, every other comparison n = 15
  (before QC exclusions). A knee without a usable reference scan is skipped and listed; a
  configuration that fails (subdivision, thickness, registration) is NaN for that comparison and
  listed, without dropping the knee's other comparisons.
- Scans carrying `scan_qc_flags` (orientation, transfer QC) or a registration-residual flag are
  measured and written but **excluded from the cohort statistics by default**, with
  `--include-flagged` / `--exclude SCAN ...` as in `scripts/report_knee_cartilage_cohort.py`.
  Which scans of the current set get flagged is not known yet (the known-bad 90° loaded scans of
  the old set are absent from it); the report lists them.

### Deliverables

- Reusable logic in the library (suggested `morphometry/cartilage/deformation.py`: registration,
  transfer into the reference frame, unrolled femoral coordinate, Δ maps, hotspot search,
  plotting), respecting the import DAG (`tests/test_import_graph.py`).
- A batch script, suggested `scripts/analyse_cartilage_deformation.py -i <dir> -o <dir>
  [--workers N --include-flagged --exclude SCAN ... --knees K1 K3 ... --laterality-table CSV]`,
  writing into `-o`:
  - per-knee figures `<knee>_tibia_delta.png` and `<knee>_femur_delta.png` (2 × 4 grid);
  - `subregion_delta.csv` — long: knee × comparison × bone × subregion, `aMe_ref`, `aMe`,
    `delta_mm`, `delta_pct`, `qc_flags`;
  - `hotspots.csv` — knee × comparison × bone × compartment, map coordinates, world position
    (reference frame), subregion, smoothed and raw Δ;
  - `registration.csv` — residuals per (scan, bone);
  - `cohort_summary.csv`, cohort hotspot scatter figures, `report.md` and `report.meta.json`
    (commit, dirty flag, args, inputs, scan counts), mirroring the existing cohort report.
- Tests: unit tests on phantoms — a pure rigid motion of an unchanged knee gives Δ ≈ 0 and
  recovers the transform; a known thinned patch is recovered in the Δ map and as the hotspot,
  in both bones; the unrolled femoral coordinate is invariant to flexion-like rotation of the
  input; the rim restriction keeps the hotspot off an edge mismatch. Script pure halves
  unit-tested without data, as for the other knee scripts.
- `README.md`, `CLAUDE.md` (scripts section) and the docstrings updated, per the repo
  conventions.

## Left to the builder

Registration algorithm details (ICP variant, trimming, the bone band used), Δ-map cell size,
smoothing kernel / `neighbourhood_mm` default, colour map and colour-scale limits (shared within
a figure; symmetric about 0), how the reference's subregion borders are drawn on the unrolled
femur, the background used for the cohort hotspot scatter, the registration-residual QC
threshold (from the data, not from Δ), and caching of per-scan measurements (~20 s per scan,
113 scans).
