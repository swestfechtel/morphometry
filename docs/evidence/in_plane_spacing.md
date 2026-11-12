# In-plane voxel spacing and the axial-plane angles (task #10)

The spacing surveys behind `geometry.validate_in_plane_isotropy`: which cohorts are square in
plane, which are not, and what the shear would cost where it is only hypothetical. Moved verbatim
out of `CLAUDE.md`, which keeps the precondition itself, where it is checked, the one path that
can turn a cohort CSV cell into NaN, the argument against recomposing in physical coordinates,
and the two measurements of the same class that are unfixed because they are unreachable.

**A note on cross-references.** These paragraphs were written inside `CLAUDE.md`, so an
"above", "below" or "recorded here" may point at a bullet that stayed there rather than at
something on this page. Nothing was reworded, which is the point; the file it came from is
`CLAUDE.md` and the two are meant to be read together.

## The cohort spacing surveys, and what the shear is worth

**Measured spacing survey** (header zooms permuted into the post-LPI axis order — the reorientation can move the slice axis *into* the plane): Augsburg MRI **5715/5715** `.nii.gz` under `preprocessed/` square in plane (values 1.09375, 1.145833, 1.25, 1.302083, 1.640625 mm; slice 4–10 mm — but note that list is the **union over regions, and the per-region sets differ**, which is what makes a spacing-dependent hazard region-specific: the 952 **knee** segmentations carry `{1.09375: 949, 1.145833: 1, 1.302083: 1, 1.640625: 1}` and the 951 **hip** ones `{1.09375: 949, 1.145833: 1, 1.25: 1}`, so 1.25 mm occurs on no knee mask and the coarsest knee spacing, 1.640625, occurs on no hip mask — which is exactly why the millimetre-ordering defect recorded under the Murphy gate is *demonstrated* on the condylar line and only *possible* on the Murphy axis) and 450/450 in `reader_study_augsburg/` at 1.09375, NaKo sample 1.0 mm isotropic 60/60 — every one of them with its two in-plane zooms **bit-identical**, so nothing legitimate sits near the tolerance. Whole-leg CT **is now surveyed, and the risk is zero** (`scripts/survey_ct_spacing.py`, 2026-11-02, headers only so it runs over the cohort in minutes):

**1520 of 1520** NMDID files (1474 cohort segmentations + the 46 verification ground truths) are in-plane square with their two post-LPI zooms **bit-identical**, and **1520 of 1520** reorient to LPI (all are raw LPS), so there is no CT analogue of the `dfg_knees` non-LPI cases either. No case would have its CT femoral torsion turned into a NaN row by the assert — that worry is closed. What the survey *does* overturn is the spacing itself: the **0.8 × 0.8 × 3.0 mm quoted here was the acquisition protocol and matches not one single file**. The real in-plane spacing is per-case (reconstructed FOV over a fixed matrix):

**259 distinct values from 0.623 to 1.367 mm**, with the slice axis 3.0 mm throughout, so the slice aspect runs 2.19–4.82. That per-case variation has a consequence for the Murphy gate — see the `murphy_axis_short` bullet, whose threshold is in in-plane *voxels*.

**This is hypothetical, not published: no sheared value has ever been produced in this repo.** `calculate_knee_rotation_angle`'s only non-test callers are `scripts/process_augsburg_{reader_study,large_dataset}.py` (Augsburg, square) and an exploratory notebook; the `dfg_knees` scripts are cartilage-only, and `dfg_knees` has no ankle label, so the tibial torsion cannot reach it either.

**No value moved**: the composition arithmetic is untouched (ASTs of `signed_axial_angle`, both torsion composers, `_knee_rotation_angle`, `_mirror_landmarks`, `_proximal_angle` and the fold helpers are byte-equivalent to the previous commit with docstrings stripped), the PA000001 femoral (Lee and Murphy) and tibial torsion are bit-identical with and without the spacings and to their pre-change values, and no golden moved (the NaKo `anteversion_left` golden included — NaKo is isotropic).
