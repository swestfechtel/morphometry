# NMDID outlier review: code defects found (2026-11-13, open)

After the NMDID re-run at `6cec5b1` (`results/nmdid_6cec5b1/`, superseded by `nmdid_d0f233f`, same
values), four subagents reviewed 84 outlier cases visually and re-ran their landmarks exactly as
`scripts/process_nmdid.py` prepares the legs. The data-side causes (field of view, pose, crossed legs,
dislocations, mislabelled arms) are now **flagged** — see
[`qc_gates.md`](qc_gates.md#nmdid-extent-and-plausibility-flags-2026-11-13). The **code defects** below
are **not fixed**. Each was re-read in the code at `d0f233f`; line numbers are as of that commit.

**Fixing any of them moves measured values**, so each fix needs a golden re-capture
([`golden_recaptures.md`](golden_recaptures.md)), a new NMDID run, and a do-not-pool note.
Thresholds the reviewers suggested (lesser-trochanter bound ~30 mm, tilt 20°, head radius 15–40 mm)
were **read off the reviewed cases** — do not ship them without choosing them on held-out data
(CLAUDE.md, "a parameter is never chosen against the data it will then be quoted against").

The review's scripts, renders and per-case tables were in this session's scratchpad on the cluster and
are not in the repo. Case ids below are enough to reproduce: `scripts/process_nmdid.py`'s `load_case` +
`split_legs` + `split_ct_image` give exactly the production legs.

## Femoral torsion (CT Murphy): the lesser-trochanter search, `morphometry/femur.py`

The most important group: on a 44-leg sample of `murphy_landmark_failed` legs, most masks are sound
and the search misses. `base_at_head_level` failures are almost all search defects (8 of 10 sound);
`shaft_not_found` is mostly a consequence of an earlier miss; `trochanter_implausible` is mixed;
`trochanter_no_components` and `ValueError` are mostly correct refusals (bad mask, FOV, missing femur).
Cohort counts at `6cec5b1`: trochanter_implausible 143, base_at_head_level 132, shaft_not_found 33,
trochanter_no_components 19, ValueError 10; CT femoral torsion is NaN on 433 of 2948 legs.

1. **Smaller component taken (line ~815):** `smaller_component = min(connected_components, ...)`
   is chosen even when the plausible candidate is the larger one. In 9 of 10 traced legs the
   "trochanter" was a 1–84-voxel speck 15–24 mm from the head. Cases: 125709 R, 131956 L, 111791 L,
   130790 R, 161675 L.
2. **Speck guard is MRI-only (line ~787):** components under 10 voxels are skipped only when
   `isotropic=True`; CT passes `False`, so 1-voxel specks count.
3. **Exactly two components required (line ~784):** `len(connected_components) == 2`. The hip crop is
   30% of the visible femur (`_ct_proximal_landmarks`, around line 1490), so on a full femur it pulls the
   distal shaft into the medial sagittal slices and the true lesser-trochanter slices read as three
   components (head + LT at 42–54 mm + shaft) and are skipped. Cases: 111791 L, 130790 R, 161675 L,
   181824 R.
4. **Lower distance bound 40 mm (line ~772):** `75 > distance_world > 40`; on four sound legs the LT
   sits at 36.0–39.8 mm (125709 L, 139552 R, 131956 L). The fix is not "30" — choose it on held-out data.
5. **Pose:** the sagittal-slice scan assumes neutral hip ab/adduction and flexion; on the frequent
   abducted / adducted / flexed post-mortem legs it cuts the neck obliquely (105669, 139618, 160816 L,
   176386 R, 181483 L). After a miss the selected slice is lateral and the shaft search only steps
   further lateral, giving `shaft_not_found`. The structural fix is a femur-aligned frame (rotate
   coordinates so the shaft is vertical — consistent with "rotate coordinates, not the image").
6. **Crop as a fraction of the visible femur:** on a truncated femur (150555) the 30% crop is ~54 mm
   and ends above the LT; on a full femur it is too long (defect 3). A fixed-mm crop below the head
   addresses both.

A what-if with 2 + 3 + a lower bound gave bases 29–48 mm below the head on 9 sound legs (mask-only,
not visually confirmed); 139618 R and 181483 L still failed.

## Tibial torsion

7. **Knee crop ends at the fibula top (`morphometry/image_io.py:552`, `split_ct_image`):**
   `knee_end = min z of fibula`. At 3 mm slices this leaves 2–6 tibia slices in the outliers
   (median 3) against 4–9 in controls (median 6); the reference layer is a partial plateau or the
   eminence, the notch search fails, and the line is short or random. Behind 9 of 16 sampled
   |TT| > 70° legs (106172 R, 106371, 122783 R, 138142 R, 164164 R, 188320 R, 189047 L) and several
   failures (122783 L, 141066 L, 188197 R, 145316 R). These are the legs `tibial_condylar_line_short`
   catches. Suggested: end at max(fibula top, tibia top + ~20 mm).
8. **Inverted knee crop:** when a mislabelled fibula (arm bone) sits above the patella,
   `knee_end < knee_start` and the crop is empty/inverted, failing later with a raw IndexError
   (140223, 149024 R). Raise a named `ValueError` when the crop end is not below its start.
9. **Ankle centroids over all components (`morphometry/tibia.py:52-63`, `get_distal_reference_line`):**
   the tibia and fibula centroids average every voxel of the label on the slice, including the other
   leg's fragments. 188320 L: −79.1° → 37.9° using largest components only. Also legs crossing at the
   ankle (151348, 175037 L, 145316 L). A fibula area of 0 gives a NaN centroid.
10. **Array sized on the wrong axis (`morphometry/tibia.py:22`):** `np.zeros(segmentation_mask.shape[0])`
    for a loop over `shape[2]`; on an empty tibia the layer comes out as 255 and line 58 raises
    IndexError.

## Center-edge angle, `morphometry/measurements/hip.py`

11. **Edge below the head accepted (`_lateral_acetabular_edge_point`, line ~881; angle at ~1006/1012):**
    the band only bounds the search from above, so an edge at or below the head centre is accepted,
    and `calculate_angle_between_vectors` is unsigned (arccos), so an uncovered head reads as a large
    angle (dislocations 112453 R, 131186 R, 148766 R, 151561 L, 151687 L: CE 108–165°). Now flagged by
    `center_edge_not_above_head`; the measurement should refuse instead. Separately, the edge rule
    (most inferior voxel of the most lateral column) lands only ~10 mm above the centre on some sound
    hips (107488 R, CE 73°).

## Error handling

12. **Raw IndexError / ValueError on empty sections (`morphometry/utils.py:167`, `:327`, `:367`).**
    `process_nmdid.py` catches IndexError, but `morphometry/docker/compute_torsion.py` catches only
    `(RuntimeError, AssertionError, ValueError)`, so an IndexError there could abort the torsion
    container instead of writing one NaN. **Inferred from the code, not reproduced.**

## Script-side (not library)

13. **`clean_leg` keeps arms that reach below the pelvis top** (`scripts/process_nmdid.py`): it drops
    only components lying *entirely* above the pelvis, so an arm beside the thigh survives labelled
    femur/tibia (131874 L, 172742, 174848 — on 174848 L the head fit lands on the humerus, r = 129 mm).
14. **Femur specks under 10% of the bone are kept** (the `qc_flags` 10% speck rule only governs the
    `pieces` flag): a femur sliver at the split column seeds the CT head sphere fit (124799 L r = 80 mm,
    124837 L r = 214 km, 151446 L). Now flagged by `femoral_head_radius_implausible`; `clean_leg` could
    drop them.
