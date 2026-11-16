# NMDID held-out annotation (for the outlier-review threshold fixes)

Manual landmarks on held-out NMDID legs, used to choose — and separately to quote — the thresholds of
the outlier review's open defects #4 (lesser-trochanter distance bound), #6 (fixed-mm hip crop) and
#7 (knee crop end); see [`../nmdid_outlier_review.md`](../nmdid_outlier_review.md). Set up 2026-11-16;
annotation by the user, not started at the time of writing.

## Files

| file | what |
|---|---|
| `reviewed_cases.txt` | the 84 cases of the 2026-11-13 review (supplied by the user), excluded from every set |
| `annotation_sets.csv` | the draw: `order, case, set, repeat` — **the record; do not redraw once annotation has started** |
| `../../../scripts/draw_nmdid_annotation_sets.py` | the seeded draw (seed 20261116); re-running reproduces the CSV byte for byte |
| `../../../scripts/slicer_annotate_nmdid.py` | the 3D Slicer helper that opens each case and saves the points |

Annotations themselves are written next to the data, `nmdid/annotations/<case>_<R|L>.mrk.json`
(repeat pass: `annotations/repeat/`), not into this repo.

## The draw

1409 `usable` cases with both a CT and a segmentation; excluded: the 84 reviewed cases, every
six-digit case id named in `CLAUDE.md` / `docs/` / `scripts/` / `tests/`, and the 46
verification-sample cases (149 ids in all) → 1304 eligible. Drawn: **20 cases (40 legs) for set A**
(thresholds are chosen here), **20 cases (40 legs) for set B** (results are quoted here, never tuned
on), and **5 set-A cases (10 legs) for a repeat pass** at least a week later (intra-rater error).
Whole cases, shuffled into one annotation order, so learning or fatigue does not differ between A and
B. Random, not stratified — so failure rates measured on them transfer to the cohort.

## The protocol (five points per leg)

| label | point |
|---|---|
| `LT_superior` | most proximal point of the lesser trochanter, where its upper border leaves the shaft/neck (the Murphy level) |
| `LT_tip` | the lesser trochanter's apex, most prominent posteromedial point |
| `LT_inferior` | most distal point, where its lower border merges into the shaft |
| `tib_post_med` | most posterior point of the medial tibial condyle |
| `tib_post_lat` | most posterior point of the lateral tibial condyle — **same axial slice** as `tib_post_med`, just below the joint surface, where both posterior margins are fully visible |

Marked on the CT (bone window), segmentation as an outline only; slice views not rotated or
reformatted (the thresholds act in the scanner's axial frame); blind to the pipeline's landmarks and
values; patient side as Slicer displays it (`R`/`L`; image side is derived). A point that cannot be
placed stays unplaced with the reason in its description — those legs are where the code should refuse.
The head centre, tibia top and fibula top are computed from the masks, not marked.
