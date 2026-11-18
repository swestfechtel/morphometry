# NMDID held-out annotation: results (set A 2026-11-17, set B 2026-11-18)

Per-leg tables: [`results_set_A.csv`](results_set_A.csv) (choose), [`results_set_B.csv`](results_set_B.csv)
(quote), written by `scripts/analyse_nmdid_annotations.py` at the commit that shipped #7. Legs are
prepared exactly as `scripts/process_nmdid.py` prepares them (`load_case` + `split_legs`, so with the
group-A and #9 fixes); a right image half is reflected. Set A was run under the **old** knee crop
(its `current_*` columns are the legacy rule, `x15_*` the shipped one); set B was run once, after the
threshold was frozen in the code (`legacy_*`, `production_*`; `x15_*` equals `production_*` on all
39 legs, a check that the script reproduces production).

## The annotations themselves

80 legs, all files present (the repeat pass is not done). Every placed point lies on or within ~2 mm
of the right bone label; both tibial points of a leg are on one axial slice; the three LT points are
in craniocaudal order. `R`/`L` is the patient side (R in the image-left half), as the helper intends.
Unplaced, all in **set B**: 167053 both LTs (broken bone), 193351 L LTs (artefacts), 176282 R tibia
(knee out of frame). **Medial/lateral probably swapped** on 160054 R (the fibula lies on the side
marked medial) and possibly 137844 L (fibula directly posterior to the tibia — a strongly rotated
leg); neither matters below, since the condylar line is compared as an undirected line.

## #7 knee crop end -- shipped: `max(fibula top, tibia top + 15 mm)`

Angle between the pipeline's tibial posterior condylar line and the annotated one (undirected, degrees):

| set A (choose), 40 legs | median | p90 | max | legs > 10° |
|---|---|---|---|---|
| fibula top (legacy) | 2.02 | 7.00 | 62.2 | 2 |
| + 12 mm | 2.02 | 6.96 | 62.2 | 2 |
| **+ 15 mm** | **1.85** | **4.95** | 38.6 | **1** |
| + 18 mm | 2.17 | 5.48 | 28.2 | 2 |
| + 21 mm (the review's "~20") and beyond, to 60 | 2.32–2.34 | 6.02 | 66.7 | 3 |

| set B (quote), 39 legs | median | p90 | max | legs > 10° | failed |
|---|---|---|---|---|---|
| fibula top (legacy) | 1.98 | 9.18 | 72.2 | 4 | 1 (132907 R, `ValueError`) |
| **+ 15 mm (shipped)** | 2.04 | 8.33 | 72.2 | 4 | 0 |

**What the quote supports, and what it does not.** On set B the rule is **roughly neutral**: the
median and the >10° count are unchanged, the p90 is slightly lower, one leg that failed is now
measured (132907 R, at 18.6° — a 7.8-voxel line, which the condylar QC gate flags), and one leg got
much worse (165717 R, 19.8° → 64.8°; its legacy line was 2.8 voxels long, so neither value was
usable). 14 of 39 set-B legs (15 of 40 on A) read their line on a different layer — **do not pool**
NMDID tibial torsion, mechanical-axis deviation or HKA rows across this change (the MAD/HKA knee
centre is the tibia's centroid inside the knee crop, so it moves too). The set-A advantage of 15 mm
rested on one or two legs; set B does not reproduce it as an improvement, only as no harm plus one
rescued failure. **Longer is not safer**: from ~18 mm on, the largest-convex-area layer often lands
one slice below the plateau, where a single slice can swing the line by 25–65°. That fragility of
the layer choice — not the crop end — is what produces the large errors on random legs, and it is
the thing to look at next for tibial torsion.

## #4 LT lower bound and #6 hip crop -- left unchanged

(x, z) distance from the head centre to the annotated LT, measured as `check_plausibility` measures it:

| | LT tip min / median / max (mm) | LT superior min (mm) | crop needed to reach `LT_inferior` | 30 % crop |
|---|---|---|---|---|
| set A, 40 legs | 42.3 / 51.7 / 63.4 | 35.0 | 84–111 mm | 111–150 mm, covers every leg |
| set B, 37 legs | 39.2 / 48.5 / 69.4 | — | max 105 mm | covers every leg |

On set A the Murphy search found a base on **all 40** legs, so the 40 mm bound and the 30 % crop
never bit and set A could not choose either. **Correction (2026-11-18):** in conversation this was
read as a hint that the group-A fixes had removed much of the Murphy tail; set B refutes that —
5 of 37 annotated set-B legs fail (193351 R, 128937 L, 137844 L, 176282 R: "no plausible
candidates"; 165717 L: base at head level), and a sixth (132907 R) returns a base 30 mm above the
annotated LT. Only one of those (193351 R, LT tip 39.2 mm) has its LT below 40 mm, so lowering #4
would not repair most; they look like #3/#5 (component count, pose). These set-B figures are a
**quote of the current code, not a basis for tuning** — set B is spent for #4/#6. Choosing them needs a
new held-out sample drawn from the failing legs of the next NMDID run (excluding the reviewed cases
and these 40), where the bounds actually matter.

Murphy base vs annotated `LT_superior`: median +6 mm (the base lies two slices *below* the annotated
top on both sets). A definitional offset — the search's "superior extent" is that of the component on
the first medial sagittal slice where the LT separates, not of the whole trochanter — not a failure.
