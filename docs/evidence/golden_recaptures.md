# Golden re-capture log

Every deliberate re-capture of a committed golden, with what moved it and why. Moved verbatim out
of `CLAUDE.md`, which keeps the capture rule, what the knee-cartilage goldens do and do not pin,
and the one standing instruction (`knee_rotation_angle_left` must be re-captured if
`KNEE_ROTATION_POSITIVE_SIGN` is ever flipped).

The discipline these entries assume: `MORPH_UPDATE_GOLDEN=1` rewrites **every** golden it
evaluates, not only the failing ones, so a capture names the failing test node ids explicitly and
`git diff tests/golden/` is checked afterwards.

**A note on cross-references.** These paragraphs were written inside `CLAUDE.md`, so an
"above", "below" or "recorded here" may point at a bullet that stayed there rather than at
something on this page. Nothing was reworded, which is the point; the file it came from is
`CLAUDE.md` and the two are meant to be read together.

## The 2026-09-10 CT re-capture

The CT goldens were re-captured on 2026-09-10 against the fixed CT femoral-head landmark (`_femoral_head_slab_bound` plus a largest-component, interior-point column start) and the `cone_axis='superior'` default. Six moved, led by `fhc_ct_radius_left` 66.94 -> 28.34 mm: the old sphere was centred ~63 mm down the femoral shaft, so `ccd_ct_left`, `femoral_torsion_ct_left`, `hka_ct_left`, `mad_ct_left` and `subchondral_distance_ct_left` all inherited that error. `case-105094` — the case `tests/data_paths.py` pins as the CT fixture — was the single hip out of 152 NMDID hips whose landmark was broken, so the pre-2026-09-10 CT goldens were meaningless rather than merely different. `hka_ct_left` and `mad_ct_left` then moved a second time, for an unrelated reason: they had *also* inherited the `split_ct_image` crop-affine bug, which placed the knee and ankle landmarks above the femoral head. The two fixes were developed in parallel and re-captured together at the merge, so those two goldens carry both (hka 5.916, mad 21.688 mm); the crop-affine error was by far the larger of the two.

## Mirror equivariance of the right half

`femoral_torsion_right_lee` / `femoral_torsion_right_murphy` (Augsburg PA000001 image-right half) were re-captured when the right side became mirror-equivariant (see Core data flow item 3); every left-side golden is unchanged.

## The seeded consensus head-centre fit

`femoral_torsion_{left,right}_{lee,murphy}` and `femoral_torsion_PA000147_right_lee` were re-captured when the MRI femoral-head centre became a seeded consensus fit (see Conventions); `femoral_torsion_PA001023_left_lee` and every NaKo (isotropic) and CT golden are unchanged.

## Murphy's neck base moving above the lesser trochanter

`femoral_torsion_{left,right}_murphy`, `femoral_torsion_ct_left` and `anteversion_left` (NaKo; `calculate_anteversion` uses the Murphy line) were re-captured when Murphy's neck base moved to the first slice above the lesser trochanter with continuous head and base coordinates; the CT golden also moved because its neck-base end is no longer truncated to an index.

## The CEA search band and the isotropic head centre

The NaKo goldens (every `tests/characterization/test_hip.py` golden except the stochastic cartilage one) were re-captured when the CEA search band became one factor and the isotropic head centre stopped flooring its centroid (see Core data flow item 3): `center_edge_angle` [27.273, 34.807] -> [27.956, 34.101], `alpha_angle_left` [27.003, 28.842] -> [24.466, 29.211], `acetabular_depth_left` 9.734 -> 9.379, `min_distance_left` 2.068 -> 2.322, `subchondral_distance_left` mean/std/min/max 4.213/0.969/2.236/8.124 -> 4.259/0.991/2.236/7.681, `acetabular_anteversion` [14.485, 13.811] -> [14.465, 13.832], `anteversion_left` -1.459 -> -1.417, `ccd_left` [127.513, 130.120] -> [127.394, 129.918], `ccd_right` [127.022, 131.473] -> [127.037, 131.427]. No Augsburg or CT golden moved.

## CEA_UPPER_MARGIN_FACTOR 1.5 to 1.1

`center_edge_angle` [27.956, 34.101] -> [30.123, 34.101] when `CEA_UPPER_MARGIN_FACTOR` went from 1.5 to 1.1 (only the image-left value moves on this fixture). No other golden moved.

## The MRI shaft axis centroids

`ccd_left` [127.394, 129.918] -> [127.435, 129.938] and `ccd_right` [127.037, 131.427] -> [126.809, 131.099] (NaKo) were re-captured again when `get_femoral_shaft_axis` stopped truncating its centroids. No Augsburg golden covers the MRI CCD; none moved.

## The CT shaft axis endpoints

`ccd_ct_left` was re-captured again when `get_femoral_shaft_axis_ct` stopped truncating its physical-mm endpoints to `int32` (152.3307/152.7929 -> 152.3345/152.7966). Truncation toward zero shifted the axis by up to 1 mm depending on each coordinate's sign, i.e. on where the leg sat in the scanner frame; over 9 NMDID cases x 2 sides CT femoral offset rose by +0.44 mm on average (+0.03 to +0.98) and CCD moved by at most 0.15 degrees.

## The no-notch condylar fallback, and the CT verification that followed

**No committed MRI golden moved when the no-notch condylar fallback was fixed** (Core data flow item 3): PA000001 and the NaKo hips resolve their notch, so `tibial_torsion_{left,right}`, `knee_rotation_angle_left`, `femoral_torsion_{left,right}_{lee,murphy}` and the NaKo goldens are unchanged (`-m "not needs_ct"`: 1306 passed, 7 xfailed, 22 deselected).

**The CT goldens were verified on 2026-11-02, when the sshfs mount came back, and two moved**: `case-105094` does take the fallback on both tibial halves, so `tibial_torsion_ct_left` 78.116 → **66.683** and `tibial_torsion_ct_right` 77.525 → **41.966** were re-captured (evidence and the three validation cues are in Core data flow item 3). **`femoral_torsion_ct_left` and every other CT golden are unchanged** — the CT *femoral* line resolves its notch on this case, so it never took the fallback. The `femoral_torsion_ct` test passing also settles one open question from task #10: `case-105094` is **0.76171875 × 0.76171875 × 3.0 mm**, in-plane bit-identical, so the isotropy assert does not fire on it and the CT femoral torsion is not turned into a NaN row here. (That is one case, not the cohort — the 0.8 × 0.8 × 3.0 mm in the survey below was the protocol figure, and the real in-plane spacing is 0.762.) Three goldens were newly captured, for Augsburg legs that do take the fallback: `tibial_torsion_fallback_PA000077_left` 40.077 (previously a `RuntimeError`), `tibial_torsion_fallback_PA000347_left` 39.953 (was 29.49) and `tibial_torsion_fallback_PA000717_right` 44.432 (was 28.15), in `tests/characterization/test_condylar_fallback.py`; the fallback's geometry is pinned on synthetic sections by `tests/unit/test_knee_reference_line.py` (notch resolvable / absent, the posterior-vs-anterior cortex, rotation covariance, a featureless ellipse, the divider at the posterior apex).
