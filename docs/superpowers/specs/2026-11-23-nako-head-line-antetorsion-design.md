# NaKo antetorsion against the inter-head line — design

Status: approved 2026-11-23 (short design; small change).

## Intent

The NaKo reference readers measured femoral antetorsion (Murphy) against the line connecting the
left and right femoral head centres. The pipeline (`calculate_anteversion`,
`morphometry/measurements/hip.py`) measures the same Murphy neck axis against the image x-axis, so
pelvic rotation in the scanner enters our value but not theirs. A new variant measures against the
inter-head line so the evaluation compares like with like.

## Goals / non-goals

- Goal: a variant of the NaKo antetorsion using the inter-head line as reference, reported per side.
- Goal: if the inter-head line cannot be established, fall back to the image x-axis (the existing
  convention) rather than returning NaN. *(user requirement)*
- Non-goal: changing `calculate_anteversion` or the existing `AT_murphy` column. No golden of the
  existing function moves; no do-not-pool warning is needed for `AT_murphy`.
- Non-goal: touching the Augsburg / CT femoral-torsion pipelines (they use the posterior condylar line).
- Anti-goal: choosing any parameter against the 60 NaKo reader cases.

## Approaches considered

1. **New two-sided function on the whole mask (picked).** Like `calculate_center_edge_angle`: split
   into halves, get both Murphy lines (head centre + neck base) via
   `get_proximal_reference_line(method='murphy')`, shift the right half's landmarks by `+half` on x,
   compose with `geometry.signed_axial_angle`. Head centres come from the same call as the neck
   line, so reference and neck axis share landmarks.
2. A `reference=` flag on `calculate_anteversion` — rejected: that function sees one half only, so
   it would need the other head centre passed in, pushing the pairing into every script.
3. Post-hoc rotation of the existing angle in the scripts by the inter-head bearing — rejected: the
   existing angle is folded to acute with an image-frame sign, so un-folding it is fragile.

## Design

- `calculate_anteversion_head_line(segmentation_mask, segmentation_label=1, isotropic=False,
  x_ratio=None, plot=False) -> Tuple[float, float, str]` in `morphometry/measurements/hip.py`,
  re-exported from `morphometry.measurements`. Returns `(left, right, reference)` with values for
  the **image** halves and `reference` in `{'head_line', 'x_axis'}`.
- Per half, the angle is `signed_axial_angle(reference_line, head_centre − neck_base, side)` —
  positive = antetorsion, the same sign as `calculate_anteversion` and as femoral torsion.
  With the inter-head line parallel to the x-axis it must equal `calculate_anteversion`'s value
  (pinned by a test).
- Axial plane only: the inter-head line's z component is ignored (the readers measure on axial slices).
- Refuses anisotropic in-plane spacing (`validate_in_plane_isotropy`, `ValueError`).
- **Fallback:** if either side's Murphy landmarks fail, or the two head centres coincide in the
  axial plane, every side whose own landmarks succeeded is measured against the image x-axis
  (`reference='x_axis'`); a side whose own landmarks failed comes back as `NaN`, and if both
  sides fail the function raises `ValueError` (caught by the scripts as NaN). *My call:* the return value always carries `reference`, so a fallback row is never
  silently mixed with head-line rows.
- Scripts: `process_nako.py` and `process_nako_large.py` write a new `AT_murphy_headline` column
  (appended after the historical columns) and add `at_headline_fallback:x_axis` to `qc_flags` when
  the fallback fired. `eval_nako.py` evaluates `AT_murphy_headline` against the readers alongside
  `AT_murphy`.
- Tests: mirror equivariance (reflected mask swaps and preserves values), invariance under in-plane
  rotation of a synthetic pelvis, the parallel-line identity above, the fallback path, and a NaKo
  golden (`needs_nako`).
- Docs: CLAUDE.md (hip section + scripts), `docs/scripts.md`, README.

## Left to the builder

Plot overlay details; exact synthetic-phantom construction for the tests.
