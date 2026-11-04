# The Murphy femoral-neck QC on whole-leg CT

Backlog task #3, CT half. Regenerate every number here with

```bash
python scripts/survey_ct_murphy_qc.py \
  -i /home/simon/sshfs/hpcproject/workspace_simon/nmdid/verification_sample/ground_truth \
  -o out/ct_murphy_qc
```

(46 NMDID verification cases = 92 image halves, ~10 min on 8 processes over sshfs;
measured at `da8a198`.)

## Why this existed

`murphy_neck_base_qc` was built and its threshold placed on **MRI only** — 2128
Augsburg and NaKo image halves, 1887 of which yield a landmark and 5 of which are
flagged. The sshfs mount holding the NMDID cohort was down throughout that work, so
`CLAUDE.md` recorded three unknowns. All three are now answered.

## Conclusion

**The gate behaves on CT, the untested failure mode does not materialise, and the
flag rate is ~11x the MRI one — which is the gate working, not misfiring.**

| | CT (this survey) | MRI (Augsburg + NaKo) |
| --- | ---: | ---: |
| halves | 92 | 2128 |
| yield a Murphy landmark | **87 (94.6%)** | 1887 (88.7%) |
| no landmark | 5 (5.4%) | 241 (11.3%) |
| flagged, of those measurable | **3 of 87 (3.4%)** | 5 of 1887 (0.3%) |

### 1. The landmark failure rate is *lower* on CT than on MRI

5 of 92, by reason: `base_at_head_level` 2, `trochanter_implausible` 2,
`shaft_not_found` 1. The MRI failures are dominated by `trochanter_implausible`
(109) and `trochanter_no_components` (69); neither dominates here, and
`trochanter_no_components` does not occur at all — unsurprising, since a whole-leg
CT resolves the lesser trochanter far better than a 10 mm-slice hip MRI.

### 2. The hard one-voxel raise does **not** fire

`get_femoral_neck_base` raises when the axis is under one in-plane voxel
(`axis_unresolved`). It never fired on the 1887 MRI halves, and `CLAUDE.md` flagged
it as **a new, untested failure mode on CT**, where the voxels are smaller while the
neck is the same size. It fires on **0 of 92** halves here. The shortest axis in the
sample is 7.90 voxels — close to the MRI minimum of 6.4 and nowhere near 1. So the
concern was reasonable and is answered negatively.

### 3. The three flagged halves

| case | side | axis (mm) | axis (voxels) | head clearance (mm) | flag |
| --- | --- | ---: | ---: | ---: | --- |
| `case-105094` | left | 10.46 | 13.73 | +40.88 | `murphy_axis_short:10.46` |
| `case-133286` | right | 21.98 | 28.48 | **−0.82** | `murphy_base_in_head:0.82` |
| `case-158269` | right | 7.41 | 7.90 | +0.82 | `murphy_axis_short:7.41` |

**`case-105094` is the CT test fixture.** Its image-left half — the one every
`*_ct_left` golden is measured on — carries a degenerate Murphy neck: a 10.46 mm
axial-plane axis, 13.73 in-plane voxels against a 14.30 threshold, so one voxel of
landmark error tilts it by more than 4°. The CT femoral torsion is Murphy-based
(`_ct_proximal_landmarks` is shared by `get_proximal_reference_line_ct` and the QC),
so **the `femoral_torsion_ct_left` golden rests on a neck this QC calls degenerate.**
The golden is still the right value to pin — it is what the code produces, and the
flag is about conditioning rather than about the number looking wrong — but it should
not be quoted as a well-conditioned reference, and a fixture with a sound neck would
be a better one if another NMDID case can serve.

`case-133286` right is the first case **anywhere** whose neck base falls inside the
fitted head sphere in 3-D by the full-distance test: −0.82 mm. The MRI minimum was
+5.99 mm *outside*, with the single NaKo exception caught by the axis gate as well.
So `murphy_base_in_head` earns its place on CT independently.

### 4. The 4° tolerance is in a stable region on CT; 2.5° would not be

| tolerance | threshold (voxels) | halves flagged |
| ---: | ---: | ---: |
| 2.5° | 22.90 | **24 of 87 (28%)** |
| 3.0° | 19.08 | 4 |
| **4.0°** | **14.30** | **2** |
| 5.0° | 11.43 | 1 |

On MRI, 2.5° flagged 161 of 1887 (8.5%); on CT it would flag 28%. The CT axis
distribution has a gap — the two shortest are 7.90 and 12.91 voxels, then p5 is
19.36 — so 3° to 5° all pick out essentially the same one or two halves, and the
shipped 4° sits inside that plateau rather than on a cliff. **This is not
independent validation of 4°**: the threshold was placed on the MRI cohorts'
distribution, and this is one more cohort it happens to suit.

### 5. The threshold's unit is consequential in principle, inert here

The gate is **in in-plane voxels**, which is the right unit for its actual argument
(one voxel of landmark error tilting the axis past tolerance is a statement about
sampling, not about anatomy). The consequence is that in millimetres it is not one
threshold: over this sample's spacings (0.707–1.363 mm, and 0.623–1.367 over the
whole cohort — see `scripts/survey_ct_spacing.py`) the 14.30-voxel gate is anywhere
from **10.1 mm to 19.5 mm**. The same anatomical neck can therefore be flagged on a
coarse reconstruction and not on a fine one.

Measured, that does not bite here: an absolute-millimetre rule set at the sample's
median spacing (13.2 mm) flags **exactly the same two halves**, with **zero**
disagreement, because nothing sits in the ambiguous band. Worth re-checking on any
cohort whose axis distribution lacks that gap.

## Scope — read this before quoting the rate

These are the **46 verification cases (92 halves)**, which is the sample every other
CT figure in `CLAUDE.md` uses. It is **not** the 1474-case cohort
`process_nmdid.py` runs over, and a 3.4% flag rate on 87 halves is 3 events: its
confidence interval is wide. The full cohort is now reachable (the mount is back)
and the script takes a directory, so the cohort-wide rate is a run away — at roughly
32x this survey's cost.
