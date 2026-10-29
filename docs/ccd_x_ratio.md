# What the MRI CCD's `x_ratio` is, and why the obvious correction is wrong

Backlog task #6. Regenerate every number here with

```bash
python scripts/analyse_ccd_x_ratio.py -o out/ccd_x_ratio
```

(50 Augsburg reader-study patients, both legs = 100 image halves, ~2 min on 12
processes; measured at `fddac1a`.)

## The item as it was filed

Every torsion pipeline computes

```python
x_ratio = abs(hip.spacing[2]) / 2 * abs(hip.spacing[0])
```

which is not a ratio of anything: it is `mm / 2 x mm`. The slice aspect is
`spacing[2] / spacing[0]`. On the Augsburg hips (1.09375 mm in plane, **10.0 mm**
slices) the pipelines therefore pass **5.469** where the aspect is **9.143**, and
the item was to correct it.

## Conclusion

**Do not correct it. It is not a slice aspect and should never have been named
like one — it is an empirical conditioning parameter for the femoral-neck
landmark, and the dimensionally correct value measurably degrades the CCD.**

Changing 5.469 to 9.143 moves the CCD by a median 4.1-5.5 degrees (max 19.0) and
is worse on every criterion available:

| | pipeline 5.469 | true aspect 9.143 |
| --- | ---: | ---: |
| within-patient left/right Pearson (no reader involved) | **+0.770** | +0.393 |
| mean \|L-R\| | **3.60°** | 5.61° |
| bias vs reader mean, patient left / right | **+3.73 / +4.81** | +3.95 / +7.77 |
| MAD vs reader mean, patient left / right | **5.38 / 7.26** | 7.00 / 9.62 |

## Why — the mechanism, not the correlation

`hip.get_femoral_neck_center` scales the point cloud by `(1, 1, x_ratio)` and hands
it to `_femoral_neck_points_and_center`, which samples a **sphere** shell of radius
`r .. FEMORAL_NECK_SPHERE_OUTER_FACTOR * r` around the head centre and intersects it
with the femur voxels (KD-tree radius 2, in that same space). A larger `x_ratio`
shrinks the shell's reach **in slices**. With 10 mm slices and a head radius of a
median 17.9 in-plane voxels, that reach is the whole measurement:

| `x_ratio` | neck voxels (median) | neck slice span (median) | halves spanning ≤2 slices |
| ---: | ---: | ---: | ---: |
| 1.000 | 1235 | 11 | 0 / 100 |
| 3.000 | 784 | 7 | 0 / 100 |
| 4.000 | 665 | 5 | 0 / 100 |
| **5.469** | **583** | **4** | **0 / 100** |
| 7.000 | 571 | 3 | 3 / 100 |
| **9.143** | **556** | **3** | **30 / 100** |
| 12.000 | 534 | 2 | 86 / 100 |

At the true aspect, **30 of 100 halves** get a neck point cloud spanning two slices
or fewer, so the neck centroid's superior-inferior coordinate is pinned to one or
two 10 mm sample positions instead of interpolated across a band. The wrong factor
is buying conditioning by over-reaching in z. That is a defect of the landmark, not
a virtue of 5.469.

## Two things this settles

**The whole item is confined to the CCD.** Femoral torsion is bit-identical between
the two factors — maximum absolute difference **0.000e+00 degrees** over the 190
(half x method) measurements that resolve at both. On the non-isotropic MRI path
`x_ratio` reaches nothing but the femoral-head slab bound, which does not bite on
this geometry. `calculate_anteversion` already derives `spacing[2] / spacing[0]`
itself. The alpha angle and acetabular depth take the array-only head-centre
fallback `(1, 1, x_ratio)`, whose centre sits ~5 mm (max 14) in z from the spacing
fit — but their only production callers are the **isotropic** NaKo scripts, which
pass neither a bad `x_ratio` nor anisotropic data, so that error is latent and has
never been realised. `calculate_min_distance_between_femoral_head_and_acetabulum`
has no production caller at all.

**The CCD's absolute level is an artefact of this factor and no value of it fixes
that.** Over the sweep the cohort mean runs 153.5° (at 1.0) down to 138.4° (at
5.469) and then flattens around 139-140°. Adult CCD is usually quoted as 125-135°
and these readers sit at 129.9-136.3°. **The minimum over the whole family is
138.4°, at the value already shipped** — so the ~138° that
`docs/augsburg_reader_bias.md` flags cannot be explained, or repaired, by
`x_ratio`. It belongs to the neck landmark itself.

## What this does *not* license

**It does not license picking a factor from these numbers**, and the script says so
in its own docstring. The two criteria disagree — left/right agreement peaks near
3-4 (r +0.846 at 3.0, mean \|L-R\| 3.14° at 4.0) while the reader bias and MAD are
smallest at 5.469 — and both are computed on the same 50 patients any chosen value
would then be quoted against. That is the calibration-on-the-evaluation-set mistake
`CLAUDE.md` already records for `CEA_UPPER_MARGIN_FACTOR`. The sweep is a diagnosis
of the landmark, not a tuning curve.

## The real fix, left open

The neck centre should not depend on `x_ratio` at all. Sampling the shell in
millimetres with a slice-aware thickness (or reconstructing the neck surface across
slices rather than intersecting voxels with a sphere) would remove the parameter and
is the only route to the CCD's absolute level. It moves every MRI CCD ever produced
and needs its own backlog item, evidence and golden re-capture.
