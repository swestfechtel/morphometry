# The three QC gates: prevalence surveys and verification

The evidence behind `murphy_neck_base_qc`, `lee_neck_axis_qc` and `knee_condylar_line_qc`: the
cohort surveys that placed each threshold, the specificity checks, and how each gate was shown to
move no value. Moved verbatim out of `CLAUDE.md`, which keeps every flag name and threshold, the
never-raises contract, the wiring and the cautions that travel with the numbers.

Two things to carry forward. A threshold's *transferability* is a property of the quantity, not of
the gate design: Lee's `a` is a dimensionless ratio and compares across cohorts, where a condylar
line length is in voxels and does not. And a gate should be shown non-mutating at **runtime**, not
only by diff — measure, call the gate, measure again, compare bitwise.

**A note on cross-references.** These paragraphs were written inside `CLAUDE.md`, so an
"above", "below" or "recorded here" may point at a bullet that stayed there rather than at
something on this page. Nothing was reworded, which is the point; the file it came from is
`CLAUDE.md` and the two are meant to be read together.

## The Lee ring gate: zero-movement verification and test provenance

**Zero movement, checked rather than asserted**: `git diff tests/golden/` is empty (and the goldens pin both femoral torsion methods, the tibial torsion, the alpha angle and the NaKo hip family), the full fast suite went 1794 -> **1850 passed, 1 xfailed, 0 failed** (+56 tests, nothing broken; 52 of them the two new Lee modules, confirmed by `--collect-only`, and 4 in `test_murphy_qc_scripts.py`). Provenance of that figure, since three runs are quoted in the history:

**1848** was measured independently by the lead and by the reviewer on the tree *before* a `needs_nako` characterization test was added, and **1850** is the final tree with it — the implementer's run, consistent with 1848 + 2, and the one pre-existing test that changed (`test_compute_torsion_pipeline.py`) changed *additively* — a new import, the new key, and that key added to the expected-keys set, with no existing expected value touched. Pinned by `tests/unit/test_lee_neck_axis_qc.py` and `tests/characterization/test_lee_neck_qc.py`. One reporting caveat the docstring states: the printed `<a>` is rounded to three decimals while the comparison is not, so read the string as a label and not as the number the gate saw.

## The Lee ring gate: independent corroboration on PA000346

**An independent corroboration found end-to-end, which is the strongest single thing in the gate's favour.** Run through the production wrappers on the reader-study fixture, `PA000346` image-left returns `lee_ring_within_head_radius:0.768` **and** `murphy_landmark_failed:trochanter_no_components` — i.e. the one half the new gate flags is also the half whose *Murphy* landmark search fails outright, on a completely separate landmark path (trochanter detection, no shared code with the notch or the ring). So that hip is independently bad rather than the victim of an unlucky Lee layer, which is exactly the corroboration a conditioning gate normally cannot get. Its image-right half and both halves of `PA000001` return `''` from both gates.

## The Lee ring gate: four qualifications from implementation and review

**Four points from the implementation and review that qualify or strengthen the above, and the first is the most important.** (1)

**The tie among 0.9 / 1.0 / 1.1 rests on the Augsburg cohort alone**: NaKo flags **0 of 120 at all three** candidates and the reader study the same **1 of 100** at all three, so neither replication discriminates between them — they establish that `a < 1` is rare, not that 1.0 is the right member of the family. "Data-broken tie" reads as more evidence than it is unless that is said. What breaks the tie is specifically that **1.1 lands 0.0048 from the next observed value and inside a cluster of five halves between 1.1048 and 1.1232**, so it separates nothing and its flag set is one re-measurement away from changing, while 1.0 and 0.9 both sit in the empty interval and flag the same four. (2)

**The gate is robust to the head fit being under-sized on the very case that motivates it**, which is worth knowing given the slab-depth finding above: `PA000346` image-left's fitted head radius is **13.4 mm**, below the 18-26 mm adult range `MURPHY_NECK_BASE_MIN_HEAD_DISTANCE_MM`'s own comment cites, and since `a = d / r` a too-small `r` makes `a` *larger* — so correcting the head fit would push that half **further** below the threshold, not across it. (3) The anisotropy guard is necessary by measurement, not by analogy: one anatomy defined in mm, sampled onto three in-plane grids through the real path, gives `a` = 0.771 at 1.0 × 1.0 (flagged), 0.848 at 1.0 × 1.4 (flagged) and **1.188 at 1.4 × 1.0 (clean)** — same anatomy, verdict flipped — while an isotropic control (0.5 × 0.5 -> 0.714) moves `a` without flipping it, which is what separates shear from mere discretisation. It also closes a real asymmetry the review found: `measurements/femur.py:142` already validates the hip image's own spacing, so without the guard the wrapper was the only half of a matched pair silently returning "sound" for a leg whose angle its own measurement refuses by name. (4)

**There is no CT counterpart**, because `get_proximal_reference_line_ct` implements only the Murphy line; that is a documented absence, not an omission.

## The Murphy gate: prevalence across four cohorts

**87 of 92 halves yield a landmark (94.6%, a better rate than MRI's 88.7%)** and 5 do not (`base_at_head_level` 2, `trochanter_implausible` 2, `shaft_not_found` 1; `trochanter_no_components`, the second-largest MRI reason, does not occur at all — a whole-leg CT resolves the lesser trochanter far better than a 10 mm-slice hip MRI).

**3 of the 87 are flagged (3.4%, against 0.3% on MRI)**: `case-105094` image-left `murphy_axis_short:10.46`, `case-158269` image-right `murphy_axis_short:7.41` (the sample's shortest axis, 7.90 in-plane voxels) and `case-133286` image-right `murphy_base_in_head:0.82` — the first case anywhere whose neck base falls **inside** the fitted head sphere on the 3-D test, where the MRI minimum was 5.99 mm outside, so that gate earns its place on CT independently.

## The Murphy flag unit defect: millimetres did not order severity

**`murphy_axis_short`'s printed millimetre value did not order severity — a structural defect of the flag string, not of the threshold (found 2026-11-06, FIXED the same day; the shipped flag prints in-plane voxels, see the end of this paragraph).** It compares a length in in-plane **voxels** and prints **millimetres**, so wherever a cohort has more than one in-plane spacing the two orderings disagree. Augsburg-large's 951 hip files carry three distinct in-plane spacings (949 at 1.09375, one at 1.145833, one at 1.25 mm, all in-plane square), making **[15.64, 17.88] mm an ambiguous band**: an unflagged half at the finest spacing prints at least 15.64 mm while a flagged half at the coarsest can print up to 17.88, and with a Murphy axis median of ~31 mm that band is populated.

**Whether it is *realised* on the 3 flagged Augsburg-large halves was recorded here as unmeasured, needing the full QC pass at +3.1 s per case; it is now settled by a header read instead, and the answer is NO (2026-11-09).** The reduction is the whole trick: a violation needs a *flagged* half to print more than the least an *unflagged* half can print, and both bounds are `14.3007 × spacing`, so only the flagged halves' **spacings** matter and the axis lengths are already in this file. All three flagged halves — `PA001127` image-left and image-right, `PA000445` image-right — sit at the finest spacing 1.09375, so they print 6.97, 12.76 and **14.95 mm** against a 15.64 mm floor for any unflagged half:

**0.69 mm of headroom, no violation**. (The 12.76 against the 12.77 recorded above is recomputing from the rounded voxel figure rather than the unrounded axis; it is 0.01 mm and changes nothing.)

**But read the headroom, because the cohort escapes by luck and not by margin**: the two cases that carry the odd spacings are `PA000867` (1.145833) and `PA000872` (1.25), and neither is flagged. A flagged half at 1.145833 violates from **13.65** voxels and one at 1.25 from **12.51** — and `PA000445` image-right is **13.67** voxels, i.e. its axis length is already inside the violating range for *both* odd spacings (it would have printed 15.66 and 17.09 mm). So the defect missed this cohort because of which two of 951 cases happen to be off-protocol, not because any flag sits far from the band. That makes it a demonstrated hazard and not a demonstrated wrong row — the conclusion recorded before, now with the measurement behind it rather than in front of it.

**The same construction was measured end to end on the condylar line, where it IS realised: 51 unflagged lines are shorter in millimetres than the longest flagged one** (16.41-21.24 mm against `PA000435`'s 21.39 mm at a 1.640625 mm spacing), the knee cohort having four spacings reaching coarser than the hip's.

## The cost of re-resolving the Murphy landmark, on MRI and CT

**Measured 2026-11-09, and the answer is: the mechanism is right, the cost is not worth it, so the follow-up is CLOSED as not worth doing.** The structural half of the claim holds exactly — on 8 NaKo halves the line resolve is **94.4%** of the whole `get_murphy_neck_qc` call (0.279 s of 0.294 s; the verdict itself is 0.014 s), so an out-parameter really would recover essentially all of it.

**The CT cost is now measured (2026-11-09): `get_murphy_neck_qc_ct` is a median 1.308 s per image half, i.e. ~2.62 s per case**, timed over all 92 verification halves. For scale on the same halves, `split_ct_image` is 0.531 s and the whole tibial torsion 0.087 s, so the QC is **the single most expensive step per leg** of those three — it duplicates a femoral-head fit and a trochanter search, exactly as on MRI. Quote it as a ratio and not as CPU-hours, for the reason above. The run also reproduced this bullet's own figures exactly, which is what makes it trustworthy:

## The condylar gate: zero-movement, checked five ways

**Zero movement, checked five ways**: the suite went 1850 -> **1883 passed, 1 xfailed, 0 failed** (exactly its 33 new tests), `git diff tests/golden/` is empty, an AST comparison against HEAD (definitions compared by name with docstrings stripped, so comment and prose edits are invisible to it) shows `knee.py` **purely additive — 3 definitions added, 0 changed, 0 removed**; `git diff` over `measurements/`, `femur.py`, `tibia.py`, `geometry.py`, `utils.py` and `image_io.py` is empty.

**Scope that claim honestly**, because the obvious shorthand overstates it: the two Augsburg scripts' `add_qc_flags` and `process_patient` *are* changed, necessarily, since that is where the gate is called; and `constants.py` and `docker/compute_torsion.py` show no definition change only because their edits are **module-level**, which a def-level comparison does not cover. So the defensible statement is "no existing *library* definition changed", not "nothing changed anywhere", and the gate is **non-mutating at runtime** (see the standard recorded above). The one pre-existing test that changed is `test_compute_torsion_pipeline.py`, additively and for the same reason the Lee gate changed it: that test's purpose is to mirror the container's exact results-dict key set, so leaving it alone would make the oracle stop mirroring production. Pinned by `tests/unit/test_condylar_line_qc.py` and `tests/characterization/test_condylar_line_qc.py`; the tests were mutation-tested rather than assumed (gate-never-fires 13 failures, gate-always-fires 10, reflection removed 8).
