"""Central definitions for segmentation labels and tunable algorithm thresholds.

Historically these values were scattered as bare literals across the region
modules (e.g. ``segmentation_label == 3`` for the acetabulum, ``1.2 * r`` for the
femoral-neck sphere shell). Collecting them here removes that duplication and
documents the assumptions in one place.

Conventions:
- Segmentation labels are *modality-scoped* because the same anatomy carries
  different labels in MRI hip scans vs. whole-leg CT (e.g. the acetabulum is
  label 3 in MRI, whereas in CT the hip bone, label 7, plays that role). Label
  values are exposed as defaults for function parameters so callers can still
  override them per dataset.
- Thresholds that no caller currently overrides live here as module constants;
  thresholds a caller varies stay as function parameters defaulting from here.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class MRIHipLabels:
    """Segmentation labels for MRI proximal-femur / hip masks."""
    femur: int = 1
    cartilage: int = 2
    acetabulum: int = 3


@dataclass(frozen=True)
class WholeLegCTLabels:
    """Segmentation labels for whole-leg CT masks."""
    femur: int = 1
    tibia: int = 2
    fibula: int = 3
    patella: int = 5
    hip: int = 7


#: Default label set for MRI hip segmentations.
MRI_HIP = MRIHipLabels()
#: Default label set for whole-leg CT segmentations.
CT_WHOLE_LEG = WholeLegCTLabels()

# --- Axial-plane angle composition ------------------------------------------
#: Relative tolerance on the equality of the two in-plane (x, y) voxel spacings, checked by
#: :func:`morphometry.geometry.validate_in_plane_isotropy` before any angle that is composed
#: from voxel *indices* in the axial plane (femoral torsion, tibial torsion, knee rotation and
#: the femoral anteversion's neck angle).
#: Such a composition maps a physical direction ``(cos t, sin t)`` to the index direction
#: ``(cos t / sx, sin t / sy)``, i.e. to the index angle ``atan2(k sin t, cos t)`` with
#: ``k = sx / sy``; the angle between two lines therefore comes out *sheared* by an amount
#: that depends on how the limb happens to lie in the slice, not only on ``k``. Worst case
#: over all limb orientations, for a true angle of 15 / 30 / 45 degrees:
#: ``k`` 1.001 -> 0.01 / 0.03 / 0.04, 1.01 -> 0.15 / 0.29 / 0.40, 1.05 -> 0.7 / 1.4 / 2.0,
#: 1.1 -> 1.5 / 2.8 / 4.0, 1.2 -> 3.0 / 5.6 / 7.9, 1.5 -> 7.3 / 13.8 / 18.7,
#: 2.0 -> 14.5 / 26.4 / 34.3, 2.4 -> 20.1 / 35.5 / 44.7 degrees (at ``k`` >= 3 a 45 degree
#: angle can even be folded to the other side of the (-90, 90] seam). The error grows with the
#: true angle, so those columns are not the worst case: at ``k`` = 1.001 the supremum over all
#: true angles is 0.0573 degrees, approached as the true angle tends to 90 (where the folded
#: angle can also cross the (-90, 90] seam). 1e-3 is therefore the
#: point at which the shear is numerically irrelevant (at most 0.06 degrees) while still
#: admitting the float noise real headers carry: every cohort in reach stores its two in-plane
#: zooms bit-identically (Augsburg 1.09375, NaKo 1.0), so nothing legitimate sits near the
#: threshold -- the only measured violation is ``dfg_knees``, at ``k = 2.4``.
#:
#: The same tolerance serves :func:`morphometry.geometry.validate_isotropy`, the stricter 3-D
#: sibling that the alpha angle's oblique reslice needs (it rotates in the ``(x, z)`` index plane
#: and reads its contour in the ``(rotated x, y)`` plane, so all three edges must be equal, not
#: just the in-plane pair). The reasoning for the value is the same -- numerically irrelevant
#: shear on one side, float noise in real headers on the other -- and every cohort is either
#: exactly isotropic (NaKo 1 mm) or far outside it (Augsburg hip 1.09375 x 1.09375 x 10 mm,
#: NMDID CT 3 mm slices), so nothing legitimate sits near the threshold here either.
IN_PLANE_SPACING_RTOL = 1e-3

# --- Femoral head / neck -----------------------------------------------------
#: Outer radius factor of the hollow sphere shell used to sample femoral-neck points.
FEMORAL_NECK_SPHERE_OUTER_FACTOR = 1.2
#: Outlier cutoff (in standard deviations) when filtering femoral-neck candidate points.
FEMORAL_NECK_OUTLIER_STD = 2.0
#: |z-score| above which a femoral-head slice is treated as too small and skipped.
FEMORAL_HEAD_LAYER_ZSCORE = 2.0
#: Growth of the column's slice-equivalent radius, *after* it has started shrinking,
#: that is read as the neck / greater trochanter merging into the femoral head's
#: connected component — i.e. the signal that the slab has left the head. Expressed in
#: the in-plane length unit the caller works in (millimetres for CT, in-plane voxels for
#: the MRI path, which are sub-millimetre on the hip protocols). Chosen small enough to
#: catch the merge on the slice it happens and large enough to ride out the
#: one-voxel-per-slice rasterisation jitter of a smoothly narrowing head.
FEMORAL_HEAD_RADIUS_JUMP = 0.5
#: Slices added to the measured head diameter when capping the femoral-head slab. One is
#: geometry: a head of radius r spans 2r/sz slice *intervals*, and the column walk stops
#: on the first slice past the head, one further down. The second is tolerance, for a head
#: marginally taller than wide or a starting slice a touch above the superior pole —
#: without it the cap trimmed a single slice off three of 92 otherwise untouched NMDID
#: hips (radius moved 0.26-0.42 mm). Two slices leaves the cap inert on every hip measured
#: while still cutting the runaway case from 63 slices to 18.
FEMORAL_HEAD_SLAB_MARGIN_SLICES = 2
#: Inlier band, in mm, of the seeded consensus sphere fit of the MRI femoral head
#: (:func:`morphometry.hip.get_femoral_head_center`, non-isotropic path): a slab contour point
#: joins the fit when its distance from the current sphere's surface is at most this. About two
#: in-plane voxels on the hip protocols -- wide enough to take in the rasterised head surface of
#: every slice through the head, narrow enough to leave out femoral-neck and trochanter surface,
#: which lies well inside or outside the head sphere. Not knife-edge: against 2.5 mm, tolerances of
#: 2.0 / 3.0 mm move the centre by a median 0.25 / 0.21 mm (90th percentile 0.44 / 0.42 mm, max
#: 3.1 / 3.7 mm) over the 300 fits of the 100 Augsburg reader-study hips (left, reflected right and
#: direct right halves). Checked against the head's own circle centres on its top slices, the
#: in-plane error is lowest at 2.0-2.5 mm (90th percentile 0.62-0.71 mm) and grows from 3 mm, where
#: a wider band starts to admit neck surface.
FEMORAL_HEAD_CONSENSUS_TOLERANCE_MM = 2.5
#: How far below the top slice of the femoral-head slab, in mm, the consensus fit may take its seed
#: points from. About one head radius (adult femoral heads are ~18-26 mm): the top slab slice sits at
#: the superior pole, so this keeps the seed on the head down to roughly its equator, where the
#: medial surface is still pure head. Deeper slab slices can hold the neck basin or the calcar, which
#: the slab reaches whenever the column walk stays in bone (it is bounded by the pipelines'
#: ``x_ratio``, which admits more slices than the true slice aspect); a seed started there stays there
#: at every tolerance. At 10 mm slices this leaves three or four seed slices.
FEMORAL_HEAD_SEED_DEPTH_MM = 30.0
#: Refit rounds after which the consensus head fit stops even if its inlier set still changes. The
#: inlier set creeps to a fixed point; on the Augsburg hips that takes a median 15 and at most 48 rounds
#: at the default tolerance. Hitting the cap logs a WARNING on the ``morphometry.hip`` logger.
FEMORAL_HEAD_CONSENSUS_MAX_ITERATIONS = 100

#: Minimum distance, in mm, of Murphy's neck-base level below the femoral head centre. Adult femoral
#: heads have radii of ~18-26 mm, so a level less than about half a radius below the centre lies inside
#: the head: the "lesser trochanter" was mis-detected and the head-to-base axis would be a few mm long.
#: On the Augsburg, NMDID and NaKo-sample hips the levels fall either below 5 mm (all mis-detections)
#: or above 11 mm; 5 and 10 mm reject the same legs.
MURPHY_NECK_BASE_MIN_HEAD_DISTANCE_MM = 10.0

#: Angular tolerance, in degrees, that sets how short Murphy's head-to-base axis may become before
#: the neck is flagged as ill-conditioned (``murphy_axis_short``, see
#: :func:`morphometry.femur.murphy_neck_base_qc`). The torsion is the *direction* of that axis in
#: the axial plane, so displacing either landmark by one in-plane voxel tilts it by
#: ``atan(voxel / axis length)``; requiring that tilt to stay below this tolerance makes the
#: minimum axis length ``1 / tan(tolerance)`` in-plane voxels (14.30 voxels at 4 degrees, i.e.
#: 15.6 mm on the 1.09 mm Augsburg hip MRI, 14.3 mm on the 1 mm NaKo hips, 11.4 mm on 0.8 mm CT).
#:
#: **The threshold was placed using these cohorts' own axis-length distribution, so their flag rate
#: is not an unbiased estimate of it.** What the data show: measured in in-plane voxels (the unit
#: the landmark error is in), the Augsburg large cohort's shortest axes are 6.37, 11.67, 13.67,
#: then 16.66, 16.84, 16.94 -- a 3.0-voxel gap -- so *every* tolerance from 3.44 to 4.18 degrees
#: flags exactly those three halves and nothing more. 4 degrees sits in the middle of that window
#: rather than at an arbitrary point. Pooled over all three cohorts the gap is not empty: the two
#: NaKo halves at 14.50 and 15.95 voxels fall inside it, and both are the same hip, which
#: ``murphy_base_in_head`` flags independently. Rates and specificity (Augsburg halves, against the
#: 3.7% base rate of ``|z| > 2`` on the Murphy angle itself, mean 25.1, SD 15.6): 5 degrees flags
#: 1 of 1887 halves at 26.8x enrichment, **4 degrees 3 at 17.9x**, 3 degrees 26 at 8.5x, 2.5
#: degrees 161 at 3.2x. At 4 degrees two of the three are the two halves of PA001127 (Murphy -67.1
#: and -31.8 degrees, ``|z|`` 5.9 and 3.7, the cohort's most negative values) and one is a
#: plausible false alarm (PA000445 right, 20.2 degrees, ``|z|`` 0.3).
#:
#: **Zero reader-study halves are flagged at any tolerance >= 3 degrees** (their axis minimum is
#: 20.28 voxels), and PA001127 is in the large cohort, not the 50-case reader set -- so **this gate
#: cannot be adjudicated against the human readers at all**. 4 degrees is also about a quarter of
#: the cohort spread of the angle itself and several times the systematic landmark effects this
#: package has had to care about (1-2 degrees from the neck-base slice choice, ~1.1 degrees from
#: the femoral-head centre).
MURPHY_AXIS_ANGLE_TOLERANCE_DEG = 4.0

#: Smallest head-centre-to-notch distance, in units of the fitted femoral head radius, at which the
#: Lee proximal neck axis is still considered to rest on neck cortex (``lee_ring_within_head_radius``, see
#: :func:`morphometry.femur.lee_neck_axis_qc`). ``morphometry.femur.contour_femoral_neck`` measures
#: ``a = |notch - head centre| / r`` and keeps the neck contour in the ring
#: ``(a - 0.1) * r .. (a + 0.1) * r`` about the head centre, so ``a`` is the ring's centre radius in
#: head radii: at ``a < 1`` the ring is drawn closer to the head axis than the head's own radius
#: rather than peripheral to it, and at ``a < 0.9`` its outer edge is too. It is then far too small
#: to follow the peripheral neck contour the fit is meant to use -- on the four flagged halves the
#: notch sits 6.2 to 16.1 mm from the head centre, against 32.2 and 33.1 mm on the two sound halves
#: of PA000001 -- which is anatomically implausible rather than merely unusual.
#:
#: **This is NOT the claim that the ring lies inside the femoral head, and that stronger claim is
#: false on three of the four flagged halves.** ``a`` is an in-plane distance in the selected axial
#: layer; the head's cross-section there has radius ``sqrt(r^2 - dz^2)``, and the Lee layer is the
#: most proximal qualifying one, so it sits well off the equator. Measured with the *unrounded* head
#: centre, that cross-section is 0.516 / 0.496 / 0.000 / 0.332 of ``r`` on PA000346 left, PA000622
#: right, PA000810 right and PA000932 left, so the notch is **outside** it on three, and on PA000810
#: the layer lies below the fitted sphere entirely. A containment test would flag 1 of the 4, so
#: ``a < 1`` is deliberately the more inclusive reading. Note
#: ``morphometry.femur.contour_femoral_neck``'s own docstring still carries the stronger "the
#: contour it keeps is head cortex rather than neck" wording from task #9, written before this was
#: measured. None of this moves the gate or the value 1.0; it is the *mechanism* that is weaker than
#: it first looked. It is also why the flag is named for the head RADIUS and not for the head: the
#: verdict it is modelled on, ``murphy_base_in_head``, tests the full 3-D distance in mm and really
#: does mean inside the sphere.
#:
#: Like ``murphy_base_in_head``, this is not a free parameter fitted
#: to a cohort: the candidates are the three points at which the ring crosses the head radius, and
#: the radius comes from the same fit that produced ``a``. But the geometry offers three of them
#: (0.9, 1.0, 1.1) and it was the cohort distribution that chose among them, so the claim is
#: "geometric family, data-broken tie" and **not** "the data had no say" -- see the two paragraphs
#: below, which give the distribution first and then the tie-break.
#:
#: **The threshold was placed using one cohort's own distribution of ``a``, so its flag rate there
#: is not an unbiased estimate of it.** What the data show, over the 1896 of the 1908 Augsburg
#: large-cohort image halves (954 cases x 2) that resolve a Lee axis -- the 12 that do not are 6
#: halves of 3 missing files, 3 ``ValueError`` and 3 ``RuntimeError`` -- with min 0.3089, median
#: 1.6222, max 2.2406: the 15 lowest values are 0.3089, 0.3265, 0.7684, 0.8434, **1.0192**,
#: 1.0477, 1.1048, 1.1061, 1.1067, 1.1080, 1.1232, 1.1788, 1.1793, 1.1831 and 1.1887, i.e. there is
#: a gap from 0.8434 to 1.0192 and *every* threshold in [0.844, 1.019] flags exactly the same four
#: halves (PA000346 image-left 0.7684, PA000622 image-right 0.8434, PA000810 image-right 0.3265,
#: PA000932 image-left 0.3089 -- 4 of 1896, 0.211%) and nothing more. Rates at other values, for
#: scale: 1.05 flags 6 halves, 1.1 also 6, 1.15 flags 11, 1.20 flags 16, 1.30 flags 51. "There is a
#: gap" has to be earned per quantity and is not a general property of this family of gates. The
#: contrast, from a separate survey of the condylar line not reproduced here: a 13.32-voxel gap
#: isolating 1 of 1904 halves on the femur, and on the tibia no gap at all (a continuum, largest
#: low-end gap 1.10), so a Lee-style gate is placeable there on one bone and not on the other.
#:
#: **It replicates on two further cohorts, which is the strongest thing that can be said for it.**
#: The 50-patient reader study: **100 of 100** halves resolve, ``a`` min 0.7684 (PA000346
#: image-left, the same half), next 1.1995, median 1.5529, max 1.9197, **1 of 100** below 1.0, every
#: threshold in (0.769, 1.1995] equivalent. The 60-case NaKo sample, 1 mm **isotropic** and through
#: the ``isotropic=True`` path: **120 of 120** resolve with no failures, min **1.1388**, median
#: 1.5231, max 2.0740, and **0 of 120** below 1.0 or even below 1.1 (3 below 1.2, 11 below 1.3).
#: So on a clean second-and-third cohort the gate fires on nothing and NaKo's minimum sits 14% clear
#: of the threshold: it is specific, not trigger-happy, which is the property a gate that marks rows
#: most needs.
#:
#: **The two distributions are comparable (median 1.52 NaKo vs 1.62 Augsburg) despite completely
#: different voxel geometry** -- 1 mm isotropic against 1.09375 x 1.09375 x 10 mm -- and that is not
#: a given, it is a consequence of ``a`` being a dimensionless *ratio*. The counterexample is in
#: this repo: the condylar-line analogue would be a length in **voxels**, and there the Augsburg
#: femoral line has a clean 13.32-voxel gap isolating 1 of 1904 halves while ``dfg_knees``
#: (0.75 x 0.3125 mm in plane) puts 11 of 95 femoral lines under 10 voxels with no gap at all. Same
#: style of gate; a voxel count does not transfer between cohorts of different spacing and this
#: ratio does. (That survey is the lead's and is not reproduced here; the Augsburg, reader-study and
#: NaKo ``a`` surveys above were each run twice, independently.)
#:
#: **Even so, do not read "there is a gap" as the main argument, because the pooled gap is thin**:
#: the binding lip is the large cohort's 1.0192, which sits only **1.9% above** the threshold, so a
#: single re-measurement of one half could close it. The order of the three supports is: the
#: geometry (the ring crossing the head radius) first, the replication second, the gap last. The
#: prevalence figures are what say the gate is rare enough to be worth having at all -- 4 of 1896,
#: 1 of 100, 0 of 120.
#:
#: **Three distinct points of this geometry could be the threshold -- 0.9, 1.0 and 1.1 -- so do not
#: "correct" the value to one of the others without reading this.** The ring spans
#: ``(a - 0.1) * r .. (a + 0.1) * r``, so its OUTER edge crosses the head radius at ``a = 0.9``
#: (below which the *whole* ring is within that radius), its CENTRE at ``a = 1.0``, and its INNER
#: edge at ``a = 1.1`` (below which *some* of it is). It is the centre, 1.0, that
#: is shipped -- note in particular that 1.0 is **not** the inner-edge crossing, which is 1.1. The
#: choice among the three is settled by the data and not by the geometry, because all three readings
#: are defensible: 1.1 lands 0.0048 from the next observed value (1.1048) and inside a cluster of
#: five halves between 1.1048 and 1.1232, so it separates nothing and its flag set is one
#: re-measurement away from changing; 1.0 and 0.9 both sit in the empty interval and flag the same
#: four. Of those two, 1.0 is taken as the earlier and more inclusive condition.
#:
#: **The tie is broken by the Augsburg large cohort alone, and the other two cohorts cannot break
#: it** -- say which data, because a reader will otherwise assume all three were consulted. On NaKo
#: all of 0.9, 1.0 and 1.1 flag **0 of 120** (only 1.15 separates anything, at 1 half), and on the
#: reader study all three flag the same **1 of 100**. So both are *consistent* with every candidate
#: and neither discriminates; the whole tie-break rests on the Augsburg cluster at 1.1048-1.1232.
#: Their contribution is the replication above -- that ``a < 1`` is rare anatomy-wide and not an
#: Augsburg artefact -- not the choice of value.
#:
#: One way the gate is **robust** rather than merely rare, worth knowing before anyone revisits the
#: head fit: PA000346's fitted head radius is 13.4 mm, below the 18-26 mm adult range that
#: :data:`MURPHY_NECK_BASE_MIN_HEAD_DISTANCE_MM`'s own comment cites. Since ``a = d / r``, an
#: under-sized ``r`` makes ``a`` *larger*, so correcting that fit would push this half further below
#: the threshold rather than out of the flag set.
#:
#: What motivates the gate is PA000346 image-left, whose Lee torsion moved 35.755 -> 3.168 degrees
#: across the task #9 read-out change -- the single largest mover of the 100 reader-study halves --
#: where its own neighbouring eligible layer reads ``a = 1.753`` and its contour fit kept 7 points
#: against a cohort median of 53. That is 1 of 100 in the reader study, which is why
#: ``contour_femoral_neck`` deliberately did not gate on it at the time; the cohort survey above is
#: the prevalence evidence it was waiting for. Three of the four flagged halves (PA000622, PA000810,
#: PA000932) are large-cohort cases outside the 50-patient reader study, so the gate is only even
#: *comparable* with the human readers on PA000346, and no such comparison was made: backlog task #2
#: closed the premise that the Augsburg readers can adjudicate this method at all (their
#: Lee-minus-Murphy gap is ~11 degrees too narrow for landmarks this far apart).
LEE_MIN_NOTCH_DISTANCE_FACTOR = 1.0

#: Largest landmark-error-to-tilt tolerance, in degrees, at which the posterior condylar reference
#: line (``morphometry.knee.get_knee_reference_line``) is still considered well conditioned
#: (``condylar_line_short``, see :func:`morphometry.knee.knee_condylar_line_qc`). As for
#: :data:`MURPHY_AXIS_ANGLE_TOLERANCE_DEG`, the tolerance is turned into a minimum length by
#: ``1 / tan(tolerance)`` **in in-plane voxels**: at 4 degrees a line shorter than 14.30 in-plane
#: voxels is one on which a single voxel of landmark error would tilt the line by more than the
#: tolerance. Only the line's in-plane *direction* enters the femoral torsion, the tibial torsion
#: and the knee rotation angle, so that tilt is the whole error -- one voxel turns a 9.5-voxel line
#: by 6.0 degrees and a 44-voxel line by 1.3.
#:
#: **This is deliberately a separate constant holding the same number as
#: :data:`MURPHY_AXIS_ANGLE_TOLERANCE_DEG`, not an alias of it.** What is inherited is the
#: *construction* -- "one voxel of landmark error must not outweigh the tolerance" -- which is why
#: no new value was fitted here and why the threshold is non-arbitrary by construction rather than
#: by a cohort. What is not inherited is the evidence: the Murphy constant's 4 degrees was placed
#: inside a measured gap in the Murphy axis-length distribution and its specificity was scored
#: against that angle's own outliers, and none of that transfers to a different landmark on a
#: different bone. Writing ``= MURPHY_AXIS_ANGLE_TOLERANCE_DEG`` would mean re-tuning the Murphy
#: gate against hip data silently moved this gate too, which is exactly the coupling to avoid; the
#: two are independent and may diverge.
#:
#: **What the resulting threshold does and does not rest on.** Surveyed over the Augsburg large
#: cohort -- 954 case directories, 952 of which hold a ``knee_seg.nii.gz``, i.e. **1904 image
#: halves** (``PA000255`` and ``PA000964`` have no knee segmentation) -- through the real
#: ``get_knee_reference_line``, loaded as the scripts load it (LPI, ``remove_outliers``, split at
#: ``shape[0] // 2``), and **measuring the line the measurements actually use**: a right half is
#: reflected first, as all three consumers reflect it
#: (``measurements/femur.py``, ``measurements/knee.py``, ``measurements/tibia.py``). Every figure
#: below is that reflected reading, which is the quantity this threshold is compared against.
#:
#: * **Femur**: 1904 of 1904 halves resolve. Min **5.39** voxels, then 22.80, 23.19, 29.27 -- the
#:   largest gap anywhere in the low end is that first one, **17.42 voxels wide** -- p1 33.06,
#:   median 45.10, max 67.00.
#: * **Tibia**: 1902 of 1904 resolve; the two that do not are both halves of ``PA000120``, whose
#:   *tibia* label is empty (its femoral line resolves normally, so this is not a whole-knee
#:   failure). Low tail **11.05**, 13.00, 13.04, 15.00, 15.03, 15.26, 15.62, 16.00 -- a
#:   **continuum** whose largest low-end gap is 1.96 -- p1 17.03, median 29.65, max 53.67.
#:
#: So 14.30 lands inside a wide empty interval on the femur as a *consequence* of the construction
#: -- it was not placed there -- and on the tibia it lands in the middle of a continuum, where no
#: threshold is distinguished and 14.30 is simply the length at which the tolerance is reached. The
#: femoral median is 45.10 voxels against the tibial 29.65, which is why 12-16 voxels is ordinary on
#: a tibia and degenerate on a femur. At 14.30 the gate flags **1 of 1904 femoral lines (0.05%)**
#: and **3 of 1902 tibial ones (0.16%)**, all four on image-right halves:
#:
#: * ``PA000877`` right femur, **5.39** voxels, reference layer 0, no-notch fallback.
#: * ``PA000155`` right tibia, **11.05** voxels, layer 17, fallback.
#: * ``PA000571`` right tibia, **13.00** voxels, layer 17, fallback.
#: * ``PA000435`` right tibia, **13.04** voxels, layer 22, **no fallback** -- so the gate is not a
#:   proxy for the no-notch fallback, which fires on 0.32% of femoral and 15.77% of tibial
#:   sections and is measured to be nearly harmless in its own right.
#:
#: **The reading matters, and a survey of the unreflected right half gets a different answer.**
#: Searching a right half directly reproduces the shape of the distribution (femoral min 9.49 then
#: 22.80, median 44.34; tibial min 12.04 then 13.04, 14.14, median 30.81) and arrives at the same
#: two *counts*, but **two of the three tibial legs it names are different ones** (it flags
#: ``PA000591`` and ``PA000671`` right, which reflect to 18.11 and 18.25 voxels and are not
#: flagged, and misses ``PA000435`` and ``PA000571``). Over the 1903 measurable right halves the two
#: readings differ by a median 2.27 voxels, up to 18.03, and agree exactly on only 276, so the
#: agreement in counts is a coincidence of this cohort. Left halves are identical under both
#: readings. Task #5's mirror figure bounds the line's **bearing** (median 0.33, max 2.06 degrees)
#: and must not be read as bounding its length.
#:
#: What motivates the gate is ``PA000877`` image-right, whose femoral search lands on reference
#: layer 0 (the edge of the field of view), takes the no-notch centroid fallback and returns a line
#: of **5.39 in-plane voxels** -- the cohort minimum, and the only half the gate flags. Its
#: image-**left** femur is on layer 0 too (those two are the only layer-0 selections in the cohort,
#: so the leg is identifiable two independent ways) and takes the fallback as well, yet measures
#: 41.44 voxels and is **not** flagged: neither the layer nor the fallback separates the two halves,
#: and the length does. **Three different lengths exist for this one leg and must not be
#: conflated**: 5.39 is the reflected length the gate compares and prints; **9.49** is the
#: unreflected length of the same half; and the 7.3 direct / 6.3 reflected that CLAUDE.md's task #17
#: paragraph quotes predate task #5's condylar read-out change and belong to neither. That same task
#: #17 paragraph records this leg's knee rotation angle moving 69.16 -> 14.97 degrees against a
#: cohort mean of about 4 degrees with neither answer trustworthy, which is the evidence that it
#: needs a gate -- but **that move was measured on the old read-out**, so it is not the same
#: measurement as the 5.39 above. It was the one leg the task #17 change could not account for, and
#: it had no gate; the second-largest mover there was 7.01 degrees.
#:
#: **The threshold is a count of in-plane voxels, which is only a length where the two in-plane
#: voxel edges agree**, so the gate refuses in-plane anisotropic data outright rather than
#: comparing against it (:func:`morphometry.geometry.validate_in_plane_isotropy`, backlog task
#: #10 -- the same precondition the three angles themselves carry). No rate from the ``dfg_knees``
#: knee MRI is quoted above or anywhere else for this gate: that cohort is 0.75 x 0.3125 mm in
#: plane, where a voxel count is not a length at all, so the comparison is void on units rather
#: than merely inconvenient. For the same reason the flag reports **voxels** and not the
#: millimetres ``murphy_axis_short`` reports -- Augsburg alone carries four in-plane spacings, so a
#: millimetre value would not be monotone in the verdict (see
#: :func:`morphometry.knee.knee_condylar_line_qc`).
CONDYLAR_LINE_ANGLE_TOLERANCE_DEG = 4.0

# --- Subchondral / cartilage -------------------------------------------------
#: Outer radius factor of the sphere shell used to find femoral-head surface points
#: facing the acetabulum (minimum-distance measurement).
SUBCHONDRAL_SPHERE_OUTER_FACTOR = 1.05
#: Inner / outer radius factors for extracting cartilage surface points.
CARTILAGE_SPHERE_INNER_FACTOR = 0.8
CARTILAGE_SPHERE_OUTER_FACTOR = 1.2
#: Distance percentile kept when filtering cartilage surface points.
CARTILAGE_SURFACE_QUANTILE = 0.75

# --- Ray-traced subchondral distance ----------------------------------------
DEFAULT_N_RAYS = 200
DEFAULT_CONE_ANGLE_DEG = 45.0
DEFAULT_RAY_LENGTH_FACTOR = 3.0

# --- Trochanter plausibility (femoral-head-centre distance, mm) --------------
#: Plausible distance range from femoral head centre to greater trochanter.
TROCHANTER_MAJOR_FH_DIST_MM = (425.0, 515.0)
#: Plausible distance range from femoral head centre to lesser trochanter.
TROCHANTER_MINOR_FH_DIST_MM = (40.0, 75.0)

# --- Center-edge angle -------------------------------------------------------
#: Superior margin factor (x femoral-head radius) defining the search band for the
#: lateral acetabular edge. One value for both image sides: the two halves hold mirror-image
#: anatomy, so a per-side value measures the same hip differently depending on which side it
#: is on (the former 1.5 left / 1.1 right accounted for ~1.9 degrees of the NaKo cohort's
#: patient-left > patient-right CEA). The value 1.1 was CHOSEN AGAINST the 60 NaKo reader
#: cases, i.e. calibrated on the evaluation set: the CEA under-reads both readers there, and
#: agreement improves monotonically as the factor drops (1.5 -> 1.1: ICC2 vs MH right/left
#: 0.37/0.39 -> 0.48/0.53, vs TH 0.50/0.47 -> 0.62/0.62; signed bias MH -6.0/-6.5 -> -4.8/-4.6,
#: TH -2.5/-4.6 -> -1.3/-2.7 degrees). Applies to every modality; on the 46 NMDID CT
#: verification cases the CEA is identical at 1.1 and 1.5.
CEA_UPPER_MARGIN_FACTOR = 1.1
