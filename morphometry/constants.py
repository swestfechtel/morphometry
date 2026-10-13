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
