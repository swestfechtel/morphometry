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
#: lateral acetabular edge, per image side. The left/right asymmetry is inherited
#: from the original implementation and is flagged for review.
CEA_UPPER_MARGIN_FACTOR_LEFT = 1.5
CEA_UPPER_MARGIN_FACTOR_RIGHT = 1.1
