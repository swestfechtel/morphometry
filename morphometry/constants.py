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
