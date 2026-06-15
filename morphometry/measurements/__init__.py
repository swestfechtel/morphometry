"""Public morphometric measurement functions.

All ``calculate_*`` measurements live in per-region submodules here
(``hip``, ``femur``, ``knee``, ``tibia``, ``ankle``, ``whole_leg``, ``cartilage``).
The landmark / reference-line helpers (``get_*``) they build on remain in the
region modules at the package root (``morphometry.hip`` etc.).

Convention reminder: the ``side`` argument everywhere refers to the *image* side
(``array[:shape[0]//2]`` is "left"), which is the opposite of the patient side.
"""
from morphometry.measurements.hip import (
    calculate_ccd,
    calculate_ccd_ct,
    calculate_anteversion,
    calculate_alpha_angle,
    calculate_acetabular_anteversion,
    calculate_acetabular_depth,
    calculate_center_edge_angle,
    calculate_min_distance_between_femoral_head_and_acetabulum,
    calculate_subchondral_distance_ray_tracing,
    calculate_subchondral_distance_ray_tracing_ct,
    calculate_cartilage_thickness_knn,
    calculate_cartilage_thickness_ray_tracing,
    calculate_femoral_offset,
    calculate_femoral_offset_projected,
)

__all__ = [
    "calculate_ccd",
    "calculate_ccd_ct",
    "calculate_anteversion",
    "calculate_alpha_angle",
    "calculate_acetabular_anteversion",
    "calculate_acetabular_depth",
    "calculate_center_edge_angle",
    "calculate_min_distance_between_femoral_head_and_acetabulum",
    "calculate_subchondral_distance_ray_tracing",
    "calculate_subchondral_distance_ray_tracing_ct",
    "calculate_cartilage_thickness_knn",
    "calculate_cartilage_thickness_ray_tracing",
    "calculate_femoral_offset",
    "calculate_femoral_offset_projected",
]
