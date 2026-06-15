"""Public morphometric measurement functions.

All ``calculate_*`` measurements live in per-region submodules here
(``hip``, ``femur``, ``knee``, ``tibia``, ``ankle``, ``whole_leg``, ``cartilage``).
The landmark / reference-line helpers (``get_*``) they build on remain in the
region modules at the package root (``morphometry.hip`` etc.).

Convention reminder: the ``side`` argument everywhere refers to the *image* side
(``array[:shape[0]//2]`` is "left"), which is the opposite of the patient side.
"""
from morphometry.measurements.femur import (
    calculate_femoral_torsion,
    calculate_femoral_torsion_ct,
    get_femoral_torsion_landmarks,
)
from morphometry.measurements.knee import (
    calculate_knee_rotation_angle,
    calculate_joint_line_convergence_angle,
)
from morphometry.measurements.tibia import (
    calculate_tibial_torsion,
    get_tibial_torsion_landmarks,
)
from morphometry.measurements.ankle import calculate_pma_angle
from morphometry.measurements.whole_leg import (
    calculate_mechanical_axis_deviation,
    calculate_hip_knee_ankle_angle,
    calculate_bone_length,
    calculate_bone_length_ct,
)
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
from morphometry.measurements.cartilage import (
    calculate_tibial_cartilage_thickness,
    calculate_femoral_cartilage_thickness,
    calculate_knee_cartilage_thickness,
)

__all__ = [
    "calculate_femoral_torsion",
    "calculate_femoral_torsion_ct",
    "get_femoral_torsion_landmarks",
    "calculate_knee_rotation_angle",
    "calculate_joint_line_convergence_angle",
    "calculate_tibial_torsion",
    "get_tibial_torsion_landmarks",
    "calculate_pma_angle",
    "calculate_mechanical_axis_deviation",
    "calculate_hip_knee_ankle_angle",
    "calculate_bone_length",
    "calculate_bone_length_ct",
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
    "calculate_tibial_cartilage_thickness",
    "calculate_femoral_cartilage_thickness",
    "calculate_knee_cartilage_thickness",
]
