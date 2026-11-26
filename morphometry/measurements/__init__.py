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
    get_femoral_torsion_landmarks_ct,
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
    calculate_anteversion_head_line,
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
import importlib

# Cartilage re-exports are resolved lazily (PEP 562) so that importing any measurement -- notably the
# torsion docker image's entry point -- does not load the cartilage code.
_CARTILAGE_MODULE = "cartilage"
_CARTILAGE_NAMES = (
    "calculate_tibial_cartilage_thickness",
    "calculate_femoral_cartilage_thickness",
    "calculate_knee_cartilage_thickness",
    "calculate_thinnest_cartilage_area",
    "plot_knee_segments",
    "plot_knee_thickness",
    "find_thinnest_area",
    "find_thinnest_area_by_compartment",
    "split_tibia_thickness_by_compartment",
    "plot_thinnest_point",
)


def __getattr__(name):
    """Resolve a cartilage re-export on first access (PEP 562).

    :param name: attribute requested from ``morphometry.measurements``.
    :return: the cartilage object of that name (cached in module globals afterwards).
    :raises AttributeError: if ``name`` is not one of the lazily exported cartilage names.
    """
    if name in _CARTILAGE_NAMES:
        # Deliberately NOT an import statement and with a non-literal first argument: cartilage is
        # intentionally outside the torsion image's versioned import closure (the release tooling's
        # static walker does not follow this). A change here must be reviewed with that in mind.
        module = importlib.import_module(__name__ + "." + _CARTILAGE_MODULE)
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    """List module attributes including the lazily exported cartilage names."""
    return sorted(set(globals()) | set(_CARTILAGE_NAMES))


__all__ = [
    "calculate_femoral_torsion",
    "calculate_femoral_torsion_ct",
    "get_femoral_torsion_landmarks",
    "get_femoral_torsion_landmarks_ct",
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
    "calculate_anteversion_head_line",
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
    "calculate_thinnest_cartilage_area",
    "plot_knee_segments",
    "plot_knee_thickness",
    "find_thinnest_area",
    "find_thinnest_area_by_compartment",
    "split_tibia_thickness_by_compartment",
    "plot_thinnest_point",
]
