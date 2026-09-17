"""Tibia measurement functions: tibial torsion.

Combines the proximal posterior-condylar line (``get_knee_reference_line``) and the
distal tibia-fibula line (``get_distal_reference_line``) into the tibial torsion
angle. Landmarks are exposed separately via :func:`get_tibial_torsion_landmarks`.

"left"/"right" refers to the image side, not the patient side.
"""
from typing import Tuple

import numpy as np
from matplotlib import pyplot as plt

from morphometry.knee import get_knee_reference_line
from morphometry.tibia import get_distal_reference_line
from morphometry import geometry as G


def _tibial_torsion_reference_points(knee_mask: np.ndarray, ankle_mask: np.ndarray, tibia_label_knee: int,
                                     tibia_label_ankle: int, fibula_label: int):
    """Resolve the proximal (knee) and distal (ankle) tibial reference points, ordered."""
    knee_bin = np.where(knee_mask == tibia_label_knee, 1, 0)
    _, knee_start, knee_end = get_knee_reference_line(knee_bin, 'tibia')
    if knee_start[0] < knee_end[0]:  # knee_start gets the larger x (medial on a left image half, lateral on a right one)
        knee_start, knee_end = knee_end, knee_start
    _, ankle_start, ankle_end = get_distal_reference_line(ankle_mask, tibia_label_ankle, fibula_label)
    return knee_start, knee_end, ankle_start, ankle_end


def _tibial_torsion_angle(knee_start: np.ndarray, knee_end: np.ndarray,
                          ankle_start: np.ndarray, ankle_end: np.ndarray, side: str) -> float:
    """Compose the tibial torsion from the posterior condylar line and the tibia -> fibula line.

    Sign derivation (LPI index frame: x increases towards the patient's left, y posteriorly). A left
    image half holds a patient-right leg, so lateral is -x and posterior is +y. With ``p`` along the
    (undirected) condylar line and ``d`` directed tibia centroid -> fibula centroid (pointing
    laterally), external torsion turns ``d`` from lateral towards posterior relative to ``p``:
    ``atan2(-(p_x d_y - p_y d_x), p . d)``, folded into (-90, 90]. This is the same rotation sense as
    the femoral antetorsion, so both use :func:`morphometry.geometry.signed_axial_angle`; a right
    image half (mirror-image leg, lateral = +x) negates the x components. The result is invariant to
    an in-plane rotation of the whole leg and does not depend on the condylar endpoint order.

    This replaces a composition that folded both lines to acute angles against the image x-axis and
    combined them by the signs of their y-offsets. That equalled the signed angle only while the
    fibula centroid lay posterior to the tibia centroid in the image; otherwise it returned the
    negated torsion and, past 90 degrees, a wrong magnitude. The angle is computed from voxel indices
    (the knee and ankle masks are assumed to share the in-plane frame and have equal x/y spacing).
    :param knee_start: One posterior tibial condyle reference point.
    :param knee_end: The other posterior tibial condyle reference point.
    :param ankle_start: The tibia centroid at ankle level.
    :param ankle_end: The fibula centroid at ankle level.
    :param side: Image side, 'left' or 'right'.
    :return: The tibial torsion in degrees, in (-90, 90]; negative means internal torsion.
    """
    condyles = np.asarray(knee_start, dtype=float) - np.asarray(knee_end, dtype=float)
    tibia_to_fibula = np.asarray(ankle_end, dtype=float) - np.asarray(ankle_start, dtype=float)
    return G.signed_axial_angle(condyles, tibia_to_fibula, side)


def calculate_tibial_torsion(knee_mask: np.ndarray, ankle_mask: np.ndarray, tibia_label_knee: int,
                             tibia_label_ankle: int, fibula_label: int, side: str = 'left',
                             plot: bool | plt.Axes = False) -> float:
    """
    Calculate the tibial torsion angle.

    The signed axial-plane angle from the posterior tibial condylar line (knee level) to the
    line from the tibia centroid to the fibula centroid at the level of the tibia's largest
    cross-section, positive for external torsion (the distal line turned from lateral towards
    posterior relative to the condylar line) and negative for internal torsion, relative to
    the landmarks. See :func:`_tibial_torsion_angle`.
    :param knee_mask: A 3D segmentation mask of the knee.
    :param ankle_mask: A 3D segmentation mask of the ankle.
    :param tibia_label_knee: The label of the tibia at knee level.
    :param tibia_label_ankle: The label of the tibia at ankle level.
    :param fibula_label: The label of the fibula.
    :param side: Side of the image (not patient!), either 'left' or 'right'.
    :param plot: A sequence of two matplotlib Axes to draw the reference lines on, or False.
    :return: The tibial torsion angle in degrees.
    """
    G.validate_side(side)

    knee_start, knee_end, ankle_start, ankle_end = _tibial_torsion_reference_points(
        knee_mask, ankle_mask, tibia_label_knee, tibia_label_ankle, fibula_label)

    angle = _tibial_torsion_angle(knee_start, knee_end, ankle_start, ankle_end, side)

    if plot is not False:
        knee_bin = np.where(knee_mask == tibia_label_knee, 1, 0)
        plot[0].imshow(knee_bin[:, :, int(knee_start[2])].T)
        plot[0].plot([knee_start[0], knee_end[0]], [knee_start[1], knee_end[1]], color='red')
        plot[1].imshow(ankle_mask[:, :, int(ankle_start[2])].T)
        plot[1].plot([ankle_start[0], ankle_end[0]], [ankle_start[1], ankle_end[1]], color='red')
    return angle


def get_tibial_torsion_landmarks(knee_mask: np.ndarray, ankle_mask: np.ndarray, tibia_label_knee: int,
                                 tibia_label_ankle: int, fibula_label: int, side: str = 'left') -> dict:
    """
    Return the tibial-torsion reference landmarks as a dict.

    Provided separately from :func:`calculate_tibial_torsion` so the calculation keeps
    a single return type. ``side`` is accepted for call-site symmetry; the landmark
    points themselves do not depend on it.
    :return: ``{'knee_start', 'knee_end', 'ankle_start', 'ankle_end'}`` as numpy arrays.
    """
    G.validate_side(side)
    knee_start, knee_end, ankle_start, ankle_end = _tibial_torsion_reference_points(
        knee_mask, ankle_mask, tibia_label_knee, tibia_label_ankle, fibula_label)
    return {'knee_start': knee_start, 'knee_end': knee_end, 'ankle_start': ankle_start, 'ankle_end': ankle_end}
