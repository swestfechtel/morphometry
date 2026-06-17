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
from morphometry.utils import calculate_angle_between_vectors
from morphometry import geometry as G


def _tibial_torsion_reference_points(knee_mask: np.ndarray, ankle_mask: np.ndarray, tibia_label_knee: int,
                                     tibia_label_ankle: int, fibula_label: int):
    """Resolve the proximal (knee) and distal (ankle) tibial reference points, ordered."""
    knee_bin = np.where(knee_mask == tibia_label_knee, 1, 0)
    _, knee_start, knee_end = get_knee_reference_line(knee_bin, 'tibia')
    if knee_start[0] < knee_end[0]:  # knee_end is always lateral (left) of knee_start
        knee_start, knee_end = knee_end, knee_start
    _, ankle_start, ankle_end = get_distal_reference_line(ankle_mask, tibia_label_ankle, fibula_label)
    return knee_start, knee_end, ankle_start, ankle_end


def calculate_tibial_torsion(knee_mask: np.ndarray, ankle_mask: np.ndarray, tibia_label_knee: int,
                             tibia_label_ankle: int, fibula_label: int, side: str = 'left',
                             plot: bool | plt.Axes = False) -> float:
    """
    Calculate the tibial torsion angle.

    The angle between the posterior tibial condyles at knee level and the line
    connecting the tibia and fibula centroids at the level of the tibia's largest
    cross-section. The proximal and distal angles are each folded to [0, 90] and
    added or subtracted depending on their anterior-posterior orientation signs.
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

    proximal_angle = G.fold_to_acute(calculate_angle_between_vectors(knee_end - knee_start, np.array([-1, 0, 0])))
    proximal_orientation = (knee_end[1] - knee_start[1]) if side == 'left' else (knee_start[1] - knee_end[1])

    x = np.array([-1, 0, 0]) if side == 'left' else np.array([1, 0, 0])
    distal_angle = G.fold_to_acute(calculate_angle_between_vectors(ankle_end - ankle_start, x))
    distal_orientation = ankle_end[1] - ankle_start[1]

    if np.sign(proximal_orientation) != np.sign(distal_orientation):
        angle = distal_angle + proximal_angle
    else:
        angle = distal_angle - proximal_angle

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
