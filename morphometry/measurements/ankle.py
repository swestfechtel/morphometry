"""Ankle measurement functions: plafond-malleolus angle (PMA).

Note: ``get_plafond_reference_line`` in ``morphometry.ankle`` is currently a stub
(it returns a fixed sagittal axis pending a proper 3D plane fit), so the PMA value
is provisional. The reference helpers stay in ``morphometry.ankle``.
"""
import numpy as np

from morphometry.ankle import get_plafond_reference_line, get_malleoli_reference_line
from morphometry.utils import calculate_angle_between_vectors


def calculate_pma_angle(segmentation_mask: np.ndarray, tibia_label: int = 1, fibula_label: int = 2) -> float:
    """
    Calculate the plafond-malleolus angle (PMA).
    :param segmentation_mask: A 3D segmentation mask of the ankle.
    :param tibia_label: The segmentation label of the tibia.
    :param fibula_label: The segmentation label of the fibula.
    :return: The plafond-malleolus angle in degrees.
    """
    plafond_start, plafond_end = get_plafond_reference_line(segmentation_mask)
    malleoli_start, malleoli_end = get_malleoli_reference_line(segmentation_mask, tibia_label, fibula_label)

    plafond_line = plafond_end - plafond_start
    malleoli_line = malleoli_end - malleoli_start

    return calculate_angle_between_vectors(plafond_line, malleoli_line)
