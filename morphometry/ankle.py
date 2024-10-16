import numpy as np
from typing import Tuple
from scipy.ndimage import center_of_mass
from morphometry.utils import angle_between


def get_plafond_reference_line(tibia_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get the reference line for the tibial plafond.

    For now just a dummy implementation that returns a vector along the sagittal axis because a proper
    implementation requires isotropic images.
    :param tibia_mask: A 3D segmentation mask of the tibia, where the tibia is labeled 1 and everything else 0.
    :return: The start and end points of the reference line.
    """
    # TODO: Implement a proper reference line for the tibial plateau with 3-dimensional plane fitting.
    return np.array([0, 0, 0]), np.array([0, 0, 1])


def get_malleoli_reference_line(segmentation_mask: np.ndarray, tibia_label: int = 1, fibula_label: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get the reference line connecting the malleoli.

    The malleoli reference line is the line connecting the tips of the tibial and fibular malleoli.
    :param segmentation_mask: A 3D segmentation mask of the ankle.
    :param tibia_label: The segmentation label of the tibia.
    :param fibula_label: The segmentation label of the fibula.
    :return: The start and end points of the reference line.
    """
    tibia_mask = np.where(segmentation_mask == tibia_label, 1, 0)
    fibula_mask = np.where(segmentation_mask == fibula_label, 1, 0)

    most_inferior_tibia_layer = np.max(np.argwhere(tibia_mask)[:, 0])
    most_inferior_fibula_layer = np.max(np.argwhere(fibula_mask)[:, 0])

    com_tibia = center_of_mass(tibia_mask[most_inferior_tibia_layer])
    com_fibula = center_of_mass(fibula_mask[most_inferior_fibula_layer])

    com_tibia = np.array([most_inferior_tibia_layer, com_tibia[0], com_tibia[1]])
    com_fibula = np.array([most_inferior_fibula_layer, com_fibula[0], com_fibula[1]])

    return com_tibia, com_fibula


def calculate_pma_angle(segmentation_mask: np.ndarray, tibia_label: int = 1, fibula_label: int = 2) -> float:
    """
    Calculate the plafond malleolus angle.
    :param segmentation_mask: A 3D segmentation mask of the ankle.
    :param tibia_label: The segmentation label of the tibia.
    :param fibula_label: The segmentation label of the fibula.
    :return: The plafond malleolus angle in degrees.
    """
    plafond_start, plafond_end = get_plafond_reference_line(segmentation_mask)
    malleoli_start, malleoli_end = get_malleoli_reference_line(segmentation_mask, tibia_label, fibula_label)
    print(malleoli_start, malleoli_end)

    plafond_line = plafond_end - plafond_start
    malleoli_line = malleoli_end - malleoli_start

    return np.degrees(angle_between(plafond_line, malleoli_line))

