import numpy as np
from morphometry.knee import get_knee_reference_line
from morphometry.bresenham import bresenhamline
from morphometry.utils import calculate_angle_between_vectors
from scipy.ndimage import center_of_mass
from skimage.measure import regionprops, label
from typing import Tuple
from matplotlib import pyplot as plt
from morphometry.utils import draw_line


def get_layer_with_largest_diameter(segmentation_mask: np.ndarray) -> int:
    """
    Return the layer with the biggest ellipse covering the mask area.

    For each layer, the diameter of an ellipse covering an area equivalent to the mask area is calculated. The layer
    with the biggest diameter is returned.
    :param segmentation_mask: A 3D segmentation mask.
    :return: The layer index with the biggest mask area.
    """

    diameter = np.zeros(segmentation_mask.shape[0])
    # save diameters of the layers
    for i in range(segmentation_mask.shape[2]):
        if len(np.nonzero(segmentation_mask[:, :, i])[0]) != 0:
            props = regionprops(label(segmentation_mask[:, :, i]))
            if props.__len__() > 1:
                i_biggest = 0
                for j in range(props.__len__()):
                    if props[j].equivalent_diameter > props[
                            i_biggest].equivalent_diameter:
                        i_biggest = j
                diameter[i] = props[i_biggest].equivalent_diameter
            else:
                diameter[i] = props[0].equivalent_diameter

    # find index of the layer with the biggest diameter
    indices = np.argsort(diameter)
    return indices[-1]

def get_distal_reference_line(segmentation_mask: np.ndarray, tibia_label: int, fibula_label: int) -> Tuple[int, np.ndarray, np.ndarray]:
    """
    Get the distal reference line of a segmentation mask for calculating the tibial torsion.

    Finds the layer with the biggest diameter of the tibia and calculates the center of mass of the tibia and fibula
    on that layer. The reference line connects both center of mass points.
    :param segmentation_mask: A 3D segmentation mask of the tibia and fibula.
    :param tibia_label: The segmentation label of the tibia.
    :param fibula_label: The segmentation label of the fibula.
    :return: The layer and start and end points of the reference line.
    """
    tibia_mask = np.where(segmentation_mask == tibia_label, 1, 0)
    fibula_mask = np.where(segmentation_mask == fibula_label, 1, 0)
    # find index of the layer with the biggest diameter of the tibia
    layer_index = get_layer_with_largest_diameter(tibia_mask)

    # calculate center of mass of tibia und fibula on the layer
    com_tibia = center_of_mass(tibia_mask[:, :, layer_index])
    com_fibula = center_of_mass(fibula_mask[:, :, layer_index])

    # transform points from layer mask to 3D mask
    com_tibia = np.array([com_tibia[0], com_tibia[1], layer_index])
    com_fibula = np.array([com_fibula[0], com_fibula[1], layer_index])

    return layer_index, com_tibia, com_fibula
