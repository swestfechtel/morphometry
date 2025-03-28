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


def calculate_tibial_torsion(knee_mask: np.ndarray, ankle_mask: np.ndarray, tibia_label_knee: int, tibia_label_ankle: int, fibula_label: int, side: str = 'left', plot: bool = False, mark_mask: bool = False) -> float | Tuple[float, plt.Figure] | Tuple[float, np.ndarray, np.ndarray] | Tuple[float, plt.Figure, np.ndarray, np.ndarray]:
    """
    Calculate the tibial torsion angle.

    The tibial torsion angle is calculated as the angle between the line connecting the posterior condyles of the tibia
    at knee level the line connecting the centers of mass of the tibia and fibula on the layer with the biggest diameter
    of the tibia.
    :param knee_mask: A 3D segmentation mask of the knee.
    :param ankle_mask: A 3D segmentation mask of the ankle.
    :param tibia_label_knee: The segmentation label of the tibia at knee level.
    :param tibia_label_ankle: The segmentation label of the tibia at ankle level.
    :param fibula_label: The segmentation label of the fibula.
    :param side: Side of the image (not patient!), either 'left' or 'right'.
    :param plot: If True, plot the distal reference line and the line connecting the center of mass points.
    :param mark_mask: Whether to mark landmarks and reference lines on the segmentation masks.
    :return: The tibial torsion angle in degrees.
    """
    assert side in ['left', 'right'], 'Side must be either "left" or "right"'

    knee_mask = np.where(knee_mask == tibia_label_knee, 1, 0)
    # get proximal reference line
    knee_layer, knee_start, knee_end = get_knee_reference_line(knee_mask, 'tibia')
    print(f'knee_start: {knee_start}, knee_end: {knee_end}')
    proximal_line = knee_end - knee_start

    # knee_end is always left of knee_start
    if knee_start[0] < knee_end[0]:  # if this is somehow not the case, swap the points
        tmp = knee_start
        knee_start = knee_end
        knee_end = tmp

    x = np.array([-1, 0, 0])  # because end is always left of start

    proximal_angle = calculate_angle_between_vectors(proximal_line, x)

    if proximal_angle > 90:
        proximal_angle = 180 - proximal_angle

    proximal_orientation = knee_end[1] - knee_start[1]
    print(f'proximal orientation: {proximal_orientation}')
    if side == 'left':
        if proximal_orientation < 0:  # lateral condyle is anterior to medial condyle
            proximal_angle = -proximal_angle
    else:
        if proximal_orientation > 0:  # lateral condyle is anterior to medial condyle
            proximal_angle = -proximal_angle


    # get distal reference line
    ankle_layer, ankle_start, ankle_end = get_distal_reference_line(ankle_mask, tibia_label_ankle, fibula_label)
    distal_line = ankle_end - ankle_start
    x = np.array([-1, 0, 0]) if side == 'left' else np.array([1, 0, 0])
    distal_angle = calculate_angle_between_vectors(distal_line, x)

    if distal_angle > 90:
        distal_angle = 180 - distal_angle

    distal_orientation = ankle_end[1] - ankle_start[1]
    if distal_orientation < 0:  # fibula is anterior to tibia
        distal_angle = -distal_angle

    angle = distal_angle - proximal_angle

    print(f'proximal line: {proximal_line}')
    print(f'distal line: {distal_line}')
    print(f'proximal angle: {proximal_angle}')
    print(f'distal angle: {distal_angle}')
    print(f'angle: {angle}')

    if not mark_mask and not plot:
        return angle

    if plot:
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(knee_mask[:, :, knee_layer].T)
        ax[0].plot([knee_start[0], knee_end[0]], [knee_start[1], knee_end[1]], color='red')
        ax[1].imshow(ankle_mask[:, :, ankle_layer].T)
        ax[1].plot([ankle_start[0], ankle_end[0]], [ankle_start[1], ankle_end[1]], color='red')
        if not mark_mask:
            return angle, fig

    if mark_mask:
        knee_mask = np.where(knee_mask == 1, tibia_label_knee, knee_mask)
        ankle_mask = draw_line(ankle_mask, ankle_layer, ankle_start, ankle_end)
        knee_mask = draw_line(knee_mask, knee_layer, knee_start, knee_end)

        if not plot:
            return angle, ankle_mask, knee_mask

    return angle, fig, ankle_mask, knee_mask
