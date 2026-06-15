"""Knee measurement functions: knee rotation angle and joint-line convergence angle.

The posterior-condylar reference lines are computed by ``get_knee_reference_line``
in ``morphometry.knee``; this module combines them into angles. "left"/"right"
refers to the image side, not the patient side.
"""
import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import center_of_mass

from morphometry.image_io import Segmentation
from morphometry.knee import get_knee_reference_line
from morphometry.utils import calculate_angle_between_vectors, extract_connected_components_2d
from morphometry import geometry as G


def _order_condyle_points(start: np.ndarray, end: np.ndarray):
    """Order condyle endpoints so the end is medial to the start (start[0] >= end[0])."""
    if start[0] < end[0]:
        return end, start
    return start, end


def calculate_knee_rotation_angle(segmentation_mask: np.ndarray, femur_label: int, tibia_label: int,
                                  side: str = 'left', plot: bool | plt.Axes = False) -> float:
    """
    Calculate the knee rotation angle.

    The angle between the line connecting the posterior femoral condyles and the line
    connecting the posterior tibial condyles. The two side-dependent posterior-condyle
    orientations decide whether the proximal and distal angles add or subtract.
    :param segmentation_mask: A 3D segmentation mask of the knee.
    :param femur_label: The segmentation label of the femur.
    :param tibia_label: The segmentation label of the tibia.
    :param side: Side of the image (not patient!), either 'left' or 'right'.
    :param plot: A sequence of two matplotlib Axes to draw the reference lines on, or False.
    :return: The knee rotation angle in degrees.
    """
    G.validate_side(side)

    femur_mask = np.where(segmentation_mask == femur_label, 1, 0)
    proximal_layer, femur_start, femur_end = get_knee_reference_line(femur_mask, 'femur')
    femur_start, femur_end = _order_condyle_points(femur_start, femur_end)
    proximal_line = femur_end - femur_start

    tibia_mask = np.where(segmentation_mask == tibia_label, 1, 0)
    distal_layer, tibia_start, tibia_end = get_knee_reference_line(tibia_mask, 'tibia')
    tibia_start, tibia_end = _order_condyle_points(tibia_start, tibia_end)
    distal_line = tibia_end - tibia_start

    x = np.array([-1, 0, 0])
    proximal_angle = calculate_angle_between_vectors(proximal_line, x)
    distal_angle = calculate_angle_between_vectors(distal_line, x)

    proximal_orientation = (femur_end[1] - femur_start[1]) if side == 'left' else (femur_start[1] - femur_end[1])
    distal_orientation = (tibia_end[1] - tibia_start[1]) if side == 'left' else (tibia_start[1] - tibia_end[1])

    if np.sign(proximal_orientation) != np.sign(distal_orientation):
        angle = proximal_angle + distal_angle
    else:
        angle = proximal_angle - distal_angle

    if angle == 180:
        angle = 0

    if plot is not False:
        plot[0].imshow(segmentation_mask[:, :, proximal_layer].T)
        plot[0].plot([femur_start[0], femur_end[0]], [femur_start[1], femur_end[1]], color='red')
        plot[0].set_title(f'Angle: {angle:.2f}°')
        plot[1].imshow(segmentation_mask[:, :, distal_layer].T)
        plot[1].plot([tibia_start[0], tibia_end[0]], [tibia_start[1], tibia_end[1]], color='red')
    return angle


def calculate_joint_line_convergence_angle(segmentation_mask: Segmentation, femur_label: int, tibia_label: int,
                                           side: str = 'left', plot: bool | plt.Axes = False) -> float:
    """
    Calculate the joint line convergence angle (JLCA).

    Builds a femoral line between the two condyle centroids and a tibial line between
    the medial/lateral plateau apices, projects both to the coronal plane and returns
    the acute angle between them.
    :param segmentation_mask: A 3D Segmentation of the knee.
    :param femur_label: The segmentation label of the femur.
    :param tibia_label: The segmentation label of the tibia.
    :param side: Side of the image (not patient!), either 'left' or 'right'.
    :param plot: A sequence of two matplotlib Axes to draw the reference lines on, or False.
    :return: The joint line convergence angle in degrees.
    """
    # --- Femoral reference line: line between the two condyle centroids ---
    array = np.where(segmentation_mask.array == femur_label, 1, 0)

    saw_two = False
    split_layer = None
    for i in range(array.shape[2]):
        components = extract_connected_components_2d(array[:, :, i])
        if len(components) == 2:
            saw_two = True
        if saw_two and len(components) == 1:
            split_layer = i
            break
    if not (saw_two and split_layer is not None):
        raise RuntimeError('Could not find condyles!')

    components = None
    condyle_1 = None
    for i in range(array.shape[2] - 1, -1, -1):
        components = extract_connected_components_2d(array[:, :, i])
        if len(components) == 2:
            smaller = components[0] if np.count_nonzero(components[0]) < np.count_nonzero(components[1]) else components[1]
            c = center_of_mass(smaller)
            condyle_1 = np.array([c[0], c[1], i])
            break
    if condyle_1 is None:
        raise RuntimeError('Could not find condyles!')

    condyle_2 = None
    for i in range(array.shape[2] - 1, -1, -1):
        components = extract_connected_components_2d(array[:, :, i])
        if len(components) == 1:
            c = center_of_mass(components[0])
            condyle_2 = np.array([c[0], c[1], i])
            break
    if condyle_2 is None:
        raise RuntimeError('Could not find condyles!')

    proximal_reference_line = condyle_2 - condyle_1

    # --- Tibial reference line: line between medial/lateral plateau apices ---
    array = np.where(segmentation_mask.array == tibia_label, 1, 0)
    if np.count_nonzero(array) == 0:
        raise ValueError('No tibia found!')

    first_layer = next(i for i in range(array.shape[2]) if np.count_nonzero(array[:, :, i]) > 0)
    eminence = center_of_mass(array[:, :, first_layer])
    eminence = np.array([eminence[0], eminence[1], first_layer])

    points = np.argwhere(array == 1)
    leftmost = points[points[:, 0].argmin()]
    leftmost = np.array([leftmost[0], leftmost[1], first_layer])
    rightmost = points[points[:, 0].argmax()]
    rightmost = np.array([rightmost[0], rightmost[1], first_layer])

    d_1 = np.linalg.norm(leftmost - eminence)
    d_2 = np.linalg.norm(rightmost - eminence)

    # restrict to points well lateral / medial of the eminence (>2/3 of the apex distance)
    left_points = points[points[:, 0] < (eminence[0] - d_1 * 2 / 3)]
    right_points = points[points[:, 0] > (eminence[0] + d_2 * 2 / 3)]

    leftmost_point = left_points[left_points[:, 2].argmin()]
    rightmost_point = right_points[right_points[:, 2].argmin()]
    distal_reference_line = leftmost_point - rightmost_point

    # project to the coronal plane
    proximal_reference_line[1] = 0
    distal_reference_line[1] = 0

    angle = G.fold_to_acute(calculate_angle_between_vectors(proximal_reference_line, distal_reference_line))

    if plot is not False:
        femur = np.where(segmentation_mask.array == femur_label, 1, 0)
        plot[0].imshow(femur[:, (int(condyle_1[1]) + int(condyle_2[1])) // 2].T)
        plot[0].plot([condyle_1[0], condyle_2[0]], [condyle_1[2], condyle_2[2]], 'g-')
        plot[0].set_title('Femoral JLCA line')
        tibia = np.where(segmentation_mask.array == tibia_label, 1, 0)
        plot[1].imshow(tibia[:, (int(leftmost_point[1]) + int(rightmost_point[1])) // 2].T)
        plot[1].plot([leftmost_point[0], rightmost_point[0]], [leftmost_point[2], rightmost_point[2]], 'g-')
        plot[1].set_title('Tibial JLCA line')

    return angle
