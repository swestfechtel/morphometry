"""Whole-leg measurement functions: mechanical-axis deviation, hip-knee-ankle
angle, and bone lengths (MRI two-image and whole-leg CT variants).

The Mikulicz line and distal articulating surface are resolved by get_* helpers in
``morphometry.whole_leg``; this module turns them into the measurements.
"""
import numpy as np
from numpy import floating
from typing import Any, Tuple
from scipy.ndimage import center_of_mass

from morphometry.image_io import Image, Segmentation, split_ct_image
from morphometry.knee import get_knee_center
from morphometry.whole_leg import get_mechanical_axis, get_distal_articulating_surface
from morphometry.utils import get_vector_through_point_perpendicular_to_line, calculate_angle_between_vectors


def calculate_mechanical_axis_deviation(whole_leg_image: Segmentation, femur_label: int = 1, tibia_label: int = 2, fibula_label: int = 3, patella_label: int = 5, hip_label: int = 7, side:str = 'left') -> float:
    """
    Calculate the deviation of the center of the knee from the Mikulicz line.
    :param whole_leg_image: An Image object of the whole leg segmentation mask.
    :param femur_label: The segmentation label of the femur.
    :param tibia_label: The segmentation label of the tibia.
    :param fibula_label: The segmentation label of the fibula.
    :param patella_label: The segmentation label of the patella.
    :param hip_label: The segmentation label of the hip.
    :param side: The side of the image (not patient!), either 'left' or 'right'.
    :return: The deviation of the center of the knee from the Mikulicz line.
    """

    femoral_head_center, ankle_center = get_mechanical_axis(whole_leg_image, femur_label=femur_label, tibia_label=tibia_label, fibula_label=fibula_label, patella_label=patella_label, hip_label=hip_label, side=side)
    _, masks = split_ct_image(whole_leg_image, None)
    knee_mask = masks[1]

    knee_mask = knee_mask.array.copy()
    knee_mask = np.where(knee_mask == tibia_label, 1, 0)
    knee_mask = np.where(knee_mask > 1, 0, knee_mask)

    knee_center = get_knee_center(knee_mask)
    knee_center = whole_leg_image.transform_index_to_physical_point(knee_center)

    femoral_head_center[1] = 0  # project to frontal plane
    knee_center[1] = 0
    ankle_center[1] = 0

    mikulicz_line = femoral_head_center - ankle_center
    _, projection_vector = get_vector_through_point_perpendicular_to_line(femoral_head_center, mikulicz_line, knee_center)
    d = np.linalg.norm(knee_center - projection_vector)

    if side == 'left':  # remember here side refers to image side
        if projection_vector[0] > knee_center[0]:  # if the mikulicz line is lateral to the knee, it is a medial deviation, which is encoded as a negative value
            return -d

    if side == 'right':
        if projection_vector[0] < knee_center[0]:
            return -d

    return d

def calculate_hip_knee_ankle_angle(whole_leg_image: Segmentation, femur_label: int = 1, tibia_label: int = 2, fibula_label: int = 3, patella_label: int = 5, hip_label: int = 7, side: str = 'left') -> float:
    """
    Calculate the hip-knee-ankle angle.
    :param whole_leg_image: An Image object of the whole leg segmentation mask.
    :param femur_label: The segmentation label of the femur.
    :param tibia_label: The segmentation label of the tibia.
    :param fibula_label: The segmentation label of the fibula.
    :param patella_label: The segmentation label of the patella.
    :param hip_label: The segmentation label of the hip.
    :param side: The side of the image (not patient!), either 'left' or 'right'.
    :return: The hip-knee-ankle angle in degrees.
    """
    femoral_head_center, ankle_center = get_mechanical_axis(whole_leg_image, femur_label=femur_label, tibia_label=tibia_label, fibula_label=fibula_label, patella_label=patella_label, hip_label=hip_label, side=side)
    _, masks = split_ct_image(whole_leg_image, None)
    knee_mask = masks[1]

    knee_mask = knee_mask.array.copy()
    knee_mask = np.where(knee_mask == tibia_label, 1, 0)
    knee_mask = np.where(knee_mask > 1, 0, knee_mask)
    knee_center = get_knee_center(knee_mask)
    knee_center = whole_leg_image.transform_index_to_physical_point(knee_center)

    hip_knee_vector = knee_center - femoral_head_center
    knee_ankle_vector = ankle_center - knee_center

    # Project to coronal plane
    hip_knee_vector[1] = 0
    knee_ankle_vector[1] = 0

    angle = calculate_angle_between_vectors(hip_knee_vector, knee_ankle_vector)

    return angle

def calculate_bone_length(proximal_image: Image, distal_image: Image, proximal_label: int, distal_label: int, tibia=False) -> floating[Any]:
    """
    Calculate the length of the femur, tibia or whole leg.
    :param proximal_image: An Image object of the proximal segmentation mask.
    :param distal_image: An Image object of the distal segmentation mask.
    :param proximal_label: The segmentation label of the bone in the proximal image.
    :param distal_label: The segmentation label of the bone in the distal image.
    :param tibia: Whether to calculate the length of the tibia (True).
    :return: The length of the bone.
    """
    proximal_cleaned = np.where(proximal_image.array == proximal_label, 1, 0)
    distal_cleaned = np.where(distal_image.array == distal_label, 1, 0)

    most_proximal_layer = np.min(np.argwhere(proximal_cleaned)[:, 2])
    if not tibia:
        most_distal_layer = np.max(np.argwhere(distal_cleaned)[:, 2])
    else:
        most_distal_layer = get_distal_articulating_surface(distal_cleaned)

    proximal_layer_centroid = center_of_mass(proximal_cleaned[:, :, most_proximal_layer])
    proximal_layer_centroid = (proximal_layer_centroid[0], proximal_layer_centroid[1], most_proximal_layer)

    distal_layer_centroid = center_of_mass(distal_cleaned[:, :, most_distal_layer])
    distal_layer_centroid = (distal_layer_centroid[0], distal_layer_centroid[1], most_distal_layer)

    proximal_world = proximal_image.transform_index_to_physical_point(proximal_layer_centroid)
    distal_world = distal_image.transform_index_to_physical_point(distal_layer_centroid)

    return np.linalg.norm(np.array(proximal_world) - np.array(distal_world))

def calculate_bone_length_ct(whole_leg_image: Segmentation, segmentation_label: int) -> floating[Any]:
    """
    Calculate the length of the femur or tibia in a whole leg CT segmentation.
    :param whole_leg_image: An Image object of the whole leg segmentation mask.
    :param segmentation_label: The segmentation label of the bone in the whole leg image.
    :return: The length of the bone.
    """
    cleaned = np.where(whole_leg_image.array == segmentation_label, 1, 0)

    most_proximal_layer = np.min(np.argwhere(cleaned)[:, 2])
    most_distal_layer = np.max(np.argwhere(cleaned)[:, 2])

    proximal_layer_centroid = center_of_mass(cleaned[:, :, most_proximal_layer])
    proximal_layer_centroid = (proximal_layer_centroid[0], proximal_layer_centroid[1], most_proximal_layer)

    distal_layer_centroid = center_of_mass(cleaned[:, :, most_distal_layer])
    distal_layer_centroid = (distal_layer_centroid[0], distal_layer_centroid[1], most_distal_layer)

    proximal_world = whole_leg_image.transform_index_to_physical_point(proximal_layer_centroid)
    distal_world = whole_leg_image.transform_index_to_physical_point(distal_layer_centroid)

    return np.linalg.norm(np.array(proximal_world) - np.array(distal_world))
