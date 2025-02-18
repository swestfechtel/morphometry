import numpy as np
from numpy import floating

from morphometry.hip import get_femoral_head_center
from morphometry.knee import get_knee_center
from morphometry.utils import get_minimum_distance_between_line_and_point, get_point_orientation_to_vertical_line, \
    get_minimum_distance_between_line_and_point_
from morphometry.image_io import Image
from typing import Tuple, Any
from scipy.ndimage import center_of_mass

def get_mikulicz_line(hip_mask: np.ndarray, ankle_mask: np.ndarray, side: str = 'left', x_ratio: float = 1., isotropic: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get the Mikulicz line of one image side.

    The Mikulicz line is the line connecting the center of the femoral head and the center of the ankle.
    :param hip_mask: A 3D segmentation mask of the hip, where the femur should be labeled 1 and everything else 0.
    :param ankle_mask: A 3D segmentation mask of the ankle, where tibia and fibula should both be labeled 1 and
    everything else 0.
    :param side: The side of the image (not patient!), either 'left' or 'right'.
    :param x_ratio: Correction factor for slice thickness.
    :param isotropic: Whether the image has isotropic voxels.
    :return: The proximal and distal points of the Mikulicz line, i.e. the center of the femoral head and the center
    of the ankle.
    """
    assert side in ['left', 'right'], 'Side must be either "left" or "right"'
    r, femoral_head_center = get_femoral_head_center(hip_mask, side=side, x_ratio=x_ratio, isotropic=isotropic)
    ankle_center = np.array(center_of_mass(ankle_mask))

    return femoral_head_center, ankle_center


def calculate_mikulicz_deviation(hip_image: Image, knee_image: Image, ankle_image: Image, side:str = 'left', x_ratio: float = 1., proximal_femur_label: int = 1, distal_femur_label: int = 1, proximal_tibia_label: int = 2, distal_tibia_label: int = 1, distal_fibula_label: int = 2) -> float:
    """
    Calculate the deviation of the center of the knee from the Mikulicz line.
    :param hip_image: An Image object of the hip segmentation mask.
    :param knee_image: An Image object of the knee segmentation mask.
    :param ankle_image: An Image object of the ankle segmentation mask.
    :param side: The side of the image (not patient!), either 'left' or 'right'.
    :param x_ratio: Correction factor for slice thickness.
    :param proximal_femur_label: The segmentation label of the femur at hip level.
    :param distal_femur_label: The segmentation label of the femur at knee level.
    :param proximal_tibia_label: The segmentation label of the tibia at knee level.
    :param distal_tibia_label: The segmentation label of the tibia at ankle level.
    :param distal_fibula_label: The segmentation label of the fibula at ankle level.
    :return: The deviation of the center of the knee from the Mikulicz line.
    """
    # prepare masks
    hip_mask = np.where(hip_image.get_array() == proximal_femur_label, 1, 0)

    knee_mask = np.where(knee_image.get_array() == distal_femur_label, 1, knee_image.get_array())  # set distal femur to 1
    knee_mask = np.where(knee_mask == proximal_tibia_label, 1, knee_mask)  # set proximal tibia to 1
    knee_mask = np.where(knee_mask == 1, 1, 0)  # null everything else

    ankle_mask = np.where(ankle_image.get_array() == distal_tibia_label, 1, ankle_image.get_array())  # set distal tibia to 1
    ankle_mask = np.where(ankle_mask == distal_fibula_label, 1, ankle_mask)  # set distal fibula to 1
    ankle_mask = np.where(ankle_mask == 1, 1, 0)  # null everything else

    femoral_head_center, ankle_center = get_mikulicz_line(hip_mask, ankle_mask, side=side, x_ratio=x_ratio)
    knee_center = get_knee_center(knee_mask)

    fhc_world = np.array(hip_image.transform_index_to_physical_point(femoral_head_center))
    kc_world = np.array(knee_image.transform_index_to_physical_point(knee_center))
    ac_world = np.array(ankle_image.transform_index_to_physical_point(ankle_center))

    # switch coordinates back to numpy ordering
    fhc_world = np.array([fhc_world[2], fhc_world[1], fhc_world[0]])
    kc_world = np.array([kc_world[2], kc_world[1], kc_world[0]])
    ac_world = np.array([ac_world[2], ac_world[1], ac_world[0]])

    print(f'Femoral head center: {fhc_world}, Knee center: {kc_world}, Ankle center: {ac_world}')
    # discard y coordinate since we are only interested in the deviation in the x-z plane
    # + minimum distance method is only defined for 2D points as of now.
    fhc_world_2d = np.array([fhc_world[0], fhc_world[-1]])
    kc_world_2d = np.array([kc_world[0], kc_world[-1]])
    ac_world_2d = np.array([ac_world[0], ac_world[-1]])
    orientation = get_point_orientation_to_vertical_line(fhc_world_2d, ac_world_2d, kc_world_2d)
    print(f'Knee center is to the {orientation} image side of the Mikulicz line.')

    return get_minimum_distance_between_line_and_point(fhc_world_2d, ac_world_2d, kc_world_2d)


def calculate_leg_length(hip_image: Image, ankle_image: Image) -> floating[Any]:
    """
    Calculate the length of the leg.
    :param hip_image: An Image object of the hip segmentation mask.
    :param ankle_image: An Image object of the ankle segmentation mask.
    :return: The length of the leg.
    """
    hip_most_proximal_layer = np.min(np.argwhere(hip_image.get_array())[:, 0])
    ankle_most_distal_layer = np.max(np.argwhere(ankle_image.get_array())[:, 0])

    hip_com = center_of_mass(hip_image.get_array()[hip_most_proximal_layer])
    hip_com = (hip_most_proximal_layer, hip_com[0], hip_com[1])

    ankle_com = center_of_mass(ankle_image.get_array()[ankle_most_distal_layer])
    ankle_com = (ankle_most_distal_layer, ankle_com[0], ankle_com[1])

    proximal_world = hip_image.transform_index_to_physical_point(hip_com)
    distal_world = ankle_image.transform_index_to_physical_point(ankle_com)

    return np.linalg.norm(np.array(proximal_world) - np.array(distal_world))
