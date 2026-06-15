import numpy as np
from typing import Tuple
from scipy.ndimage import center_of_mass

from morphometry.hip import get_femoral_head_center_ct
from morphometry.image_io import Segmentation, split_ct_image


def get_distal_articulating_surface(ankle_image: np.ndarray, tibia_label: int = 1) -> int:
    """
    Get the slice index of the distal articulating surface of the tibia.
    :param ankle_image: A 3D segmentation mask of the ankle.
    :param tibia_label: The segmentation label of the tibia.
    :return: The slice index of the distal articulating surface of the tibia.
    """
    changes = list()
    previous_slice_size = 0
    first_slice_found = False

    for i in range(ankle_image.shape[2]):
        slice_mod = np.where(ankle_image[:, :, i] == tibia_label, 1, 0)

        if np.count_nonzero(slice_mod) == 0:
            if not first_slice_found:
                changes.append(0)
                continue
            else:
                break

        if not first_slice_found:
            first_slice_found = True
            previous_slice_size = np.count_nonzero(slice_mod)

        slice_size = np.count_nonzero(slice_mod)

        changes.append(np.abs(slice_size - previous_slice_size))
        previous_slice_size = slice_size

    changes = np.array(changes)
    largest_change = np.argmax(changes)

    return largest_change - 1  # -1 because we want the slice before the change, which is the distal articulating surface

def get_mechanical_axis(whole_leg_image: Segmentation, femur_label: int = 1, tibia_label: int = 2, fibula_label: int = 3, patella_label: int = 5, hip_label: int = 7, side: str = 'left') -> Tuple[np.ndarray, np.ndarray]:
    """
    Get the Mikulicz line of one image side.

    The Mikulicz line is the line connecting the center of the femoral head and the center of the ankle.
    :param whole_leg_image: An Image object of the whole leg segmentation mask.
    :param femur_label: The segmentation label of the femur.
    :param tibia_label: The segmentation label of the tibia.
    :param fibula_label: The segmentation label of the fibula.
    :param patella_label: The segmentation label of the patella.
    :param hip_label: The segmentation label of the hip.
    :param side: The side of the image (not patient!), either 'left' or 'right'.
    :return: The proximal and distal points of the Mikulicz line, i.e. the center of the femoral head and the center
    of the ankle.
    """
    assert side in ['left', 'right'], 'Side must be either "left" or "right"'

    r, femoral_head_center = get_femoral_head_center_ct(whole_leg_image, femur_label, side)
    _, masks = split_ct_image(whole_leg_image, None, femur_label=femur_label, fibula_label=fibula_label, patella_label=patella_label, hip_label=hip_label)
    ankle_mask = masks[2]

    ankle_mask = np.where(ankle_mask.array == tibia_label, 1, 0)  # set distal tibia to 1
    ankle_center_index = get_distal_articulating_surface(ankle_mask, 1)
    ankle_center = np.array(center_of_mass(ankle_mask[:, :, ankle_center_index]))
    ankle_center = np.array([ankle_center[0], ankle_center[1], ankle_center_index])
    ankle_center = whole_leg_image.transform_index_to_physical_point(ankle_center)

    return femoral_head_center, ankle_center
