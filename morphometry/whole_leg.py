import numpy as np
import SimpleITK as sitk
from morphometry.hip import get_femoral_head_center
from morphometry.knee import get_knee_center
from morphometry.utils import get_minimum_distance_between_line_and_point
from typing import Tuple
from scipy.ndimage import center_of_mass

def get_mikulicz_line(hip_mask: np.ndarray, ankle_mask: np.ndarray, side: str = 'left', x_ratio: float = 1.) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get the Mikulicz line of one image side.

    The Mikulicz line is the line connecting the center of the femoral head and the center of the ankle.
    :param hip_mask: A 3D segmentation mask of the hip, where the femur should be labeled 1 and everything else 0.
    :param ankle_mask: A 3D segmentation mask of the ankle, where tibia and fibula should both be labeled 1 and
    everything else 0.
    :param side: The side of the image (not patient!), either 'left' or 'right'.
    :param x_ratio: Correction factor for slice thickness.
    :return: The proximal and distal points of the Mikulicz line, i.e. the center of the femoral head and the center
    of the ankle.
    """
    assert side in ['left', 'right'], 'Side must be either "left" or "right"'
    r, femoral_head_center = get_femoral_head_center(hip_mask, side=side, x_ratio=x_ratio)
    ankle_center = np.array(center_of_mass(ankle_mask))

    return femoral_head_center, ankle_center


def calculate_mikulicz_deviation(hip_mask: np.ndarray, knee_mask: np.ndarray, ankle_mask: np.ndarray, hip_image: sitk.Image, knee_image: sitk.Image, ankle_image: sitk.Image, side:str = 'left', x_ratio: float = 1., proximal_femur_label: int = 1, distal_femur_label: int = 1, proximal_tibia_label: int = 2, distal_tibia_label: int = 1, distal_fibula_label: int = 2, hip_x_axis_flipped: bool = False) -> float:
    """
    Calculate the deviation of the center of the knee from the Mikulicz line.
    :param hip_mask: A 3D segmentation mask of the hip.
    :param knee_mask: A 3D segmentation mask of the knee.
    :param ankle_mask: A 3D segmentation mask of the ankle.
    :param hip_image: A SimpleITK image of the hip.
    :param knee_image: A SimpleITK image of the knee.
    :param ankle_image: A SimpleITK image of the ankle.
    :param side: The side of the image (not patient!), either 'left' or 'right'.
    :param x_ratio: Correction factor for slice thickness.
    :param proximal_femur_label: The segmentation label of the femur at hip level.
    :param distal_femur_label: The segmentation label of the femur at knee level.
    :param proximal_tibia_label: The segmentation label of the tibia at knee level.
    :param distal_tibia_label: The segmentation label of the tibia at ankle level.
    :param distal_fibula_label: The segmentation label of the fibula at ankle level.
    :param hip_x_axis_flipped: Whether the x-axis of the hip mask was flipped. Depending on the data, this may be
    necessary since all code expects layer indices to increase from superior to inferior.
    :return: The deviation of the center of the knee from the Mikulicz line.
    """
    # prepare masks
    hip_mask = np.where(hip_mask == proximal_femur_label, 1, 0)

    knee_mask = np.where(knee_mask == distal_femur_label, 1, knee_mask)  # set distal femur to 1
    knee_mask = np.where(knee_mask == proximal_tibia_label, 1, knee_mask)  # set proximal tibia to 1
    knee_mask = np.where(knee_mask == 1, 1, 0)  # null everything else

    ankle_mask = np.where(ankle_mask == distal_tibia_label, 1, ankle_mask)  # set distal tibia to 1
    ankle_mask = np.where(ankle_mask == distal_fibula_label, 1, ankle_mask)  # set distal fibula to 1
    ankle_mask = np.where(ankle_mask == 1, 1, 0)  # null everything else

    femoral_head_center, ankle_center = get_mikulicz_line(hip_mask, ankle_mask, side=side, x_ratio=x_ratio)
    knee_center = get_knee_center(knee_mask)

    hip_layer = hip_mask.shape[0] - femoral_head_center[0] if hip_x_axis_flipped else femoral_head_center[0]  # flip back if necessary to align with sitk image
    fhc_world = np.array(hip_image.TransformContinuousIndexToPhysicalPoint((femoral_head_center[2], femoral_head_center[1], hip_layer)))  # remember that sitk and numpy ordering for coordinates are reversed)
    kc_world = np.array(knee_image.TransformContinuousIndexToPhysicalPoint((knee_center[2], knee_center[1], knee_center[0])))
    ac_world = np.array(ankle_image.TransformContinuousIndexToPhysicalPoint((ankle_center[2], ankle_center[1], ankle_center[0])))

    print(f'Femoral head center: {fhc_world}, Knee center: {kc_world}, Ankle center: {ac_world}')
    # TODO: determine if the knee is medial or lateral to the Mikulicz line

    return get_minimum_distance_between_line_and_point(fhc_world, ac_world, kc_world)