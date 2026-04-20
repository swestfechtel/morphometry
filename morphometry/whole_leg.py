import numpy as np
from numpy import floating

from morphometry.hip import get_femoral_head_center_ct
from morphometry.knee import get_knee_center
from morphometry.utils import get_minimum_distance_between_line_and_point, get_point_orientation_to_vertical_line, \
    get_minimum_distance_between_line_and_point_, get_vector_through_point_perpendicular_to_line, \
    calculate_angle_between_vectors
from morphometry.image_io import Image, Segmentation, split_ct_image
from typing import Tuple, Any
from scipy.ndimage import center_of_mass


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


if __name__ == '__main__':
    from morphometry.image_io import Image
    from matplotlib import pyplot as plt

    image = Image('nibabel')
    image.read_image('/home/simon/Data/Augsburg_large/preprocessed/PA000001/ankle_seg.nii.gz')
    image.transform_coordinate_system()

    distal_articulating_surface_left = get_distal_articulating_surface(image.array[:image.array.shape[0]//2], 1)
    distal_articulating_surface_right = get_distal_articulating_surface(image.array[image.array.shape[0]//2:], 1)

    fig, ax = plt.subplots(ncols=2)
    ax[0].imshow(image.array[:, :, distal_articulating_surface_left].T, cmap='gray')
    ax[0].set_title(f'Distal articulating surface at slice {distal_articulating_surface_left}')
    ax[0].axis('off')
    ax[1].imshow(image.array[:, :, distal_articulating_surface_right].T, cmap='gray')
    ax[1].set_title(f'Distal articulating surface at slice {distal_articulating_surface_right}')
    ax[1].axis('off')
    plt.tight_layout()
    plt.show()