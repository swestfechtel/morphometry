import math
import numpy as np
import nibabel as nib
from matplotlib import pyplot as plt
from scipy.ndimage import center_of_mass
from scipy.optimize import curve_fit
from skimage.measure import find_contours
from morphometry.hip import get_femoral_head_center, get_femoral_head_center_ct
from morphometry.knee import get_knee_reference_line
from morphometry.utils import rotate_mask_dorsal_points, find_notch, transform_point, points_on_circle, \
    get_contour, calculate_angle_between_vectors, circle_fit, draw_circle, draw_line, extract_connected_components_2d
from morphometry.image_io import Image, Segmentation, split_ct_image
from typing import Tuple


def contour_femoral_neck(mask: np.ndarray, contour: np.ndarray, layer_selected: int, center: np.ndarray, r: float) -> \
        Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Get the contour of the femoral neck.
    :param mask: 3D segmentation mask of the proximal femur.
    :param contour: contour of segmentation mask
    :param layer_selected: index of layer with femoral neck
    :param center: center coordinates of femur head
    :param r: radius of femur head
    :return: adjusted mask, contour of the femoral neck and radii of two circles around the femoral head center with
    0.9 and 1.1 times the radius of the femoral head
    """
    # rotate mask and find notch to regulate the radii of the spheres

    mask_new = mask[:, :, layer_selected].copy()

    rotated_mask, angle1 = rotate_mask_dorsal_points(mask_new,
                                                     np.array(center_of_mass(mask_new)).astype(np.int16))

    rotated_mask, angle2 = rotate_mask_dorsal_points(rotated_mask,
                                                     np.array(center_of_mass(rotated_mask)).astype(np.int16))
    angle = angle1 + angle2

    notch_rot = find_notch(rotated_mask,
                           percentage=0.8,
                           thresh=5,
                           break_after_first=True)
    if notch_rot is None:
        raise RuntimeError('No notch found.')

    rot_offset = np.array([
        _rot_dim - _orig_dim
        for _rot_dim, _orig_dim in zip(rotated_mask.shape, mask_new.shape)
    ])
    rot_center = np.array([int(
        (rotated_mask.shape[0] - 1) / 2), int((rotated_mask.shape[1] - 1) / 2)])
    if angle == 0:
        notch = notch_rot
    else:
        notch = transform_point(notch_rot,
                                rot_center,
                                -angle,
                                offset=-rot_offset / 2)

    # get 2 new radii a little closer and further to the center than the notch
    center_2d = np.array([center[0], center[1]])  # (sagittal, coronal)

    a = np.linalg.norm(notch - center_2d) / r
    r_1 = (-0.1 + a) * r
    r_2 = (0.1 + a) * r

    # mask_new = mask[layer_selected].copy()

    # get contour only between the two circles on the selected layer
    for s in range(mask.shape[0] - 1):  # all sagittal coordinates
        if r_2 ** 2 - (s - center[0]) ** 2 >= 0:
            c_a = int(round(math.sqrt(r_2 ** 2 - (s - center[0]) ** 2)))
            if center[1] + c_a < mask.shape[1]:
                contour[s, center[1] + c_a:, layer_selected] = 0
                mask_new[s, center[1] + c_a:] = 0  # mask_new is 2D
            if center[1] - c_a > 0:
                contour[s, :center[1] - c_a, layer_selected] = 0
                mask_new[s, :center[1] - c_a] = 0
        else:
            contour[s, :, layer_selected] = 0
            mask_new[s] = 0

        if r_1 ** 2 - (s - center[0]) ** 2 >= 0:
            c_b = int(round(math.sqrt(r_1 ** 2 - (s - center[0]) ** 2)))
            if center[1] - c_b > 0 and center[1] + c_b < mask.shape[1]:
                contour[s, center[1] - c_b:center[1] + c_b, layer_selected] = 0
                mask_new[s, center[1] - c_b:center[1] + c_b] = 0

    return mask_new, contour, r_1, r_2


def get_femoral_neck_center_lee(segmentation_mask: np.ndarray, center: np.ndarray, r: float) -> np.ndarray:
    """
    Get the centre of the femoral neck as described by Lee et al.

    :param segmentation_mask: A 3D segmentation mask of the femur.
    :param center: The center of the femoral head.
    :param r: The radius of the femoral head.
    :return: The centre of the femoral neck.
    """
    layer_selected = None
    for i in range(segmentation_mask.shape[2]):
        contour_length = len(find_contours(segmentation_mask[:, :, i],
                                           0.8))  # need to use find_contours here because it can detect disconnected contours
        on_circle = points_on_circle(segmentation_mask[:, :, i], r * 2, center[:2])

        if contour_length == 1 and len(on_circle) > 0:
            layer_selected = i
            break

    if layer_selected is None:
        raise ValueError('Could not find a suitable layer for the femoral neck.')

    contour = get_contour(segmentation_mask)

    femoral_neck, contour, r_1, r_2 = contour_femoral_neck(segmentation_mask, contour,
                                                           layer_selected, center,
                                                           r)

    # center of femoral neck
    center_fn = np.array(center_of_mass(femoral_neck)).astype(np.int16)

    # get coordinates of the selected contour points
    contour_pts_l = np.nonzero(contour[:, :, layer_selected])

    distance_center = (contour_pts_l[0] -
                       center_fn[0]) ** 2 + (contour_pts_l[1] - center_fn[1]) ** 2
    r_3 = math.sqrt(np.median(distance_center)) * 1.5

    x = []
    y = []
    # get contour also only with small distance to center_fn on the selected layer
    # basically what this does is draw a circle with a certain radius around center_fn and exclude all contour points
    # that are outside of that circle
    for s in range(segmentation_mask.shape[0] - 1):  # all sagittal coordinates
        if r_3 ** 2 - (s - center_fn[0]) ** 2 >= 0:  # if less than r_3 from center_fn
            c_a = int(round(math.sqrt(r_3 ** 2 - (s - center_fn[0]) ** 2)))
            if center_fn[1] + c_a < segmentation_mask.shape[1]:  # if coronal coordinate of center_fn + c_a is within the mask, null contour points posterior to that
                contour[s, center_fn[1] + c_a:, layer_selected] = 0
                x.append(s)
                y.append(center_fn[1] + c_a)
            if center_fn[1] - c_a > 0:  # if coronal coordinate of center_fn - c_a is within the mask, null contour points anterior to that
                contour[s, :center_fn[1] - c_a, layer_selected] = 0
                x.append(s)
                y.append(center_fn[1] - c_a)
        else:
            contour[s, :, layer_selected] = 0

    contour_pts_l = np.nonzero(contour[:, :, layer_selected])

    # get index of the contour point after the largest gap
    # -> first contour point on the other side of the segmentation mask
    diff = np.ediff1d(contour_pts_l[1])
    ind_gap = np.argsort(diff)[-1] + 1

    def g(x, m):
        return m * x

    if diff.max() == 1:
        # lsf through the center and the contour points to get its slope
        popt, _ = curve_fit(g, contour_pts_l[0] - center[0],
                            contour_pts_l[1] - center[1])
        m_new = popt[0]
    else:
        # separated lsf through the center and the contour points on both sides
        # to get their slopes and calculate the mean
        popt1, _ = curve_fit(g, contour_pts_l[0][:ind_gap] - center[0],
                             contour_pts_l[1][:ind_gap] - center[1])
        popt2, _ = curve_fit(g, contour_pts_l[0][ind_gap:] - center[0],
                             contour_pts_l[1][ind_gap:] - center[1])
        m_new = np.mean([popt1[0], popt2[0]])

    # find endpoint for the reference line
    end = np.array([int(-80 + center[0]), int((-80) * m_new + center[1]), layer_selected])
    return end


def get_trochanter_major(hip_image: Image, femoral_head_centre: np.ndarray, segmentation_label: int = 1) -> np.ndarray:
    """
    Find the trochanter major.

    :param hip_image: An Image object of the hip segmentation mask.
    :param femoral_head_centre: The centre of the femoral head.
    :param segmentation_label: The label of the segmentation mask.
    :return: The coordinates of the trochanter major.
    """
    lateral_extents = np.zeros(hip_image.get_shape()[2])
    array = hip_image.array.copy()
    array = np.where(array == segmentation_label, 1, 0)  # convert to binary mask

    # iterate from superior to inferior and get the maximum lateral extent of each layer
    for i in range(hip_image.get_shape()[2]):
        if np.count_nonzero(array[:, :, i]):
            lateral_extents[i] = np.argwhere(array[:, :, i])[:, 0].min()
        else:
            lateral_extents[i] = 1000

    def check_plausibility() -> bool:
        """
        Check if the trochanter major on the given layer is plausible.

        :return: Whether the trochanter major is plausible.
        """
        return True
        tm = (int(trochanter_major_layer), int(trochanter_major[0]), int(trochanter_major[1]))
        fh = (int(femoral_head_centre[0]), int(femoral_head_centre[1]), int(femoral_head_centre[2]))
        tm_world = hip_image.transform_index_to_physical_point(tm)
        fh_world = hip_image.transform_index_to_physical_point(fh)
        distance = np.linalg.norm(np.array(tm_world) - np.array(fh_world))

        return True if (425 < distance < 515) else False

    # get the layer with the maximum lateral extent
    trochanter_major_layer = np.argmin(lateral_extents)
    while True:
        # on that layer, get the most lateral point, which should be the trochanter major
        try:
            trochanter_major_s = np.argwhere(array[:, :, trochanter_major_layer])[:,
                                 0].min()  # get the minimum sagittal coordinate, i.e. most lateral extent of this slice
            trochanter_major_c = np.median(np.argwhere(array[trochanter_major_s, :,
                                                       trochanter_major_layer]))  # select all points with the same sagittal coordinate and get the median coronal coordinate
            trochanter_major = np.array([trochanter_major_s, trochanter_major_c])
        except IndexError:
            raise RuntimeError('Could not find a plausible trochanter major.')

        if check_plausibility():
            break

        # if the trochanter major is not plausible, continue with the next layer
        trochanter_major_layer += 1

    return np.array([trochanter_major[0], trochanter_major[1], trochanter_major_layer])


def get_trochanter_minor(hip_image: Image, femoral_head_centre: np.ndarray, segmentation_label: int = 1, isotropic: bool = False) -> np.ndarray:
    """
    Find the trochanter minor.

    :param hip_image: An Image object of the hip segmentation mask.
    :param femoral_head_centre: The centre of the femoral head.
    :param segmentation_label: The label of the segmentation mask.
    :param isotropic: Whether the image has isotropic voxels.
    :return: The coordinates of the trochanter minor.
    """

    def check_plausibility(trochanter_minor) -> bool | float:
        """
        Check if the trochanter minor on the given layer is plausible.

        :return: Whether the trochanter minor is plausible.
        """
        tm = trochanter_minor.astype(np.int16)
        fh = femoral_head_centre.astype(np.int32)

        tm_world = hip_image.transform_index_to_physical_point(tm)
        fh_world = hip_image.transform_index_to_physical_point(fh)
        tm_world = np.array([tm_world[0], tm_world[2]])  # exclude coronal axis?
        fh_world = np.array([fh_world[0], fh_world[2]])

        distance_world = np.linalg.norm(np.array(tm_world) - np.array(fh_world))

        return True if 75 > distance_world > 40 else distance_world

    hip_mask = hip_image.array.copy()
    hip_mask = np.where(hip_mask == segmentation_label, 1, 0)  # convert to binary mask

    # start, stop, step = (hip_mask.shape[0], 0, -1) if side == 'right' else (0, hip_mask.shape[0], 1)
    start, stop, step = (hip_mask.shape[0] - 1, 0, -1)
    two_components_found = False
    candidate_1, candidate_2 = False, False
    components_large_enough = False
    for layer in range(start, stop, step):
        connected_components = extract_connected_components_2d(hip_mask[layer])
        if len(connected_components) == 2:
            two_components_found = True

            if isotropic and (np.count_nonzero(connected_components[0]) < 10 or np.count_nonzero(connected_components[1]) < 10):
                continue

            components_large_enough = True

            com_1 = center_of_mass(connected_components[0])
            com_2 = center_of_mass(connected_components[1])

            candidate_1 = np.array([layer, com_1[0], com_1[1]])
            candidate_2 = np.array([layer, com_2[0], com_2[1]])

            candidate_1 = check_plausibility(candidate_1)
            candidate_2 = check_plausibility(candidate_2)

            if (candidate_1 is not True) and (candidate_2 is not True):
                continue

            break
    else:
        if not two_components_found:
            raise RuntimeError('Could not find a plausible trochanter minor: no two components found.')
        elif not components_large_enough:
            raise RuntimeError(f'Could not find a plausible trochanter minor: components too small; isotropic={isotropic}.')
        elif (candidate_1 is not True) and (candidate_2 is not True):
            raise RuntimeError(f'Could not find a plausible trochanter minor: no plausible candidates found. Distances: {candidate_1}, {candidate_2}')
        else:
            raise RuntimeError('Could not find a plausible trochanter minor.')

    smaller_component = min(connected_components, key=lambda x: np.count_nonzero(x))

    start = layer
    most_distal_segmented_point_global = np.max(np.argwhere(hip_mask == segmentation_label)[:, 2])

    for layer in range(start, stop, step):
        sagittal_layer = hip_mask[layer]

        try:
            most_distal_segmented_point = np.max(np.argwhere(sagittal_layer == 1)[:, 1])
        except ValueError:
            # print(f'Something happened that should not happen - debug info. most_distal_segmented_point={most_distal_segmented_point}, most_distal_segmented_point_global={most_distal_segmented_point_global}, start={start}, stop={stop}, step={step}, layer={layer}')
            # plt.imshow(sagittal_layer)
            # plt.show()
            continue

        offset = 1 if not isotropic else 10
        if most_distal_segmented_point >= (most_distal_segmented_point_global - offset):  # if the point is at the bottom of the image, we have found the shaft
            break
    else:
        raise RuntimeError('Could not find the femoral shaft')

    com = center_of_mass(smaller_component)
    axial_layer = int(com[1])
    femur_shaft = hip_mask[:, :, axial_layer].copy()
    femur_shaft[:layer] = 0

    # axial_layer_com = center_of_mass(hip_mask[:, :, axial_layer])
    axial_layer_com = center_of_mass(femur_shaft)

    return np.array([axial_layer_com[0], axial_layer_com[0], axial_layer])


def get_femoral_neck_base(hip_image: Image, femoral_head_centre: np.ndarray, segmentation_label: int = 1, isotropic: bool = False) -> np.ndarray:
    """
    Get the centre of the base of the femoral neck as described by Murphy et al.

    Alternative implementation where search space is restricted to all slices inferior to torchanter major.
    :param hip_image: An Image object of the hip segmentation mask.
    :param femoral_head_centre: The centre of the femoral head.
    :param segmentation_label: The label of the segmentation mask.
    :param isotropic: Whether the image has isotropic voxels.
    :return: The centre of the base of the femoral neck.
    """
    trochanter_minor = get_trochanter_minor(hip_image=hip_image, femoral_head_centre=femoral_head_centre, segmentation_label=segmentation_label, isotropic=isotropic)
    trochanter_minor_layer = int(trochanter_minor[2])

    array = hip_image.array.copy()
    array = np.where(array == segmentation_label, 1, 0)

    center = np.array(center_of_mass(array[:, :, trochanter_minor_layer])).astype(np.int16)
    center = np.array([center[0], center[1], trochanter_minor_layer])
    # center = trochanter_minor.astype(np.int16)
    return center


def get_trochanter_major_center(hip_image: Image, femoral_head_centre: np.ndarray, segmentation_label: int = 1) -> Tuple[np.ndarray, float]:
    """
    Get the centre of the trochanter major at neck base level as described by Tomczak et al.
    :param hip_image: An Image object of the hip segmentation mask.
    :param femoral_head_centre: The centre of the femoral head.
    :param segmentation_label: The label of the segmentation mask.
    :return: The centre and radius of the trochanter major at neck base level.
    """
    trochanter_major = get_trochanter_major(hip_image, femoral_head_centre, segmentation_label)
    trochanter_major_layer = int(trochanter_major[2])

    array = hip_image.array.copy()
    array = np.where(array == segmentation_label, 1, 0)  # convert to binary mask

    lateral_mask_point = np.argwhere(array[:, :, trochanter_major_layer])[:, 0].min()
    medial_mask_point = np.argwhere(array[:, :, trochanter_major_layer])[:, 0].max()

    split_point = (lateral_mask_point + medial_mask_point) // 2
    lateral_mask = array[:, :, trochanter_major_layer].copy()
    lateral_mask[split_point:] = 0  # null everything medial to the split point
    center, radius = circle_fit(lateral_mask)
    center = np.array([center[0], center[1], trochanter_major_layer]).astype(np.int16)
    return center, radius


def get_proximal_reference_line(hip_image: Image, side: str = 'left',
                                method: str = 'lee', segmentation_label: int = 1,
                                x_ratio: float = 1., isotropic: bool = False) -> Tuple[
                                                                                     np.ndarray, np.ndarray, float, float] | \
                                                                                 Tuple[np.ndarray, np.ndarray, float]:
    """
    Get the proximal reference line of a segmentation mask for calculating the femoral torsion.
    :param hip_image: An Image object of the hip segmentation mask.
    :param side: Side of the image (not patient!), either 'left' or 'right'.
    :param method: The method to use for determining the reference line.
    :param segmentation_label: The label of the segmentation mask.
    :param x_ratio: Correction factor for slice thickness.
    :param isotropic: Whether the image has isotropic voxels.
    :return: The start and end points of the reference line, the radius of the femoral head centre and the radius of the trochanter major if method is tomczak.
    """
    assert side in ['left', 'right'], 'Side must be either "left" or "right"'
    assert method in ['lee', 'murphy',
                      'tomczak'], 'Currently, only "lee", "murphy" and "tomczak" are supported as methods.'

    r_fh, center, layer_high, layer_low = get_femoral_head_center(hip_image.array, side=side,
                                                                  segmentation_label=segmentation_label,
                                                                  return_layers=True, x_ratio=x_ratio,
                                                                  isotropic=isotropic)
    center = center.astype(np.int16)

    if side == 'right':  # flip sagittal axis to make this 'left-sided'
        segmentation_mask = hip_image.array[::-1]
        center[0] = center[0] + (segmentation_mask.shape[0] // 2 - center[0]) * 2  # flip sagittal coordinate of center
        tmp = nib.Nifti1Image(segmentation_mask, hip_image.affine, hip_image.header)
        hip_image = Image.from_nibabel(tmp)

    if method == 'lee':
        array = hip_image.array.copy()
        array = np.where(array == segmentation_label, 1, 0)
        end = get_femoral_neck_center_lee(array, center, r_fh)
    elif method == 'murphy':
        end = get_femoral_neck_base(hip_image=hip_image, femoral_head_centre=center, segmentation_label=segmentation_label, isotropic=isotropic)
    elif method == 'tomczak':
        end, r_tm = get_trochanter_major_center(hip_image, center, segmentation_label)

    layer_selected = end[2]

    if side == 'right':
        end[0] = end[0] + (hip_image.array.shape[0] // 2 - end[0]) * 2  # flip sagittal coordinate back
        center[0] = center[0] - (center[0] - hip_image.array.shape[0] // 2) * 2

    start: np.ndarray = np.array([center[0], center[1], center[2]])
    return (start, end, r_fh, r_tm) if method == 'tomczak' else (start, end, r_fh)


def get_proximal_reference_line_ct(femur_image: Segmentation, side: str = 'left', segmentation_label: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get the proximal reference line of a CT femur segmentation for calculating the femoral torsion.
    :param femur_image: A Segmentation object of the femur segmentation mask.
    :param side: The side of the image (not patient!), either 'left' or 'right'.
    :param segmentation_label: The label of the femur in the segmentation mask.
    :return: The start and end points of the reference line.
    """
    assert side in ['left', 'right'], 'Side must be either "left" or "right"'

    r_fh, fhc = get_femoral_head_center_ct(femur_image, segmentation_label, side)
    tmp = fhc + np.array([r_fh, 0, 0])  # rationale: point on the surface of the femoral head in arbitrary direction

    fhc = femur_image.transform_physical_point_to_index(fhc)
    tmp = femur_image.transform_physical_point_to_index(tmp)  # transform that point to index space
    r_fh = np.linalg.norm(fhc - tmp)  # recalculate radius in index space

    hip_start = np.min(np.argwhere(femur_image.array == segmentation_label)[:, 2])
    femur_end = np.max(np.argwhere(femur_image.array == segmentation_label)[:, 2])
    num_femur_slices = femur_end - hip_start + 1
    hip_end = hip_start + int(num_femur_slices * 0.30)  # take 30% of the femur length to find the femoral neck base

    mask = np.where(femur_image.array == segmentation_label, 1, 0)
    mask[:, :, :hip_start] = 0
    mask[:, :, hip_end:] = 0

    if side == 'right':
        mask = mask[::-1]
        fhc[0] = fhc[0] + (mask.shape[0] // 2 - fhc[0]) * 2  # flip sagittal coordinate of center
        tmp = nib.Nifti1Image(mask, femur_image.affine, femur_image.header)
        hip_image = Segmentation.from_nibabel(tmp)
    else:
        tmp = nib.Nifti1Image(mask, femur_image.affine, femur_image.header)
        hip_image = Segmentation.from_nibabel(tmp)

    end = get_femoral_neck_base(hip_image, fhc, segmentation_label, False)

    if side == 'right':
        end[0] = end[0] + (hip_image.array.shape[0] // 2 - end[0]) * 2  # flip sagittal coordinate back
        fhc[0] = fhc[0] - (fhc[0] - hip_image.array.shape[0] // 2) * 2

    start: np.ndarray = np.array([fhc[0], fhc[1], fhc[2]])
    return start, end


def calculate_femoral_torsion(hip_image: Image, knee_mask: np.ndarray, side: str = 'left', method: str = 'lee',
                              segmentation_label: int = 1, x_ratio: float = 1., plot: bool = False,
                              isotropic: bool = False, return_landmarks: bool = False) -> float | Tuple[float, plt.Figure] | Tuple[float, dict] | Tuple[float, plt.Figure, dict]:
    """
    Calculate the femoral torsion from a segmentation mask.
    :param hip_image: An Image object of the hip segmentation mask.
    :param knee_mask: A segmentation mask of the distal femur.
    :param side: Side of the image (not patient!), either 'left' or 'right'.
    :param method: The method to use for determining the proximal reference line.
    :param segmentation_label: The label of the segmentation mask.
    :param x_ratio: Correction factor for slice thickness.
    :param plot: Whether to plot the reference lines or not.
    :param isotropic: Whether the image has isotropic voxels.
    :param return_landmarks: Whether to return the landmarks as a dict.
    :return: The femoral torsion in degrees and optionally a matplotlib figure of the reference lines.
    """
    assert side in ['left', 'right'], 'Side must be either "left" or "right"'
    assert method in ['lee', 'murphy',
                      'tomczak'], 'Currently, only "lee", "murphy" and "tomczak" are supported as methods.'

    landmarks = get_proximal_reference_line(hip_image, side=side, method=method,
                                            segmentation_label=segmentation_label, x_ratio=x_ratio, isotropic=isotropic)
    hip_start = landmarks[0]
    fhc_layer = hip_start[2]
    hip_end = landmarks[1]
    hip_start[2] = hip_end[2]  # adjust axial coordinate to align layers
    r_fh = landmarks[2]
    if method == 'tomczak':
        r_tm = landmarks[3]

    hip_layer = hip_end[2]

    proximal_line = hip_end - hip_start
    x = np.array([-1, 0, 0]) if side == 'left' else np.array(
        [1, 0, 0])  # need to distinguish between left and right image side
    proximal_angle = calculate_angle_between_vectors(proximal_line, x)

    if proximal_angle > 90:
        proximal_angle = 180 - proximal_angle

    proximal_orientation = hip_end[1] - hip_start[1]  # positive if hip_end is posterior to hip_start

    """
    if proximal_orientation < 0:  # if hip_end is anterior to hip_start, the angle is negative
        proximal_angle = -proximal_angle
    """

    knee_mask = np.where(knee_mask == segmentation_label, 1, 0)
    knee_layer, knee_start, knee_end = get_knee_reference_line(knee_mask,
                                                               bone='femur', segmentation_label=segmentation_label)  # for both image sides, knee_start is to the right of knee_end
    if knee_start[0] < knee_end[0]:  # if this is somehow not the case, swap the points
        tmp = knee_start
        knee_start = knee_end
        knee_end = tmp

    distal_line = knee_end - knee_start

    x = np.array([-1, 0, 0])  # because end is always left of start, no need to distinguish between left and right
    distal_angle = calculate_angle_between_vectors(distal_line, x)

    if distal_angle > 90:
        distal_angle = 180 - distal_angle

    distal_orientation = (knee_end[1] - knee_start[1]) if side == 'left' else (knee_start[1] - knee_end[1])  # positive if knee_end is posterior to knee_start

    """
    if side == 'left':
        if distal_orientation < 0:  # lateral condyle is anterior to medial condyle
            distal_angle = -distal_angle
    else:
        if distal_orientation > 0:  # lateral condyle is anterior to medial condyle
            distal_angle = -distal_angle

    angle = proximal_angle - distal_angle
    """

    if np.sign(proximal_orientation) != np.sign(distal_orientation):  # add angles
        angle = proximal_angle + distal_angle
    else:
        angle = proximal_angle - distal_angle

    if not return_landmarks and not plot:
        return angle

    if plot:
        fig, ax = plt.subplots(1, 3)
        ax[0].imshow(np.where(hip_image.array[:, :, hip_layer] == 0, np.nan, hip_image.array[:, :, hip_layer]).T)
        if method == 'murphy':
            tmp = hip_image.array[:, :, fhc_layer].copy().T
            tmp = np.where(tmp == 0, np.nan, tmp)
            ax[0].imshow(tmp, alpha=.5)

            ax[2].imshow(hip_image.array[:, hip_end[1]].T)
            ax[2].plot([hip_start[0], hip_end[0]], [fhc_layer, hip_layer], 'r')
            ax[2].set_aspect(x_ratio)

        ax[0].plot([hip_start[0], hip_end[0]], [hip_start[1], hip_end[1]], 'r')
        ax[0].text(10, 10, f'Proximal angle: {proximal_angle:.2f}°', color='red', fontsize='small')
        ax[0].text(10, 30, f'Proximal orientation: {np.sign(proximal_orientation):.2f}', color='red', fontsize='small')
        ax[0].set_title(f'Angle: {angle:.2f}°')
        ax[1].imshow(knee_mask[:, :, knee_layer].T)
        ax[1].plot([knee_start[0], knee_end[0]], [knee_start[1], knee_end[1]], 'r')
        ax[1].text(10, 10, f'Distal angle: {distal_angle:.2f}°', color='red', fontsize='small')
        ax[1].text(10, 30, f'Distal orientation: {np.sign(distal_orientation):.2f}', color='red', fontsize='small')

        if not return_landmarks:
            return angle, fig

    if return_landmarks:
        landmarks = {'hip_start': hip_start, 'hip_end': hip_end, 'knee_start': knee_start, 'knee_end': knee_end}
        if not plot:
            return angle, landmarks

    return angle, fig, landmarks


def calculate_femoral_torsion_ct(femur_image: Segmentation, knee_image: Segmentation, side: str = 'left', segmentation_label: int = 1, plot: bool | plt.Axes = False) -> float:
    """
    Calculate the femoral torsion from a whole-leg CT segmentation mask.
    :param femur_image: A Segmentation object of the femur segmentation mask.
    :param knee_image: A Segmentation object of the knee segmentation mask.
    :param side: The side of the image (not patient!), either 'left' or 'right'.
    :param segmentation_label: The label of the femur in the segmentation mask.
    :param plot: Whether to plot the reference lines or not.
    :return: The femoral torsion in degrees.
    """
    assert side in ['left', 'right'], 'Side must be either "left" or "right"'

    landmarks = get_proximal_reference_line_ct(femur_image, side=side, segmentation_label=segmentation_label)
    hip_start = landmarks[0]
    fhc_layer = int(hip_start[2])
    hip_end = landmarks[1]
    hip_start[2] = hip_end[2]  # adjust axial coordinate to align
    hip_layer = int(hip_end[2])
    print(fhc_layer, hip_layer)

    proximal_line = hip_end - hip_start
    x = np.array([-1, 0, 0]) if side == 'left' else np.array(
        [1, 0, 0])  # need to distinguish between left and right image side
    proximal_angle = calculate_angle_between_vectors(proximal_line, x)

    if proximal_angle > 90:
        proximal_angle = 180 - proximal_angle

    proximal_orientation = hip_end[1] - hip_start[1]  # positive if hip_end is posterior to hip_start

    knee_mask = np.where(knee_image.array == segmentation_label, 1, 0)
    knee_layer, knee_start, knee_end = get_knee_reference_line(knee_mask,
                                                               bone='femur', segmentation_label=segmentation_label)

    if knee_start[0] < knee_end[0]:  # if this is somehow not the case, swap the points
        tmp = knee_start
        knee_start = knee_end
        knee_end = tmp

    distal_line = knee_end - knee_start
    x = np.array([-1, 0, 0])  # because end is always left of start, no need to distinguish between left and right
    distal_angle = calculate_angle_between_vectors(distal_line, x)

    if distal_angle > 90:
        distal_angle = 180 - distal_angle

    distal_orientation = (knee_end[1] - knee_start[1]) if side == 'left' else (knee_start[1] - knee_end[1])  # positive if knee_end is posterior to knee_start

    if np.sign(proximal_orientation) != np.sign(distal_orientation):  # add angles
        angle = proximal_angle + distal_angle
    else:
        angle = proximal_angle - distal_angle

    if plot is not False:
        ax = plot

        """
        ax[0].imshow(np.where(femur_image.array[:, :, hip_layer] == 0, np.nan, femur_image.array[:, :, hip_layer]).T)
        tmp = femur_image.array[:, :, fhc_layer].copy().T
        tmp = np.where(tmp == 0, np.nan, tmp)
        ax[0].imshow(tmp, alpha=.5)
        """
        ax[0].imshow(femur_image.array[:, :, hip_layer].T, cmap='gray')
        ax[0].imshow(femur_image.array[:, :, fhc_layer].T, cmap='gray', alpha=.5)

        ax[2].imshow(femur_image.array[:, hip_end[1]].T)
        ax[2].plot([hip_start[0], hip_end[0]], [fhc_layer, hip_layer], 'r')
        ax[2].set_aspect(femur_image.spacing[2] / femur_image.spacing[0])

        ax[0].plot([hip_start[0], hip_end[0]], [hip_start[1], hip_end[1]], 'r')
        ax[0].text(10, 10, f'Proximal angle: {proximal_angle:.2f}°', color='red', fontsize='small')
        ax[0].text(10, 30, f'Proximal orientation: {np.sign(proximal_orientation):.2f}', color='red', fontsize='small')
        ax[0].set_title(f'Angle: {angle:.2f}°')
        ax[1].imshow(knee_mask[:, :, knee_layer].T)
        ax[1].plot([knee_start[0], knee_end[0]], [knee_start[1], knee_end[1]], 'r')
        ax[1].text(10, 10, f'Distal angle: {distal_angle:.2f}°', color='red', fontsize='small')
        ax[1].text(10, 30, f'Distal orientation: {np.sign(distal_orientation):.2f}', color='red', fontsize='small')

    return angle
