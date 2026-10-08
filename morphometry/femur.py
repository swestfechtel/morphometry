import math
import numpy as np
import nibabel as nib
from matplotlib import pyplot as plt
from scipy.ndimage import center_of_mass
from scipy.optimize import curve_fit
from skimage.measure import find_contours
from morphometry.hip import get_femoral_head_center, get_femoral_head_center_ct
from morphometry.utils import rotate_mask_dorsal_points, find_notch, transform_point, points_on_circle, \
    get_contour, circle_fit, draw_circle, draw_line, extract_connected_components_2d
from morphometry.image_io import Image, Segmentation, reflect_image_left_right, split_ct_image
from morphometry import constants as C
from morphometry.geometry import mirror_index_x
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
    :return: ``(x, y, layer)``: the axial slice through the trochanter minor's centroid, and on it the
        centroid of the bone medial of the femoral shaft's medial edge (voxel indices).
    """
    point, _ = _find_trochanter_minor(hip_image, femoral_head_centre, segmentation_label, isotropic)
    return point


def _find_trochanter_minor(hip_image: Image, femoral_head_centre: np.ndarray, segmentation_label: int = 1,
                           isotropic: bool = False) -> Tuple[np.ndarray, int]:
    """
    Find the trochanter minor and the axial slice of its superior extent.

    The trochanter minor is the smaller component of the first sagittal slice, scanning from medial to
    lateral, that splits the femur into two components one of which lies 40-75 mm from the femoral
    head centre in the medial-lateral / superior-inferior plane.
    :param hip_image: An Image object of the hip segmentation mask.
    :param femoral_head_centre: The centre of the femoral head (voxel index).
    :param segmentation_label: The label of the segmentation mask.
    :param isotropic: Whether the image has isotropic voxels.
    :return: The point returned by :func:`get_trochanter_minor`, and the most superior axial slice
        (smallest index; LPI) holding that sagittal component.
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

    superior_layer = int(np.min(np.nonzero(smaller_component)[1]))  # sagittal slices are (y, z)
    return np.array([axial_layer_com[0], axial_layer_com[1], axial_layer]), superior_layer

def get_femoral_neck_base(hip_image: Image, femoral_head_centre: np.ndarray, segmentation_label: int = 1, isotropic: bool = False) -> np.ndarray:
    """
    Get the centre of the base of the femoral neck as described by Murphy et al.

    Murphy et al. (1987) take the centre of the femoral neck base "at the level just above the lesser
    trochanter": the level is the first axial slice superior of the trochanter minor's superior extent
    (see :func:`_find_trochanter_minor`), i.e. as close above it as the acquisition resolves -- 10 mm on
    the Augsburg hip MRI, 3 mm on whole-leg CT, 1 mm on the isotropic NaKo hips. The centre is the femur
    centroid on that slice, as a continuous index.

    The slice through the trochanter itself, used formerly, includes the trochanter and the proximal
    shaft and read 1-2 degrees more antetorsion on the Augsburg hips. A larger fixed offset was tried
    and rejected: against the NaKo raters' signed manual Murphy values (1 mm data, 120 hips) a 10 mm
    level reads +5.6 degrees high with ICC2 0.85, against 0-5 mm levels (bias -1.2 to +2.2, ICC2 up to
    0.94) -- the raters place the point just above the trochanter, as described. The Augsburg hips cannot
    tell offsets up to 10 mm apart, since they all select the same slice there.
    :param hip_image: An Image object of the hip segmentation mask.
    :param femoral_head_centre: The centre of the femoral head (voxel index).
    :param segmentation_label: The label of the segmentation mask.
    :param isotropic: Whether the image has isotropic voxels.
    :return: The centre of the base of the femoral neck, ``(x, y, layer)`` with ``x`` and ``y`` continuous.
    :raises RuntimeError: If the level is less than ``C.MURPHY_NECK_BASE_MIN_HEAD_DISTANCE_MM`` below the
        head centre (a mis-detected trochanter), or holds no femur.
    """
    _, superior_layer = _find_trochanter_minor(hip_image=hip_image, femoral_head_centre=femoral_head_centre,
                                               segmentation_label=segmentation_label, isotropic=isotropic)
    slice_mm = abs(float(hip_image.spacing[2]))
    level = superior_layer - 1
    if (level - femoral_head_centre[2]) * slice_mm < C.MURPHY_NECK_BASE_MIN_HEAD_DISTANCE_MM:
        # the neck base lies well distal to the head centre; a level within the head is a mis-detected
        # trochanter minor, and the resulting axis would be a few millimetres long
        raise RuntimeError('Trochanter minor found too close to (or above) the femoral head centre; '
                           'no neck base below the head.')
    if level < 0:
        raise RuntimeError('The neck-base level lies above the image.')
    array = np.where(hip_image.array == segmentation_label, 1, 0)
    if not array[:, :, level].any():
        raise RuntimeError('No femur on the neck-base slice above the trochanter minor.')

    center = center_of_mass(array[:, :, level])
    return np.array([center[0], center[1], float(level)])

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

    The landmark search is written for a left image half. A right image half holds the mirror
    image of that anatomy, so it is measured by reflecting it left-right
    (:func:`morphometry.image_io.reflect_image_left_right`), running the left-side search on the
    reflection, and mapping the landmarks back with ``x -> n - 1 - x``. This makes the right side
    mirror-equivariant by construction: a mirrored left half measured with ``side='right'`` yields
    exactly the mirrored left-side landmarks. (Patching a separate right-side branch had left it
    systematically different from the left side: a flip off by one voxel, a head-trim that was not
    the mirror of the left one, and rounding by truncation, which is not mirror-consistent.)
    :param hip_image: An Image object of the hip segmentation mask.
    :param side: Side of the image (not patient!), either 'left' or 'right'.
    :param method: The method to use for determining the reference line.
    :param segmentation_label: The label of the segmentation mask.
    :param x_ratio: Correction factor for slice thickness.
    :param isotropic: Whether the image has isotropic voxels.
    :return: The start and end points of the reference line (voxel indices of ``hip_image``), the radius of the
    femoral head centre and the radius of the trochanter major if method is tomczak.
    """
    assert side in ['left', 'right'], 'Side must be either "left" or "right"'
    assert method in ['lee', 'murphy',
                      'tomczak'], 'Currently, only "lee", "murphy" and "tomczak" are supported as methods.'

    if side == 'right':
        width = hip_image.array.shape[0]
        start, end, *radii = get_proximal_reference_line(reflect_image_left_right(hip_image), side='left',
                                                         method=method, segmentation_label=segmentation_label,
                                                         x_ratio=x_ratio, isotropic=isotropic)
        return (mirror_index_x(start, width), mirror_index_x(end, width), *radii)

    r_fh, center, layer_high, layer_low = get_femoral_head_center(hip_image.array, side=side,
                                                                  segmentation_label=segmentation_label,
                                                                  return_layers=True, x_ratio=x_ratio,
                                                                  isotropic=isotropic, spacing=hip_image.spacing)

    if method == 'lee':
        center = center.astype(np.int16)  # the Lee neck search indexes the mask with the centre
        array = hip_image.array.copy()
        array = np.where(array == segmentation_label, 1, 0)
        end = get_femoral_neck_center_lee(array, center, r_fh)
    elif method == 'murphy':
        # continuous head centre and neck-base centroid: the axis is short (~25 mm in-plane on the
        # Augsburg hips), so truncating either end to an index would rotate it by up to ~2 degrees
        end = get_femoral_neck_base(hip_image=hip_image, femoral_head_centre=center, segmentation_label=segmentation_label, isotropic=isotropic)
    elif method == 'tomczak':
        center = center.astype(np.int16)
        end, r_tm = get_trochanter_major_center(hip_image, center, segmentation_label)

    start: np.ndarray = np.array([center[0], center[1], center[2]])
    return (start, end, r_fh, r_tm) if method == 'tomczak' else (start, end, r_fh)

def get_proximal_reference_line_ct(femur_image: Segmentation, side: str = 'left', segmentation_label: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get the proximal reference line of a CT femur segmentation for calculating the femoral torsion.

    As in :func:`get_proximal_reference_line`, a right image half is measured as the left-right
    reflection of a left one (left-side search on
    :func:`morphometry.image_io.reflect_image_left_right`, landmarks mapped back with
    ``x -> n - 1 - x``), so the two sides are mirror-equivariant by construction. The reflection,
    not an index-relabelling mirror, is what is needed here: the femoral head centre is found in
    physical space, where the lateral direction is a physical direction.
    :param femur_image: A Segmentation object of the femur segmentation mask.
    :param side: The side of the image (not patient!), either 'left' or 'right'.
    :param segmentation_label: The label of the femur in the segmentation mask.
    :return: The start and end points of the reference line (voxel indices of ``femur_image``; the start is a
    continuous index).
    """
    assert side in ['left', 'right'], 'Side must be either "left" or "right"'

    if side == 'right':
        width = femur_image.array.shape[0]
        start, end = get_proximal_reference_line_ct(reflect_image_left_right(femur_image), side='left',
                                                    segmentation_label=segmentation_label)
        return mirror_index_x(start, width), mirror_index_x(end, width)

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

    hip_image = Segmentation.from_nibabel(nib.Nifti1Image(mask, femur_image.affine, femur_image.header))
    end = get_femoral_neck_base(hip_image, fhc, segmentation_label, False)

    start: np.ndarray = np.array([fhc[0], fhc[1], fhc[2]])
    return start, end
