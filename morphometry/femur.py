import math
import numpy as np
import nibabel as nib
from matplotlib import pyplot as plt
from scipy.ndimage import center_of_mass, label
from scipy.optimize import curve_fit
from skimage.measure import find_contours
from morphometry.hip import get_femoral_head_center, get_femoral_head_center_ct
from morphometry.utils import rotate_mask_dorsal_points, find_notch, transform_point, points_on_circle, \
    get_contour, circle_fit, draw_circle, draw_line, extract_connected_components_2d
from morphometry.image_io import Image, Segmentation, reflect_image_left_right, split_ct_image
from morphometry import constants as C
from morphometry.geometry import mirror_index_x, validate_in_plane_isotropy
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

    Two kinds of broken neck geometry are rejected here outright, because they leave no angle to
    measure: a level at or above the femoral head centre (the existing
    ``C.MURPHY_NECK_BASE_MIN_HEAD_DISTANCE_MM`` gate) and a head-to-base axis whose axial-plane
    projection is shorter than one in-plane voxel, i.e. the two landmarks are not even resolved
    apart in the plane whose direction *is* the measurement. Everything else that merely looks
    wrong -- a short (ill-conditioned) axis, a base level inside the fitted head sphere, a base
    centroid outside the bone, a base slice made of several pieces -- is left to
    :func:`murphy_neck_base_qc`, which flags it without changing the number.
    :param hip_image: An Image object of the hip segmentation mask.
    :param femoral_head_centre: The centre of the femoral head (voxel index).
    :param segmentation_label: The label of the segmentation mask.
    :param isotropic: Whether the image has isotropic voxels.
    :return: The centre of the base of the femoral neck, ``(x, y, layer)`` with ``x`` and ``y`` continuous.
    :raises RuntimeError: If the level is less than ``C.MURPHY_NECK_BASE_MIN_HEAD_DISTANCE_MM`` below the
        head centre (a mis-detected trochanter), holds no femur, or lies less than one in-plane voxel
        from the head centre in the axial plane (no in-plane direction is resolved).
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
    base = np.array([center[0], center[1], float(level)])

    # The torsion is the in-plane *direction* of head centre -> neck base, so a projection shorter
    # than one voxel carries no direction at all: the two landmarks are the same point at the
    # sampling resolution. This is a distance in the index grid, so it needs no isotropy
    # assumption, and it is independent of the gate above, which is on the z drop -- a perfectly
    # vertical neck clears that gate and lands here. On the 1887 measurable Augsburg / NaKo image
    # halves it never fires (shortest axis 6.4 voxels); it is untested on CT (no data available).
    if float(np.hypot(base[0] - femoral_head_centre[0], base[1] - femoral_head_centre[1])) < 1.0:
        raise RuntimeError('The femoral head centre and the neck base are less than one in-plane '
                           'voxel apart in the axial plane; the neck axis has no direction.')
    return base


def _significant_components(section: np.ndarray, ratio: float = 0.1) -> int:
    """Count the connected pieces of a binary slice, ignoring specks.

    Full (8-)connectivity, and only components worth at least ``ratio`` of the largest count --
    the rule ``scripts/process_nmdid.py::qc_flags`` uses on whole bones: a single stray voxel
    diagonally off the cortex is a labelling speck and moves no centroid, whereas a piece worth a
    tenth of the section is a second lump of bone.
    :param section: A binary 2D slice.
    :param ratio: Minimum size of a counted component as a fraction of the largest one.
    :return: The number of components at or above that size (0 for an empty slice).
    """
    labelled, count = label(section, np.ones((3, 3), dtype=bool))
    if count <= 1:
        return int(count)
    sizes = np.bincount(labelled.ravel())[1:]
    return int(np.count_nonzero(sizes >= ratio * sizes.max()))


def murphy_neck_base_qc(hip_image: Image, femoral_head_centre: np.ndarray, neck_base: np.ndarray,
                        head_radius: float | None = None, segmentation_label: int = 1) -> dict:
    """
    Describe how trustworthy one Murphy neck-base landmark is, without changing it.

    Murphy's femoral-torsion axis is short (~30 mm in the axial plane on the Augsburg hip MRI,
    ~29 mm on the NaKo hips), so a neck base a few millimetres out of place is a large angular
    error, and the measurement looks perfectly plausible when it happens.
    :func:`get_femoral_neck_base` rejects only the two geometries that leave no angle to measure
    at all (a level at or above the head centre, and an axis shorter than one in-plane voxel);
    this function reports the degenerate geometries that still produce a number, so a cohort run
    can mark the row instead of silently trusting it. Nothing here is repaired: a flagged neck is
    still measured, and the returned landmark is the one the measurement used.

    The flags, in the ``name:value`` / ``;``-joined format of ``scripts/process_nmdid.py``'s
    ``qc_flags`` column:

    * ``murphy_axis_short:<mm>`` -- the axial-plane head-to-base axis is shorter than
      ``1 / tan(C.MURPHY_AXIS_ANGLE_TOLERANCE_DEG)`` in-plane voxels, i.e. one voxel of landmark
      error would tilt the axis by more than that tolerance (see the constant for the derivation
      and its disclosure). ``<mm>`` is the axis length.
    * ``murphy_base_in_head:<mm>`` -- the neck base lies inside the fitted femoral head sphere,
      which no femoral neck base does: the trochanter was found far too proximal. The test is the
      full 3-D distance from the head centre (in mm, all three spacings) against ``head_radius``,
      not the superior-inferior drop alone: a horizontal neck (coxa valga, a large in-plane offset
      and a small drop) is well outside the sphere and must not be flagged for it. ``<mm>`` is how
      far inside the sphere the base sits. Skipped when no ``head_radius`` is given.
    * ``murphy_base_outside_bone:<mm>`` -- the base centroid does not fall on a femur voxel, so
      the base slice's section is not one compact cross-section the centroid can represent.
      ``<mm>`` is the distance to the nearest femur voxel centre on that slice.
    * ``murphy_base_slice_multipart:<n>`` -- the femur on the base slice is ``n`` disconnected
      pieces (:func:`_significant_components`, so a one-voxel speck does not count), and the
      centroid is then an average over disjoint bone rather than a neck centre. A real neck in two
      pieces (a hole in the segmentation) can still have its centroid near the neck centre, so
      this flags rather than raises.

    :param hip_image: The Image the landmark was found in (the Murphy ``hip_image``; for CT the
        hip crop :func:`get_proximal_reference_line_ct` builds, which shares the femur image's
        geometry). Only its spacing and the base slice of its array are read.
    :param femoral_head_centre: The femoral head centre as a (continuous) voxel index.
    :param neck_base: The neck base as returned by :func:`get_femoral_neck_base`.
    :param head_radius: The fitted femoral head radius in in-plane voxels (the ``r_fh`` of
        :func:`get_proximal_reference_line`), or ``None`` to skip ``murphy_base_in_head``.
    :param segmentation_label: The label of the femur in the segmentation mask.
    :return: ``{'axis_length_mm', 'axis_length_voxels', 'angle_per_voxel_deg', 'head_clearance_mm',
        'base_distance_to_bone_mm', 'base_in_bone', 'base_slice_components', 'flags'}``, where
        ``head_clearance_mm`` is the base's distance outside the head sphere (negative = inside it,
        ``None`` without a ``head_radius``) and ``flags`` is the ``;``-joined flag string (``''``
        when the neck looks sound). Every entry is a scalar invariant under the left-right mirror,
        except that the ``base_in_bone`` lookup breaks an exactly half-integer centroid towards
        ``+x`` on both sides (``floor(x + 0.5)``; the mirror of that rule would be
        ``ceil(x - 0.5)``), so a centroid landing exactly on a voxel boundary *at the edge of the
        mask* could in principle read differently in the mirrored frame. Otherwise the dict may be
        computed in the reflected frame a right image half is measured in.
    :raises ValueError: If the image's two in-plane voxel edges differ -- the axis length and the
        flag threshold are in in-plane voxels, which is only a length if those agree (and on such
        a dataset the torsion itself is refused, see
        :func:`morphometry.geometry.validate_in_plane_isotropy`) -- or if the base slice holds no
        femur at all (``neck_base`` cannot be a centroid of it; :func:`get_femoral_neck_base`
        rejects that case before returning).
    """
    validate_in_plane_isotropy(hip_image.spacing, 'hip image (Murphy neck-base QC)')
    spacing = np.abs(np.asarray(hip_image.spacing, dtype=float))
    head = np.asarray(femoral_head_centre, dtype=float)
    base = np.asarray(neck_base, dtype=float)
    level = int(round(float(base[2])))

    # in-plane voxels: equal x/y spacing is guaranteed above, so this is a length in voxel edges
    axis_voxels = float(np.hypot(base[0] - head[0], base[1] - head[1]))
    metrics = {
        'axis_length_mm': axis_voxels * float(spacing[0]),
        'axis_length_voxels': axis_voxels,
        # how far one in-plane voxel of landmark error turns the axis
        'angle_per_voxel_deg': float(np.degrees(np.arctan2(1.0, axis_voxels))),
        'head_clearance_mm': None,
    }
    if head_radius is not None:
        # the full 3-D distance in mm, so a neck that runs sideways rather than down is not
        # mistaken for one that starts inside the head
        offset_mm = (base - head) * spacing
        metrics['head_clearance_mm'] = float(np.linalg.norm(offset_mm)) - float(head_radius) * float(spacing[0])

    array = np.where(hip_image.array == segmentation_label, 1, 0)
    section = array[:, :, level]
    points = np.argwhere(section)
    if not len(points):
        raise ValueError('The Murphy neck-base slice holds no femur, so the given neck base is '
                         'not its centroid; the QC has nothing to describe.')
    index = (int(np.floor(base[0] + .5)), int(np.floor(base[1] + .5)))
    metrics['base_in_bone'] = bool(0 <= index[0] < section.shape[0] and 0 <= index[1] < section.shape[1]
                                   and section[index])
    metrics['base_distance_to_bone_mm'] = float(np.min(np.hypot(
        (points[:, 0] - base[0]) * spacing[0], (points[:, 1] - base[1]) * spacing[1])))
    metrics['base_slice_components'] = _significant_components(section)

    flags = []
    if axis_voxels < 1.0 / math.tan(math.radians(C.MURPHY_AXIS_ANGLE_TOLERANCE_DEG)):
        flags.append(f"murphy_axis_short:{metrics['axis_length_mm']:.2f}")
    if metrics['head_clearance_mm'] is not None and metrics['head_clearance_mm'] < 0:
        flags.append(f"murphy_base_in_head:{-metrics['head_clearance_mm']:.2f}")
    if not metrics['base_in_bone']:
        flags.append(f"murphy_base_outside_bone:{metrics['base_distance_to_bone_mm']:.2f}")
    if metrics['base_slice_components'] > 1:
        flags.append(f"murphy_base_slice_multipart:{metrics['base_slice_components']}")
    metrics['flags'] = ';'.join(flags)
    return metrics


#: Reason codes for ``murphy_landmark_failed``, matched against the exception message in order.
#: All of the Murphy landmark's own failures are ``RuntimeError``, so the type alone says nothing,
#: and this is the flag family with the highest rate by far: 241 of the 2128 Augsburg / NaKo image
#: halves have no Murphy landmark at all, as ``trochanter_implausible`` (109),
#: ``trochanter_no_components`` (69), ``base_at_head_level`` (36, the
#: ``MURPHY_NECK_BASE_MIN_HEAD_DISTANCE_MM`` gate), ``shaft_not_found`` (21) and 6 missing files.
#: The codes carry no ``,`` or ``;``, so they are safe in the CSV columns.
_LANDMARK_FAILURE_REASONS = (
    ('no two components found', 'trochanter_no_components'),
    ('components too small', 'trochanter_components_too_small'),
    ('no plausible candidates', 'trochanter_implausible'),
    ('plausible trochanter minor', 'trochanter_not_found'),
    ('Could not find the femoral shaft', 'shaft_not_found'),
    ('too close to', 'base_at_head_level'),
    ('one in-plane voxel', 'axis_unresolved'),
    ('lies above the image', 'level_above_image'),
    ('No femur on the neck-base slice', 'no_femur_on_level'),
    ('holds no femur', 'no_femur_on_level'),
    ('in-plane voxel spacing', 'anisotropic_in_plane'),
)


def _landmark_failure_flag(exc: BaseException) -> str:
    """Turn a failed Murphy landmark search into one ``murphy_landmark_failed`` flag.

    :param exc: The exception the landmark search raised.
    :return: ``murphy_landmark_failed:<reason>``, where ``<reason>`` is the matching code from
        :data:`_LANDMARK_FAILURE_REASONS` or, for anything unrecognised, the exception's type name
        (so a new failure mode is still visible, just unnamed).
    """
    message = str(exc)
    for marker, reason in _LANDMARK_FAILURE_REASONS:
        if marker in message:
            return f'murphy_landmark_failed:{reason}'
    return f'murphy_landmark_failed:{type(exc).__name__}'


def get_murphy_neck_qc(hip_image: Image, side: str = 'left', segmentation_label: int = 1,
                       x_ratio: float = 1., isotropic: bool = False) -> str:
    """
    QC flags for the Murphy femoral-neck landmarks of one hip MRI image half.

    The entry point the batch scripts use: it resolves the Murphy reference line exactly as
    :func:`morphometry.measurements.femur.calculate_femoral_torsion` and
    :func:`morphometry.measurements.hip.calculate_anteversion` do (so the flags describe the neck
    those measurements actually used) and hands the landmarks to :func:`murphy_neck_base_qc`. The
    reference line is resolved a second time rather than threaded through the measurement
    functions, which keeps their signatures and return types untouched; the cost is one extra
    femoral-head fit and trochanter search per image half.

    A right image half is measured as the left-right reflection of a left one, and every quantity
    :func:`murphy_neck_base_qc` reports is mirror-invariant, so the flags are the same whichever
    frame they are computed in.
    :param hip_image: An Image object of the proximal-femur segmentation mask (one image half).
    :param side: Side of the image (not patient!), either 'left' or 'right'.
    :param segmentation_label: The label of the femur in the segmentation mask.
    :param x_ratio: Correction factor for slice thickness (as passed to the measurement).
    :param isotropic: Whether the image has isotropic voxels.
    :return: The ``;``-joined flag string, ``''`` when the neck looks sound, or
        ``murphy_landmark_failed:<reason>`` when the landmark search itself failed (the measurement
        is then NaN too; see :func:`_landmark_failure_flag` for the reason codes).
    """
    try:
        start, end, r_fh = get_proximal_reference_line(hip_image, side=side, method='murphy',
                                                       segmentation_label=segmentation_label,
                                                       x_ratio=x_ratio, isotropic=isotropic)
        return murphy_neck_base_qc(hip_image, start, end, head_radius=r_fh,
                                   segmentation_label=segmentation_label)['flags']
    except Exception as exc:
        # a QC helper must never cost its caller a measurement, and two call sites
        # (the Augsburg scripts, compute_torsion.py) run it outside any try block
        return _landmark_failure_flag(exc)


def get_murphy_neck_qc_ct(femur_image: Segmentation, side: str = 'left', segmentation_label: int = 1) -> str:
    """
    QC flags for the Murphy femoral-neck landmarks of one whole-leg CT image half.

    The CT counterpart of :func:`get_murphy_neck_qc`: it repeats the proximal-femur crop and
    landmark search of :func:`get_proximal_reference_line_ct` (through the shared
    :func:`_ct_proximal_landmarks`) so the flags describe the neck
    :func:`morphometry.measurements.femur.calculate_femoral_torsion_ct` measured.
    :param femur_image: A Segmentation object of the femur (whole-leg) mask, one image half.
    :param side: Side of the image (not patient!), either 'left' or 'right'.
    :param segmentation_label: The label of the femur in the segmentation mask.
    :return: The ``;``-joined flag string, ``''`` when the neck looks sound, or
        ``murphy_landmark_failed:<reason>`` (see :func:`_landmark_failure_flag`). Never raises.
    """
    try:
        image = reflect_image_left_right(femur_image) if side == 'right' else femur_image
        hip_image, fhc, r_fh, base = _ct_proximal_landmarks(image, segmentation_label)
        return murphy_neck_base_qc(hip_image, fhc, base, head_radius=r_fh,
                                   segmentation_label=segmentation_label)['flags']
    except Exception as exc:
        return _landmark_failure_flag(exc)


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

def _ct_proximal_landmarks(femur_image: Segmentation, segmentation_label: int = 1) -> Tuple[Segmentation, np.ndarray, float, np.ndarray]:
    """
    Resolve the proximal-femur crop and the Murphy landmarks of a left CT image half.

    Shared by :func:`get_proximal_reference_line_ct` and :func:`get_murphy_neck_qc_ct` so the QC
    flags describe exactly the landmark the measurement used. Written for a left image half, like
    everything the femoral torsion builds on; callers reflect a right half first.
    :param femur_image: A Segmentation object of the femur (whole-leg) mask, one image half.
    :param segmentation_label: The label of the femur in the segmentation mask.
    :return: ``(hip_image, femoral_head_centre, head_radius, neck_base)`` -- the proximal 30% of
        the femur as its own Segmentation (same geometry as ``femur_image``), the head centre as a
        continuous voxel index, the head radius in in-plane voxels, and the Murphy neck base.
    """
    r_fh, fhc = get_femoral_head_center_ct(femur_image, segmentation_label, 'left')
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
    return hip_image, fhc, float(r_fh), get_femoral_neck_base(hip_image, fhc, segmentation_label, False)


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

    _, fhc, _, end = _ct_proximal_landmarks(femur_image, segmentation_label)

    start: np.ndarray = np.array([fhc[0], fhc[1], fhc[2]])
    return start, end
