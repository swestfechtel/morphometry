from typing import Tuple

import numpy as np
from scipy.ndimage import center_of_mass

from morphometry.utils import get_layer_with_biggest_convex_area, find_notch, get_contour_points, \
    rotate_mask_vec_parallel, rotate_mask_dorsal_points, transform_point, rotate_point, num_mask_points_on_line, \
    get_dorsal_mask_point, shrink_points_to_mask


def rotate_tibia(segmentation_mask: np.ndarray):
    """
    Rotate mask so that a line between the contour points with the largest
    distance would be parallel to the x axis.
    :param segmentation_mask: A 2D segmentation mask.
    """
    contour = get_contour_points(segmentation_mask)
    d = 0
    for i in range(len(contour)):
        for j in range(len(contour)):
            y1 = contour[:, 0][i]
            z1 = contour[:, 1][i]
            y2 = contour[:, 0][j]
            z2 = contour[:, 1][j]

            # vector always starts in point with smaller x value
            if z1 < z2:
                vec = np.array([y1, z1]) - np.array([y2, z2])
            else:
                vec = np.array([y2, z2]) - np.array([y1, z1])
            if np.linalg.norm(vec) > d:
                d = np.linalg.norm(vec)
                line = vec

    return rotate_mask_vec_parallel(segmentation_mask, line, np.array([0, 1]))


def get_knee_reference_line(mask: np.ndarray, bone: str, thresh: int = 2, step_size: int = 1) -> Tuple[
    int, np.ndarray, np.ndarray]:
    """
    Get the distal reference line of a segmentation mask for calculating the femoral torsion, or the proximal
    reference line for calculating tibial torsion.

    :param mask: A 3D segmentation mask of the femur.
    :param bone: Either 'femur' or 'tibia'.
    :param thresh: Minimum number of mask points on the reference line.
    :param step_size: Step size (in degrees) for rotating the reference line.
    :return: The layer and start and end points of the reference line.
    """
    assert bone in ['femur', 'tibia'], 'Bone must be either "femur" or "tibia"'

    layer_index = get_layer_with_biggest_convex_area(mask)
    mask_layer = mask[layer_index]  # mask_layer is 2D mask of the selected layer

    if bone == 'femur':
        percentage = 0.7
        notch_threshold = 1
    else:
        percentage = 0.5
        notch_threshold = 2

    notch = find_notch(mask_layer, percentage=percentage, thresh=notch_threshold)
    if notch[0] is None:
        rotated_mask, ang1 = rotate_tibia(mask_layer)
        rot_thresh = find_notch(rotated_mask, percentage=percentage, thresh=2)
        if rot_thresh[0] is None:
            rot_thresh = np.array(center_of_mass(rotated_mask)).astype(np.int16)
        rotated_mask, ang2 = rotate_mask_dorsal_points(
            rotated_mask, rot_thresh)
        angle = ang1 + ang2
        notch_rot = find_notch(rotated_mask, percentage=percentage, thresh=2)
    else:
        rotated_mask, angle = rotate_mask_dorsal_points(mask_layer, notch)

    # pre-calculate necessary values to rotate points back in original frame
    rot_offset = np.array([
        _rot_dim - _orig_dim
        for _rot_dim, _orig_dim in zip(rotated_mask.shape, mask[layer_index].shape)
    ])
    rot_center = np.array([int(
        (rotated_mask.shape[0] - 1) / 2), int((rotated_mask.shape[1] - 1) / 2)])

    # notch found
    if notch[0] is not None:
        # transform notch in coordinate frame of rotated image
        notch_rot = transform_point(notch, np.array([int(
            (mask_layer.shape[0] - 1) / 2), int((mask_layer.shape[1] - 1) / 2)]),
                                    angle,
                                    offset=rot_offset / 2)
    # rotated notch found
    else:
        # transform rotated notch back in original coordinate frame
        notch = transform_point(notch_rot,
                                rot_center,
                                -angle,
                                offset=-rot_offset / 2)

    start_pt = get_dorsal_mask_point(rotated_mask)

    # choose endpoint on the other side of the notch at the end of the mask
    if start_pt[1] < notch_rot[1]:
        x_end = rotated_mask.shape[1] - 1
        rot_dir = 1  # rotate counterclockwise
    else:
        x_end = 0
        rot_dir = -1  # rotate clockwise
    end_pt = (start_pt[0], x_end)

    # rotate line between start and endpoint
    # until the number of mask points on the line is higher than the threshold
    while num_mask_points_on_line(rotated_mask, start_pt, end_pt,
                                  notch_rot) < thresh:
        end_pt = rotate_point(start_pt, end_pt, step_size * rot_dir)

    _, final_end_pt = shrink_points_to_mask(rotated_mask, start_pt, end_pt)

    # transform final_end_pt and start_pt back
    end_pt_orig = transform_point(final_end_pt,
                                  rot_center,
                                  -angle,
                                  offset=-rot_offset / 2)
    start_pt_orig = transform_point(start_pt,
                                    rot_center,
                                    -angle,
                                    offset=-rot_offset / 2)

    start_pt_orig = start_pt_orig.astype(np.uint16)
    end_pt_orig = end_pt_orig.astype(np.uint16)
    notch = notch.astype(np.uint16)

    # transform points from layer mask to 3D mask
    start_pt_orig = np.array([layer_index, start_pt_orig[0], start_pt_orig[1]])
    end_pt_orig = np.array([layer_index, end_pt_orig[0], end_pt_orig[1]])
    notch = np.array([layer_index, notch[0], notch[1]])

    return layer_index, start_pt_orig, end_pt_orig
