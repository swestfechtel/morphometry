import math
import numpy as np
from monai.apps.pathology.transforms.post.array import find_contours

from morphometry.hip import get_femoral_head_center
from morphometry.utils import rotate_mask_dorsal_points, find_notch, transform_point, get_contour_points, \
    rotate_mask_vec_parallel, get_layer_with_biggest_convex_area, points_on_circle, get_dorsal_mask_point, \
    rotate_point, num_mask_points_on_line, shrink_points_to_mask, angle_between, get_contour
from morphometry.bresenham import bresenhamline
from skimage import measure
from scipy.ndimage import center_of_mass
from scipy.optimize import curve_fit
from typing import Tuple, Optional
from matplotlib import pyplot as plt

"""
All functions assume axis ordering is x = axial, y = coronal, z = sagittal.
"""

def contour_femoral_neck(mask: np.ndarray, contour: np.ndarray, layer_selected: int, center: np.ndarray, r: float) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Get the contour of the femoral neck.
    :param mask: 3d segmentation mask
    :param contour: contour of segmentation mask
    :param layer_selected: index of layer with femoral neck
    :param center: center coordinates of femur head
    :param r: radius of femur head
    :return: adjusted mask, contour of the femoral neck and radii of two circles around the femoral head center with
    0.9 and 1.1 times the radius of the femoral head
    """
    # rotate mask and find notch to regulate the radii of the spheres
    mask_new = mask[layer_selected].copy()
    rotated_mask, angle1 = rotate_mask_dorsal_points(mask_new,
                                                     np.array(center_of_mass(mask_new)).astype(np.int16))
    rotated_mask, angle2 = rotate_mask_dorsal_points(rotated_mask,
                                                     np.array(center_of_mass(rotated_mask)).astype(np.int16))
    angle = angle1 + angle2
    notch_rot = find_notch(rotated_mask,
                           percentage=0.8,
                           thresh=5,
                           break_after_first=True)
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
    a = np.linalg.norm(notch - np.array([center[1], center[2]])) / r
    r_1 = (-0.1 + a) * r
    r_2 = (0.1 + a) * r

    mask_new = mask[layer_selected].copy()

    # get contour only between the two circles on the selected layer
    for z in range(mask.shape[2] - 1):  # all z coordinates
        if r_2**2 - (z - center[2])**2 >= 0:
            y_b = int(round(math.sqrt(r_2**2 - (z - center[2])**2)))
            if center[1] + y_b < mask.shape[1]:
                contour[layer_selected, center[1] + y_b:, z] = 0
                mask_new[center[1] + y_b:, z] = 0
            if center[1] - y_b > 0:
                contour[layer_selected, :center[1] - y_b, z] = 0
                mask_new[:center[1] - y_b, z] = 0
        else:
            contour[layer_selected, :, z] = 0
            mask_new[:, z] = 0

        if r_1**2 - (z - center[2])**2 >= 0:
            y_c = int(round(math.sqrt(r_1**2 - (z - center[2])**2)))
            if center[1] - y_c > 0 and center[1] + y_c < mask.shape[1]:
                contour[layer_selected, center[1] - y_c:center[1] + y_c, z] = 0
                mask_new[center[1] - y_c:center[1] + y_c, z] = 0

    return mask_new, contour, r_1, r_2


def get_proximal_reference_line(segmentation_mask: np.ndarray, side: str = 'left', segmentation_label: int = 1, x_ratio: float = 1.) -> Tuple[int, np.ndarray, np.ndarray]:
    """
    Get the proximal reference line of a segmentation mask for calculating the femoral torsion.
    :param segmentation_mask: A segmentation mask of the proximal femur.
    :param side: Side of the image (not patient!), either 'left' or 'right'.
    :param segmentation_label: The label of the segmentation mask.
    :param x_ratio: Correction factor for slice thickness.
    :return: The layer and start and end points of the reference line.
    """
    assert side in ['left', 'right'], 'Side must be either "left" or "right"'

    r, center, layer_high, layer_low = get_femoral_head_center(segmentation_mask, side=side, segmentation_label=segmentation_label, return_layers=True, x_ratio=x_ratio)
    center = center.astype(np.int16)

    if side == 'right':  # flip z axis to make this 'left-sided'
        segmentation_mask = segmentation_mask[:, :, ::-1]
        center[2] = center[2] + (segmentation_mask.shape[2] // 2 - center[2]) * 2  # flip z coordinate

    layer_selected = None
    for n in range(0, segmentation_mask.shape[0]):
        contour_length = len(find_contours(segmentation_mask[n], 0.8))  # need to use find_contours here because it can detect disconnected contours
        on_circle = points_on_circle(segmentation_mask[n], r * 2, center[1:])
        if contour_length == 1 and len(on_circle) > 0:
            layer_selected = n
            break

    if layer_selected is None:
        raise ValueError('Could not find a suitable layer for the femoral neck.')

    contour = get_contour(segmentation_mask)
    femoral_neck, contour, r_1, r_2 = contour_femoral_neck(segmentation_mask, contour,
                                                       layer_selected, center,
                                                       r)

    # I think this ensures that the femoral neck is not too thin? I'm gonna leave this out for now
    """
    while np.count_nonzero(femoral_neck == 1) < 65:
        print(np.count_nonzero(femoral_neck == 1))
        layer_selected = layer_selected + 1
        femoral_neck, contour, r_1, r_2 = contour_femoral_neck(
            segmentation_mask, contour, layer_selected, center, r)
    """

    # center of femoral neck
    center_fn = np.array(center_of_mass(femoral_neck)).astype(np.int16)

    # get coordinates of the selected contour points
    contour_pts_l = np.nonzero(contour[layer_selected])

    distance_center = (contour_pts_l[0] -
                       center_fn[0]) ** 2 + (contour_pts_l[1] - center_fn[1]) ** 2
    r_3 = math.sqrt(np.median(distance_center)) * 1.5

    # get contour also only with small distance to center_fn on the selected layer
    for z in range(segmentation_mask.shape[2] - 1):  # all z coordinates
        if r_3 ** 2 - (z - center_fn[1]) ** 2 >= 0:
            y_b = int(round(math.sqrt(r_3 ** 2 - (z - center_fn[1]) ** 2)))
            if center_fn[0] + y_b < segmentation_mask.shape[1]:
                contour[layer_selected, center_fn[0] + y_b:, z] = 0
            if center_fn[0] - y_b > 0:
                contour[layer_selected, :center_fn[0] - y_b, z] = 0
        else:
            contour[layer_selected, :, z] = 0

    contour_pts_l = np.nonzero(contour[layer_selected])

    # get index of the contour point after the largest gap
    # -> first contour point on the other side of the segmentation mask
    diff = np.ediff1d(contour_pts_l[0])
    ind_gap = np.argsort(diff)[-1] + 1

    def g(x, m):
        return m * x

    if diff.max() == 1:
        # lsf through the center and the contour points to get its slope
        popt, _ = curve_fit(g, contour_pts_l[1] - center[2],
                            contour_pts_l[0] - center[1])
        m_new = popt[0]
    else:
        # separated lsf through the center and the contour points on both sides
        # to get their slopes and calculate the mean
        popt1, _ = curve_fit(g, contour_pts_l[1][:ind_gap] - center[2],
                             contour_pts_l[0][:ind_gap] - center[1])
        popt2, _ = curve_fit(g, contour_pts_l[1][ind_gap:] - center[2],
                             contour_pts_l[0][ind_gap:] - center[1])
        m_new = np.mean([popt1[0], popt2[0]])

    # find endpoint for the reference line
    end = np.array([layer_selected, int((-80) * m_new + center[1]), int(-80 + center[2])])
    if side == 'right':
        end[2] = end[2] + (segmentation_mask.shape[2] // 2 - end[2]) * 2  # flip z coordinate
        center[2] = center[2] - (center[2] - segmentation_mask.shape[2] // 2) * 2  # flip z coordinate

    return layer_selected, np.array([layer_selected, center[1], center[2]]), end


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


def get_distal_reference_line(mask: np.ndarray, thresh: int = 2, step_size: int = 1, mark_points: bool = False) -> Tuple[int, np.ndarray, np.ndarray]:
    """
    Get the proximal reference line of a segmentation mask for calculating the femoral torsion.

    :param mask: A 3D segmentation mask of the femur.
    :param thresh: Minimum number of mask points on the reference line.
    :param step_size: Step size (in degrees) for rotating the reference line.
    :param mark_points: Whether to mark the points in the mask or not.
    :return: The layer and start and end points of the reference line.
    """

    layer_index = get_layer_with_biggest_convex_area(mask)
    mask_layer = mask[layer_index]  # mask_layer is 2D mask of the selected layer

    notch = find_notch(mask_layer, percentage=0.7, thresh=1)
    if notch[0] is None:
        rotated_mask, ang1 = rotate_tibia(mask_layer)
        rot_thresh = find_notch(rotated_mask, percentage=0.7, thresh=2)
        if rot_thresh[0] is None:
            rot_thresh = np.array(center_of_mass(rotated_mask)).astype(np.int16)
        rotated_mask, ang2 = rotate_mask_dorsal_points(
            rotated_mask, rot_thresh)
        angle = ang1 + ang2
        notch_rot = find_notch(rotated_mask, percentage=0.7, thresh=2)
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


def calculate_femoral_torsion(hip_mask: np.ndarray, knee_mask: np.ndarray, side: str = 'left', segmentation_label: int = 1, x_ratio: float = 1., plot: bool = False) -> float:
    """
    Calculate the femoral torsion from a segmentation mask.
    :param hip_mask: A segmentation mask of the proximal femur.
    :param knee_mask: A segmentation mask of the distal femur.
    :param side: Side of the image (not patient!), either 'left' or 'right'.
    :param segmentation_label: The label of the segmentation mask.
    :param x_ratio: Correction factor for slice thickness.
    :param plot: Whether to plot the reference lines or not.
    :return: The femoral torsion in degrees.
    """
    assert side in ['left', 'right'], 'Side must be either "left" or "right"'

    hip_layer, hip_start, hip_end = get_proximal_reference_line(hip_mask, side=side, segmentation_label=segmentation_label, x_ratio=x_ratio)
    proximal_line = (hip_end - hip_start) if side == 'left' else (hip_start - hip_end)
    knee_layer, knee_start, knee_end = get_distal_reference_line(knee_mask)
    distal_line = knee_end - knee_start

    if plot:
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(hip_mask[hip_layer])
        ax[0].plot([hip_start[2], hip_end[2]], [hip_start[1], hip_end[1]], 'r')
        ax[1].imshow(knee_mask[knee_layer])
        ax[1].plot([knee_start[2], knee_end[2]], [knee_start[1], knee_end[1]], 'r')
        plt.show()

    # calculate angle between the two reference lines
    angle = angle_between(proximal_line, distal_line)
    return np.degrees(angle)
