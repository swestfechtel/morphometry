"""Hip measurement functions.

Public measurements (CCD, alpha angle, anteversion, acetabular version/depth,
center-edge angle, femoral-head/acetabulum distances, cartilage thickness,
femoral offset). Landmark/reference helpers they rely on stay in
``morphometry.hip`` and are imported here.

"left"/"right" refers to the image side, not the patient side.
"""
import numpy as np
import pyvista as pv
import pandas as pd
import nibabel as nib

from morphometry.utils import sphere_fit, get_contour_points, calculate_angle_between_vectors, \
    calculate_min_distance_between_point_clouds, get_vector_through_point_perpendicular_to_line, \
    get_minimum_distance_between_line_and_point, num_connected_components, \
    extract_connected_components_2d, circumference_points, intersect_ndarrays, \
    sort_points_clockwise, sort_points_by_x, fit_circle_to_points
from morphometry.image_io import Image, Segmentation
from morphometry.bresenham import bresenhamline
from morphometry import constants as C
from morphometry import geometry as G
from morphometry.hip import (get_femoral_head_center, get_femoral_head_center_ct,
    get_femoral_neck_center, get_femoral_neck_center_ct, get_femoral_shaft_axis,
    get_femoral_shaft_axis_ct, get_femoral_neck_transition, get_p1, get_p2,
    get_cartilage_inner_and_outer_surface_points)
from morphometry.femur import get_proximal_reference_line
from morphometry.measurements.femur import _proximal_angle

from scipy.ndimage import center_of_mass, label, rotate
from scipy.spatial import KDTree
from scipy.stats import zscore
from typing import Tuple, Optional
from skimage.measure import find_contours
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt


def calculate_ccd(hip_image: Image, knee_image: Image = None, side: str = 'left', segmentation_label: int = 1, isotropic: bool = False, x_ratio: float = 1., debug: bool = False, plot: plt.Axes | pv.Plotter | bool = False) -> Tuple[float, float] | Tuple[float, float, plt.Figure]:
    """
    Calculate the CCD angle from the femoral head center, femoral neck axis and femoral shaft axis.
    :param hip_image: A segmentation mask of the proximal femur.
    :param knee_image: A segmentation mask of the knee.
    :param side: Side of the image (not patient!), either 'left' or 'right'.
    :param segmentation_label: The label of the femur in the segmentation mask.
    :param isotropic: Whether the image has isotropic voxels.
    :param x_ratio: Correction factor for slice thickness.
    :param debug: Whether to display debug messages.
    :param plot: Whether to plot the results.
    :return: The actual and projected CCD angle.
    """
    assert side in ['left', 'right'], 'Side must be either "left" or "right"'

    hip_mask = hip_image.array
    r, femoral_head_center = get_femoral_head_center(hip_mask, side=side, segmentation_label=segmentation_label, x_ratio=x_ratio, isotropic=isotropic)
    femoral_neck_points, femoral_neck_center = get_femoral_neck_center(hip_mask, (r, femoral_head_center), side=side, segmentation_label=segmentation_label, x_ratio=x_ratio)

    knee_mask = knee_image.array if knee_image else None

    shaft_axis_low, shaft_axis_high = get_femoral_shaft_axis(hip_mask, knee_mask, femur_label=segmentation_label,
                                                             isotropic=isotropic)

    if not isotropic:
        # femoral_neck_center[2] -= 1  # correction because algorithm seems to over-estimate the femoral neck center by one voxel in z direction
        pass

    if knee_image is not None:
        femoral_head_center_orig = femoral_head_center.copy()
        femoral_neck_center_orig = femoral_neck_center.copy()
        shaft_axis_low_orig, shaft_axis_high_orig = shaft_axis_low.copy(), shaft_axis_high.copy()

        femoral_head_center = hip_image.transform_index_to_physical_point(femoral_head_center)  # transform to physical coordinates
        femoral_neck_center = hip_image.transform_index_to_physical_point(femoral_neck_center)
        shaft_axis_low = knee_image.transform_index_to_physical_point(shaft_axis_low)
        shaft_axis_high = hip_image.transform_index_to_physical_point(shaft_axis_high)

    if debug:
        print(f'Femoral head center: {femoral_head_center}, femoral neck center: {femoral_neck_center}, femoral shaft axis: {(shaft_axis_low, shaft_axis_high)}')

    # Calculate the angle between the femoral neck axis and the femoral shaft axis
    neck_vector = femoral_neck_center - femoral_head_center
    shaft_vector = shaft_axis_high - shaft_axis_low

    ccd = calculate_angle_between_vectors(neck_vector.astype('float32'), shaft_vector.astype('float32'))
    ccd = 180 - ccd if ccd < 90 else ccd

    # also calculate the projected ccd
    # https://math.stackexchange.com/questions/2305792/3d-projection-on-a-2d-plane-weak-maths-ressources
    """
    projection_plane_index = femoral_head_center[1]
    origin = np.array([hip_mask.shape[0] // 2, 0, hip_mask.shape[2] // 2])
    projection_plane_center = np.array([origin[0], projection_plane_index, origin[2]])
    d = np.linalg.norm(projection_plane_center - origin)
    femoral_neck_center_projected = np.array([femoral_neck_center[0], femoral_neck_center[2]]) * (d / femoral_neck_center[1])
    femoral_head_center_projected = np.array([femoral_head_center[0], femoral_head_center[2]]) * (d / femoral_head_center[1])
    femoral_shaft_axis_1_projected = np.array([shaft_axis_high[0], shaft_axis_high[2]]) * (d / shaft_axis_high[1])
    femoral_shaft_axis_0_projected = np.array([shaft_axis_low[0], shaft_axis_low[2]]) * (d / shaft_axis_low[1])
    neck_vector_projected = femoral_neck_center_projected - femoral_head_center_projected
    shaft_vector_projected = femoral_shaft_axis_1_projected - femoral_shaft_axis_0_projected
    projected_ccd = calculate_angle_between_vectors(neck_vector_projected.astype('float32'), shaft_vector_projected.astype('float32'))
    """
    femoral_neck_center_projected = np.array([femoral_neck_center[0], 0, femoral_neck_center[2]])
    femoral_head_center_projected = np.array([femoral_head_center[0], 0, femoral_head_center[2]])
    shaft_axis_high_projected = np.array([shaft_axis_high[0], 0, shaft_axis_high[2]])
    shaft_axis_low_projected = np.array([shaft_axis_low[0], 0, shaft_axis_low[2]])

    neck_vector_projected = femoral_neck_center_projected - femoral_head_center_projected
    shaft_vector_projected = shaft_axis_high_projected - shaft_axis_low_projected
    ccd_projected = calculate_angle_between_vectors(neck_vector_projected.astype('float32'), shaft_vector_projected.astype('float32'))

    ccd_projected = 180 - ccd_projected if ccd_projected < 90 else ccd_projected

    if plot:
        if knee_image is not None:
            """
            # plot.imshow(hip_mask[:, int(femoral_head_center_orig[1]), :].T)
            comb = np.concatenate((hip_mask, knee_mask), axis=2)
            plt.imshow(comb[:, int(femoral_head_center_orig[1]), :].T)
            plot.plot([femoral_head_center_orig[0], femoral_neck_center_orig[0]], [femoral_head_center_orig[2], femoral_neck_center_orig[2]],
                    'r-')
            plot.plot([shaft_axis_high_orig[0], shaft_axis_low_orig[0]], [shaft_axis_high_orig[2] + hip_mask.shape[2], shaft_axis_low_orig[2] + hip_mask.shape[2]], 'g-')
            """

            proximal_femur_coords = np.argwhere(hip_mask == segmentation_label).astype(object)
            distal_femur_coords = np.argwhere(knee_mask == segmentation_label).astype(object)

            # print(proximal_femur_coords[0], hip_image.transform_index_to_physical_point(proximal_femur_coords[0]))

            # proximal_femur_coords = np.vectorize(hip_image.transform_index_to_physical_point)(proximal_femur_coords)
            # distal_femur_coords = np.vectorize(knee_image.transform_index_to_physical_point)(distal_femur_coords)
            for i, coord in enumerate(proximal_femur_coords):
                proximal_femur_coords[i] = hip_image.transform_index_to_physical_point(coord)
                if side == 'right':
                    proximal_femur_coords[i, 0] -= hip_mask.shape[0]  # adjust for right side

            for i, coord in enumerate(distal_femur_coords):
                distal_femur_coords[i] = knee_image.transform_index_to_physical_point(coord)
                if side == 'right':
                    distal_femur_coords[i, 0] -= knee_mask.shape[0]

            plot.add_mesh(pv.PolyData(proximal_femur_coords.astype(np.float32)), color=('g' if side == 'left' else 'y'), opacity=.5)
            plot.add_mesh(pv.PolyData(distal_femur_coords.astype(np.float32)), color='b', opacity=.5)

        else:
            """
            plot.imshow(hip_mask[:, int(femoral_head_center[1]), :].T)
            plot.plot([shaft_axis_high[0], shaft_axis_low[0]], [shaft_axis_high[2], shaft_axis_low[2]], 'g-')
            plot.plot([femoral_head_center[0], femoral_neck_center[0]], [femoral_head_center[2], femoral_neck_center[2]],
                    'r-')

            plot.set_aspect(x_ratio)
            """
            femur_coords = np.argwhere(hip_mask == segmentation_label).astype(np.float32)
            for i, coord in enumerate(femur_coords):
                femur_coords[i] = hip_image.transform_index_to_physical_point(coord)
                if side == 'right':
                    femur_coords[i, 0] -= hip_mask.shape[0]  # adjust for right side


            plot.add_mesh(pv.PolyData(femur_coords), color=('g' if side == 'left' else 'y'), opacity=.1)

        femoral_head_center = hip_image.transform_index_to_physical_point(
            femoral_head_center)  # transform to physical coordinates
        femoral_neck_center = hip_image.transform_index_to_physical_point(femoral_neck_center)
        shaft_axis_low = hip_image.transform_index_to_physical_point(shaft_axis_low)
        shaft_axis_high = hip_image.transform_index_to_physical_point(shaft_axis_high)

        if side == 'right':
            femoral_head_center[0] -= hip_mask.shape[0]  # adjust for right side
            femoral_neck_center[0] -= hip_mask.shape[0]
            shaft_axis_high[0] -= hip_mask.shape[0]
            shaft_axis_low[0] -= hip_mask.shape[0]

        plot.add_lines(np.array([femoral_head_center, femoral_neck_center - 2 * neck_vector]), color='r')
        plot.add_lines(np.array([shaft_axis_high, shaft_axis_low]), color='r')

    return ccd, ccd_projected

def calculate_ccd_ct(femur_image: Segmentation, side: str = 'left', segmentation_label: int = 1, plot: bool | pv.Plotter = False) -> Tuple[float, float]:
    """
    Calculate the CCD angle from a whole-leg CT segmentation mask using PCA for the femoral shaft axis.
    :param femur_image: A segmentation mask of the whole leg.
    :param side: The side of the image (not patient!), either 'left' or 'right'.
    :param segmentation_label: The label of the femur in the segmentation mask.
    :param plot: Whether to plot the results.
    :return: The actual and projected CCD angle.
    """
    assert side in ['left', 'right'], 'Side must be either "left" or "right"'

    r, femoral_head_center = get_femoral_head_center_ct(femur_image, segmentation_label, side)
    femoral_neck_points, femoral_neck_center = get_femoral_neck_center_ct(femur_image, (r, femoral_head_center), segmentation_label, side)
    shaft_axis_low, shaft_axis_high = get_femoral_shaft_axis_ct(femur_image, segmentation_label=segmentation_label)

    neck_vector = femoral_neck_center - femoral_head_center
    shaft_vector = shaft_axis_high - shaft_axis_low

    ccd = calculate_angle_between_vectors(neck_vector.astype('float32'), shaft_vector.astype('float32'))
    ccd = 180 - ccd if ccd < 90 else ccd

    femoral_neck_center_projected = np.array([femoral_neck_center[0], 0, femoral_neck_center[2]])
    femoral_head_center_projected = np.array([femoral_head_center[0], 0, femoral_head_center[2]])
    shaft_axis_high_projected = np.array([shaft_axis_high[0], 0, shaft_axis_high[2]])
    shaft_axis_low_projected = np.array([shaft_axis_low[0], 0, shaft_axis_low[2]])

    neck_vector_projected = femoral_neck_center_projected - femoral_head_center_projected
    shaft_vector_projected = shaft_axis_high_projected - shaft_axis_low_projected
    ccd_projected = calculate_angle_between_vectors(neck_vector_projected.astype('float32'),
                                                    shaft_vector_projected.astype('float32'))

    ccd_projected = 180 - ccd_projected if ccd_projected < 90 else ccd_projected

    if plot:
        point_cloud = np.argwhere(femur_image.array == segmentation_label).astype(np.float32)
        point_cloud = np.array([femur_image.transform_index_to_physical_point(x) for x in point_cloud])

        plot.add_mesh(pv.PolyData(point_cloud.astype(np.float32)), color=('g' if side == 'left' else 'y'), opacity=.1)
        plot.add_mesh(pv.PolyData(femoral_neck_points.astype(np.float32)), color='r')
        plot.add_mesh(pv.Sphere(r, femoral_head_center), color='c', opacity=.3)
        plot.add_lines(np.array([femoral_head_center, femoral_neck_center + 2 * neck_vector]), color='r')
        plot.add_lines(np.array([shaft_axis_high, shaft_axis_low]), color='r')
        plot.enable_eye_dome_lighting()

    return ccd, ccd_projected

def calculate_anteversion(segmentation_mask: Image, side: str = 'left', segmentation_label: int = 1, isotropic: bool = False, plot: Tuple[plt.Axes, plt.Axes] | bool = False) -> float:
    """
    Calculate the femoral neck anteversion (signed proximal neck angle).

    This is the proximal half of the femoral torsion: the acute angle between the
    femoral neck axis (Murphy reference line) and the medial-lateral axis, signed
    negative when the neck end is anterior to the femoral head centre. It shares the
    proximal-angle computation with :func:`morphometry.measurements.femur.calculate_femoral_torsion`.
    :param segmentation_mask: An Image of the proximal-femur segmentation mask.
    :param side: Side of the image (not patient!), either 'left' or 'right'.
    :param segmentation_label: The label of the femur in the segmentation mask.
    :param isotropic: Whether the image has isotropic voxels.
    :param plot: A pair of matplotlib Axes to draw the reference line on, or False.
    :return: The anteversion angle in degrees.
    """
    # TODO revise: the anteversion convention here is provisional.
    G.validate_side(side)

    hip_start, hip_end = get_proximal_reference_line(
        segmentation_mask, side=side, method='murphy', segmentation_label=segmentation_label,
        isotropic=isotropic, x_ratio=1)[:2]
    fhc_layer = hip_start[2]
    hip_start[2] = hip_end[2]  # align axial coordinate

    angle = _proximal_angle(hip_start, hip_end, side)
    if hip_end[1] - hip_start[1] < 0:  # neck end anterior to head centre -> negative
        angle = -angle

    if plot is not False:
        plot[0].imshow(segmentation_mask.array[:, :, hip_start[2]].T, cmap='gray')
        tmp = segmentation_mask.array[:, :, fhc_layer].copy().T
        tmp = np.where(tmp == 0, np.nan, tmp)
        plot[0].imshow(tmp, alpha=.5)
        plot[0].plot([hip_start[0], hip_end[0]], [hip_start[1], hip_end[1]], 'r-')
        plot[0].set_aspect('equal')

        plot[1].imshow(segmentation_mask.array[:, hip_end[1]].T)
        plot[1].plot([hip_start[0], hip_end[0]], [fhc_layer, hip_end[2]], 'r')
        # plot[1].set_aspect(x_ratio)

    return angle

def calculate_alpha_angle(segmentation_mask: np.ndarray, side: str = 'left', segmentation_label: int = 1, isotropic: bool = False, x_ratio: float = 1, plot: plt.Axes | bool = False) -> float:
    """
    Calculate the alpha angle from the femoral head center and the femoral neck transition.
    :param segmentation_mask: A segmentation mask of the proximal femur.
    :param side: Side of the image (not patient!), either 'left' or 'right'.
    :param segmentation_label: The label of the femur in the segmentation mask.
    :param isotropic: Whether the image has isotropic voxels.
    :param x_ratio: Correction factor for slice thickness.
    :param plot: Whether to plot the reference lines.
    :return: The alpha angle.
    """
    assert side in ['left', 'right'], 'Side must be either "left" or "right"'

    mask = segmentation_mask.copy()
    if side == 'right':  # need to flip because rotation behaves weirdly with right side images
        mask = mask[::-1]

    r, femoral_head_center = get_femoral_head_center(mask, side='left', segmentation_label=segmentation_label, isotropic=isotropic, x_ratio=x_ratio)
    _, femoral_neck_center = get_femoral_neck_center(mask, (r, femoral_head_center), side='left', segmentation_label=segmentation_label, x_ratio=x_ratio)

    neck_axis = femoral_neck_center - femoral_head_center
    sagittal_axis = np.array([-1, 0, 0])
    # coronal_axis = np.array([0, -1, 0])

    rotation_angle = calculate_angle_between_vectors(neck_axis, sagittal_axis)  # to reconstruct the neck axis plane, the mask must be rotated around the coronal axis such that the neck axis aligns with the sagittal axis

    # print(side, rotation_angle)

    rotated_image = rotate(mask, axes=(0, 2), angle=rotation_angle, reshape=True, order=0)

    tmp = np.zeros_like(mask)
    femoral_head_center_rounded = np.round(femoral_head_center).astype(np.int16)

    for i in [femoral_head_center_rounded[0] - 1, femoral_head_center_rounded[0], femoral_head_center_rounded[0] + 1]:
        for j in [femoral_head_center_rounded[1] - 1, femoral_head_center_rounded[1], femoral_head_center_rounded[1] + 1]:
            for k in [femoral_head_center_rounded[2] - 1, femoral_head_center_rounded[2], femoral_head_center_rounded[2] + 1]:
                    tmp[i, j, k] = 1

    tmp = rotate(tmp, axes=(0, 2), angle=rotation_angle, reshape=True, order=0)
    coords = np.argwhere(tmp == 1)
    femoral_head_center_rotated = np.array([np.round(np.mean(coords[:, 0])), np.round(np.mean(coords[:, 1])), np.round(np.mean(coords[:, 2]))], dtype=np.int16)

    tmp = np.zeros_like(mask)
    femoral_neck_center_rounded = np.round(femoral_neck_center).astype(np.int16)

    for i in [femoral_neck_center_rounded[0] - 1, femoral_neck_center_rounded[0], femoral_neck_center_rounded[0] + 1]:
        for j in [femoral_neck_center_rounded[1] - 1, femoral_neck_center_rounded[1], femoral_neck_center_rounded[1] + 1]:
            for k in [femoral_neck_center_rounded[2] - 1, femoral_neck_center_rounded[2], femoral_neck_center_rounded[2] + 1]:
                tmp[i, j, k] = 1

    tmp = rotate(tmp, axes=(0, 2), angle=rotation_angle, reshape=True, order=0)
    coords = np.argwhere(tmp == 1)
    femoral_neck_center_rotated = np.array([np.round(np.mean(coords[:, 0])), np.round(np.mean(coords[:, 1])), np.round(np.mean(coords[:, 2]))], dtype=np.int16)

    femur = np.where(rotated_image == 1, 1, 0)
    contour_points = get_contour_points(femur[:, :, femoral_head_center_rotated[2]])

    anterior_contour = contour_points[contour_points[:, 1] < femoral_head_center_rotated[1]]
    posterior_contour = contour_points[contour_points[:, 1] > femoral_head_center_rotated[1]]

    anterior_points_sorted = sort_points_clockwise(anterior_contour, femoral_head_center_rotated[:2])
    posterior_points_sorted = sort_points_clockwise(posterior_contour, femoral_head_center_rotated[:2], counter_clockwise=True)

    anterior_points_sorted = sort_points_by_x(anterior_points_sorted, descending=True)
    posterior_points_sorted = sort_points_by_x(posterior_points_sorted, descending=True)

    tol = 2
    tmp = contour_points[contour_points[:, 0] > (femoral_head_center_rotated[0] - .6 * r)]

    center, radius = fit_circle_to_points(tmp)

    anterior_point = np.array([0, 0])
    posterior_point = np.array([0, 0])

    for p in anterior_points_sorted:
        d = np.linalg.norm(p - center)
        if (1.2 * radius > d > radius + tol) and (p[0] < center[0]):  # restrict the search space to everything lateral of the center
                                                        # should always be < since the image is flipped for right side
            anterior_point = p
            break

    for p in posterior_points_sorted:
        d = np.linalg.norm(p - center)
        if (1.2 * radius > d > radius + tol) and (p[0] < center[0]):  # restrict the search space to everything lateral of the center
                                                        # should always be < since the image is flipped for right side
            posterior_point = p
            break

    neck_axis_rotated = femoral_neck_center_rotated[:2] - center
    anterior_axis = anterior_point - center
    posterior_axis = posterior_point - center

    anterior_angle = calculate_angle_between_vectors(neck_axis_rotated.astype('float32'), anterior_axis.astype('float32'))
    posterior_angle = calculate_angle_between_vectors(neck_axis_rotated.astype('float32'), posterior_axis.astype('float32'))

    if plot:
        if side == 'right':
            rotated_image = rotated_image[::-1]  # flip back for visualisation
            femoral_head_center_rotated = np.array([rotated_image.shape[0], 0, 0]) - femoral_head_center_rotated * np.array([1, -1, -1])  # adjust for flipping
            femoral_neck_center_rotated = np.array([rotated_image.shape[0], 0, 0]) - femoral_neck_center_rotated * np.array([1, -1, -1])  # adjust for flipping
            anterior_point = np.array([rotated_image.shape[0], 0]) - anterior_point * np.array([1, -1])  # adjust for flipping
            posterior_point = np.array([rotated_image.shape[0], 0]) - posterior_point * np.array([1, -1])  # adjust for flipping
            center = np.array([rotated_image.shape[0], 0]) - center * np.array([1, -1])  # adjust for flipping

        plot.imshow(rotated_image[:, :, femoral_head_center_rotated[2]].T, cmap='gray')
        plot.scatter(tmp[:, 0], tmp[:, 1], s=1, c='y')
        # plot.plot([femoral_head_center_rotated[0], femoral_neck_center_rotated[0]], [femoral_head_center_rotated[1], femoral_neck_center_rotated[1]], 'r-')
        # plot.plot([anterior_point[0], femoral_head_center_rotated[0]], [anterior_point[1], femoral_head_center_rotated[1]], 'g-')
        # plot.plot([posterior_point[0], femoral_head_center_rotated[0]], [posterior_point[1], femoral_head_center_rotated[1]], 'b-')
        plot.plot([center[0], femoral_neck_center_rotated[0]], [center[1], femoral_neck_center_rotated[1]], 'r-')
        plot.plot([center[0], anterior_point[0]], [center[1], anterior_point[1]], 'g-')
        plot.plot([center[0], posterior_point[0]], [center[1], posterior_point[1]], 'b-')
        plot.add_patch(plt.Circle(femoral_head_center_rotated[:2], r, color='b', fill=False))
        plot.add_patch(plt.Circle(center, radius, color='r', fill=False))
        plot.set_aspect('equal')

    return anterior_angle, posterior_angle

def calculate_acetabular_anteversion(hip_image: Segmentation, femur_label: int = 1, acetabulum_label: int = 3, isotropic: bool = False, ct: bool = False, plot: bool | plt.Axes = False) -> Tuple[float, float]:
    """
    Calculate the acetabular anteversion for both sides from a segmentation mask.
    :param hip_image: A segmentation mask of the hip.
    :param femur_label: The label of the femur in the segmentation mask.
    :param acetabulum_label: The label of the acetabulum in the segmentation mask.
    :param isotropic: Whether the image has isotropic voxels.
    :param ct: Whether the image is a CT scan.
    :param plot: Whether to plot the results.
    :return: The acetabular anteversion for both sides.
    """
    left_mask = hip_image.array[:hip_image.shape[0] // 2]
    right_mask = hip_image.array[hip_image.shape[0] // 2:]

    left_image = nib.Nifti1Image(left_mask, hip_image.affine, hip_image.header)
    right_image = nib.Nifti1Image(right_mask, hip_image.affine, hip_image.header)
    left_image = Segmentation.from_nibabel(left_image)
    right_image = Segmentation.from_nibabel(right_image)

    if ct:
        _, left_fhc = get_femoral_head_center_ct(left_image, segmentation_label=femur_label, side='left')
        _, right_fhc = get_femoral_head_center_ct(right_image, segmentation_label=femur_label, side='right')

        left_fhc = left_image.transform_physical_point_to_index(left_fhc)
        right_fhc = right_image.transform_physical_point_to_index(right_fhc)
    else:
        _, left_fhc = get_femoral_head_center(left_image.array, side='left', segmentation_label=femur_label,
                                              isotropic=isotropic)
        _, right_fhc = get_femoral_head_center(right_image.array, side='right', segmentation_label=femur_label,
                                               isotropic=isotropic)

    slice_gap = abs(int(left_fhc[2]) - int(right_fhc[2]))
    correct_slice = min(int(left_fhc[2]), int(right_fhc[2])) + slice_gap // 2

    p1_left = get_p1(left_mask[:, :, correct_slice], side='left', segmentation_label=acetabulum_label, fhc_y_index=int(left_fhc[1]))
    p2_left = get_p2(left_mask[:, :, correct_slice], side='left', segmentation_label=acetabulum_label, fhc_y_index=int(left_fhc[1]))

    p1_right = get_p1(right_mask[:, :, correct_slice], side='right', segmentation_label=acetabulum_label, fhc_y_index=int(right_fhc[1]))
    p2_right = get_p2(right_mask[:, :, correct_slice], side='right', segmentation_label=acetabulum_label, fhc_y_index=int(right_fhc[1]))

    femur_array = np.where(hip_image.array == femur_label, 1, 0)
    left_femur = femur_array[:femur_array.shape[0] // 2]

    right_fhc_adj = right_fhc.copy()
    right_fhc_adj[0] += left_femur.shape[0]  # adjust the x coordinate of the right femoral head center to account for the splitting into left and right

    u = left_fhc[:2]  # the point of origin of the line
    v = right_fhc_adj[:2] - left_fhc[:2]  # the direction of the line
    p = p1_left  # the point the perpendicular vector goes through
    s_left, projection_left = get_vector_through_point_perpendicular_to_line(u, v, p)  # s is the vector that goes through p1 and is perpendicular to G (i.e. u + lambda * G)

    u = right_fhc_adj[:2]
    v = left_fhc[:2] - right_fhc_adj[:2]  # the direction of the line
    p = p1_right
    s_right, projection_right = get_vector_through_point_perpendicular_to_line(u, v, p)

    v1 = (p1_left - p2_left).astype('float32')
    v2 = s_left.copy()
    left_aa = calculate_angle_between_vectors(v1, v2)

    v1 = (p1_right - p2_right).astype('float32')
    v2 = s_right.copy()
    right_aa = calculate_angle_between_vectors(v1, v2)

    if plot is not False:
        plot.imshow(hip_image.array[:, :, correct_slice].T, cmap='gray')
        plot.plot([p1_left[0], p2_left[0]], [p1_left[1], p2_left[1]], 'r-', label='Right Acetabulum Rim')
        plot.plot([p1_right[0] + left_femur.shape[0], p2_right[0] + left_femur.shape[0]], [p1_right[1], p2_right[1]],
                'b-', label='Left Acetabulum Rim')
        plot.plot([left_fhc[0], right_fhc_adj[0]], [left_fhc[1], right_fhc_adj[1]], 'g-', label='Femoral Head Centers')

        plot.plot([p1_left[0], projection_left[0]], [p1_left[1], projection_left[1]], 'c--',
                label='Right Perpendicular Projection')
        plot.plot([p1_right[0] + left_femur.shape[0], projection_right[0] + left_femur.shape[0]],
                [p1_right[1], projection_right[1]], 'm--', label='Left Perpendicular Projection')
        plot.set_title(f'Right AA: {left_aa:.2f}°, Left AA: {right_aa:.2f}°')
        plot.set_aspect('equal')
        plot.legend()

    return left_aa, right_aa

def calculate_acetabular_depth(segmentation_mask: np.ndarray, side: str = 'left', femur_label: int = 1, acetabulum_label: int = 3, isotropic: bool = False) -> float:
    """
    Get the minimum distance between the line connecting the anterior and posterior acetabulum rim and the femoral head center.
    :param segmentation_mask: A segmentation mask of the hip.
    :param side: Side of the image (not patient!), either 'left' or 'right'.
    :param femur_label: The label of the femur in the segmentation mask.
    :param acetabulum_label: The label of the acetabulum in the segmentation mask.
    :param isotropic: Whether the image has isotropic voxels.
    :return: The acetabular depth.
    """
    assert side in ['left', 'right'], 'Side must be either "left" or "right"'

    _, fhc = get_femoral_head_center(segmentation_mask, side=side, segmentation_label=femur_label, isotropic=isotropic)

    correct_slice = int(fhc[2])

    p1 = get_p1(segmentation_mask[:, :, correct_slice], side=side, segmentation_label=acetabulum_label, fhc_y_index=int(fhc[1]))
    p2 = get_p2(segmentation_mask[:, :, correct_slice], side=side, segmentation_label=acetabulum_label, fhc_y_index=int(fhc[1]))

    ad = get_minimum_distance_between_line_and_point(p1, p2, fhc[:2])

    return ad

def _lateral_acetabular_edge_point(acetabulum_mask: np.ndarray, fhc: np.ndarray, radius: float,
                                   side: str, *, project: bool, upper_margin_factor: float) -> np.ndarray:
    """Find the most lateral acetabular edge point above the femoral head, one image side.

    Slices restrict the acetabulum mask to the band starting ``upper_margin_factor *
    radius`` above the femoral head centre, then take the most lateral sagittal column
    and, within it, the most inferior (largest transversal) voxel.
    :param acetabulum_mask: A binary 3D mask of the acetabulum (single image side).
    :param fhc: The femoral head centre (index coordinates) for this side.
    :param radius: The femoral head radius (index units) for this side.
    :param side: Image side, 'left' or 'right'.
    :param project: If True, zero the anterior-posterior coordinate of the result.
    :param upper_margin_factor: Multiple of ``radius`` above the FHC where the search band starts.
    :return: The lateral acetabular edge point as a 3D index coordinate.
    """
    G.validate_side(side)
    tmp = acetabulum_mask.copy()
    upper_limit = int(fhc[2] - upper_margin_factor * radius)
    tmp[:, :, :upper_limit] = 0
    points = np.nonzero(tmp)

    lateral_s = np.min(points[0]) if side == 'left' else np.max(points[0])
    candidates = points[0] == lateral_s
    candidate_coronal = points[1][candidates]
    candidate_transversal = points[2][candidates]
    idx = np.argmax(candidate_transversal)
    edge = np.array([lateral_s, candidate_coronal[idx], candidate_transversal[idx]])
    if project:
        edge[1] = 0
    return edge


def calculate_center_edge_angle(hip_image: Segmentation, femur_label: int = 1, acetabulum_label: int = 3, isotropic: bool = False, ct: bool = False, project: bool = False, plot: bool | plt.Axes = False, image_path: str = None) -> Tuple[float, float]:
    """
    Calculate the center edge angle for both sides from a segmentation mask.
    :param hip_image: A segmentation mask of the hip.
    :param femur_label: The label of the femur in the segmentation mask.
    :param acetabulum_label: The label of the acetabulum in the segmentation mask.
    :param isotropic: Whether the image has isotropic voxels.
    :param ct: Whether the image is a CT scan.
    :param project: Whether to project all landmarks onto a 2D plane before calculating the angle.
    :param plot: Whether to plot the results.
    :param image_path: File path to the original image for plotting.
    :return: The center edge angle for both sides.
    """
    left_mask = hip_image.array[:hip_image.shape[0] // 2]
    right_mask = hip_image.array[hip_image.shape[0] // 2:]

    left_image = nib.Nifti1Image(left_mask, hip_image.affine, hip_image.header)
    right_image = nib.Nifti1Image(right_mask, hip_image.affine, hip_image.header)
    left_image = Segmentation.from_nibabel(left_image)
    right_image = Segmentation.from_nibabel(right_image)

    if ct:
        r_l, left_fhc = get_femoral_head_center_ct(left_image, segmentation_label=femur_label, side='left')
        r_r, right_fhc = get_femoral_head_center_ct(right_image, segmentation_label=femur_label, side='right')

        left_fhc = left_image.transform_physical_point_to_index(left_fhc)
        right_fhc = right_image.transform_physical_point_to_index(right_fhc)

        tmp = left_fhc + np.array([r_l, 0, 0])  # rationale: point on the surface of the femoral head in arbitrary direction
        tmp = hip_image.transform_physical_point_to_index(tmp)  # transform that point to index space
        r_l = np.linalg.norm(left_fhc - tmp)  # recalculate radius in index space

        tmp = right_fhc + np.array([r_r, 0, 0])
        tmp = hip_image.transform_physical_point_to_index(tmp)
        r_r = np.linalg.norm(right_fhc - tmp)
    else:
        r_l, left_fhc = get_femoral_head_center(left_image.array, side='left', segmentation_label=femur_label,
                                              isotropic=isotropic)
        r_r, right_fhc = get_femoral_head_center(right_image.array, side='right', segmentation_label=femur_label,
                                               isotropic=isotropic)

    coronal_slice = int((right_fhc[1] + left_fhc[1]) // 2)

    if project:
        left_fhc[1] = 0
        right_fhc[1] = 0

    right_fhc_adj = right_fhc.copy()
    right_fhc_adj[0] += left_mask.shape[0]  # account for the left/right split of the sagittal axis
    G_vec = left_fhc - right_fhc_adj  # vector connecting the left and right femoral head centers

    acetabulum_array = np.where(hip_image.array == acetabulum_label, 1, 0)
    left_acetabulum = acetabulum_array[:acetabulum_array.shape[0] // 2]
    right_acetabulum = acetabulum_array[acetabulum_array.shape[0] // 2:]

    d = np.array([0, 0, -1])
    n = G_vec / np.linalg.norm(G_vec)
    d_perp = d - np.dot(d, n) * n  # d_perp is perpendicular to G and goes in proximal direction
    d_perp *= 100  # scale the perpendicular vector to a reasonable length
    s_right = right_fhc_adj + d_perp  # s is a point from the femoral head center with direction d_perp, just for visualisation
    lat_right = _lateral_acetabular_edge_point(right_acetabulum, right_fhc, r_r, 'right',
                                               project=project, upper_margin_factor=C.CEA_UPPER_MARGIN_FACTOR_RIGHT)
    s2_right = lat_right - right_fhc
    cea_right = calculate_angle_between_vectors(d_perp, s2_right)

    s_left = left_fhc + d_perp
    lat_left = _lateral_acetabular_edge_point(left_acetabulum, left_fhc, r_l, 'left',
                                              project=project, upper_margin_factor=C.CEA_UPPER_MARGIN_FACTOR_LEFT)
    s2_left = lat_left - left_fhc
    cea_left = calculate_angle_between_vectors(d_perp, s2_left)

    if plot is not False:
        if image_path:
            image = Image('nibabel')
            image.read_image(image_path)
            image.transform_coordinate_system()
            plot.imshow(image.array[:, coronal_slice].T, cmap='gray')
            plot.imshow(np.where(hip_image.array > 0, hip_image.array, np.nan)[:, coronal_slice].T, alpha=.5)
        else:
            plot.imshow(hip_image.array[:, coronal_slice].T, cmap='gray')

        plot.plot([left_fhc[0], right_fhc_adj[0]], [left_fhc[2], right_fhc_adj[2]], 'r-', label='G')

        plot.plot([left_fhc[0], s_left[0]], [left_fhc[2], s_left[2]], 'g--', label='Perpendicular Vector (right)')
        plot.plot([right_fhc_adj[0], s_right[0]], [right_fhc_adj[2], s_right[2]], 'b--', label='Perpendicular Vector (left)')

        plot.plot([left_fhc[0], lat_left[0]], [left_fhc[2], lat_left[2]], 'c-', label='Lateral Edge Point (right)')
        plot.plot([right_fhc_adj[0], lat_right[0] + left_mask.shape[0]], [right_fhc_adj[2], lat_right[2]], 'y-', label='Lateral Edge Point (left)')
        plot.set_title(f'Right CEA: {cea_left:.2f}°, Left CEA: {cea_right:.2f}°')
        plot.set_aspect('equal')
        plot.legend()

    return cea_left, cea_right

def calculate_min_distance_between_femoral_head_and_acetabulum(segmentation_mask: np.ndarray, side: str = 'left', femur_label: int = 1, acetabulum_label: int = 3, isotropic: bool = False) -> float:
    """
    Get the minimum distance between the femoral head and the acetabulum.
    :param segmentation_mask: A segmentation mask for the hip.
    :param side: Side of the image (not patient!), either 'left' or 'right'.
    :param femur_label: The label of the femur in the segmentation mask.
    :param acetabulum_label: The label of the acetabulum in the segmentation mask.
    :param isotropic: Whether the image has isotropic voxels.
    :return: The minimum distance between the femoral head and the acetabulum.
    """
    assert side in ['left', 'right'], 'Side must be either "left" or "right"'

    r, c = get_femoral_head_center(segmentation_mask, side=side, segmentation_label=femur_label, isotropic=isotropic)
    femur_mask = np.where(segmentation_mask == femur_label, 1, 0)
    point_cloud = np.argwhere(femur_mask)
    solid_sphere = pv.SolidSphere(inner_radius=0, outer_radius=1.05 * r, center=c)

    # Get the points that are distal to the femoral head center
    points_i_want = np.array(solid_sphere.points)
    points_i_want = points_i_want[points_i_want[:, 2] > c[2]]

    # Build a KDTree for the point cloud and the points we want
    pc_tree = KDTree(point_cloud)
    sphere_tree = KDTree(points_i_want)

    pairs = pc_tree.query_ball_tree(sphere_tree, 2)
    femoral_head_points = list()
    for pair in pairs:
        if len(pair) > 0:
            for index in pair:
                femoral_head_points.append(sphere_tree.data[index])

    femoral_head_points = np.array(femoral_head_points)
    acetabulum_points = np.argwhere(np.where(segmentation_mask == acetabulum_label, 1, 0))

    return calculate_min_distance_between_point_clouds(femoral_head_points, acetabulum_points)

def _sample_cone_directions(axis: np.ndarray, n_rays: int, cone_angle: float) -> np.ndarray:
    """
    Sample unit direction vectors evenly within a cone around a central axis.

    A spherical-Fibonacci cap is used so that the directions are distributed with
    even solid-angle density (a naive polar/azimuth grid would cluster samples
    near the axis). The cap is generated around the canonical +z axis (uniform in
    cos(theta) over the cap, golden-angle azimuth) and then rotated onto ``axis``
    via a Rodrigues rotation, with the (anti)parallel cases handled explicitly.
    :param axis: The 3D direction the cone is centered on (need not be unit length).
    :param n_rays: The number of directions to sample.
    :param cone_angle: The half-angle of the cone in degrees.
    :return: An (n_rays, 3) array of unit direction vectors in the same space as ``axis``.
    """
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)

    cos_min = np.cos(np.radians(cone_angle))
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))

    i = np.arange(n_rays)
    z = 1.0 - (i + 0.5) / n_rays * (1.0 - cos_min)  # uniform in cos(theta) over the cap
    r = np.sqrt(np.clip(1.0 - z * z, 0.0, None))
    phi = i * golden_angle
    local = np.stack([r * np.cos(phi), r * np.sin(phi), z], axis=1)  # directions around +z

    # rotate the +z axis onto `axis` (Rodrigues); handle the (anti)parallel cases
    z_axis = np.array([0.0, 0.0, 1.0])
    v = np.cross(z_axis, axis)
    s = np.linalg.norm(v)
    c = float(np.dot(z_axis, axis))
    if s < 1e-8:
        rot = np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])  # 180 deg about x for antiparallel
    else:
        vx = np.array([[0.0, -v[2], v[1]],
                       [v[2], 0.0, -v[0]],
                       [-v[1], v[0], 0.0]])
        rot = np.eye(3) + vx + vx @ vx * ((1.0 - c) / (s * s))

    return local @ rot.T

def _subchondral_distance_ray_tracing(image: Image, fhc_index: np.ndarray, fhc_radius_index: float,
                                      femur_mask: np.ndarray, acetabulum_mask: np.ndarray,
                                      n_rays: int = 200, cone_angle: float = 45.0,
                                      ray_length_factor: float = 3.0,
                                      plot: bool | plt.Axes | pv.Plotter = False) -> Tuple[float, float, float, float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Measure the subchondral femoral-head-to-acetabulum distance by ray tracing.

    A fan of rays is cast from the femoral head center (FHC) within an angular cone
    pointed at the acetabulum. For each ray, the voxel where it exits the femoral
    head (the first femur 1->0 transition going outward, i.e. the femoral head
    contour) and the first acetabulum voxel hit beyond that are recorded; the
    per-ray value is the Euclidean distance between those two points in physical
    (mm) space. All cone/direction geometry is done in physical space so the result
    is correct under anisotropic voxel spacing; only ray endpoints are converted to
    index space for the Bresenham walk.
    :param image: The Image/Segmentation providing index<->physical transforms.
    :param fhc_index: The femoral head center in index (voxel) coordinates.
    :param fhc_radius_index: The femoral head radius in index (voxel) units.
    :param femur_mask: A binary (0/1) mask of the femur, same grid as ``image``.
    :param acetabulum_mask: A binary (0/1) mask of the acetabulum (or hip bone for CT).
    :param n_rays: The number of rays to cast within the cone.
    :param cone_angle: The half-angle of the cone in degrees.
    :param ray_length_factor: Ray length as a multiple of the femoral head radius (mm).
    :param plot: A matplotlib Axes / PyVista Plotter to overlay the rays on, or False.
    :return: A tuple ``(mean, std, min, max, distances, exit_points, hit_points)`` where
        the distances are in mm and the exit/hit point arrays are (K, 3) physical coordinates
        for the K surviving rays.
    """
    fhc_index = np.asarray(fhc_index, dtype=float)
    shape = np.array(femur_mask.shape)

    # physical-space FHC and femoral head radius (mm)
    fhc_phys = np.asarray(image.transform_index_to_physical_point(fhc_index), dtype=float)
    radius_surface_phys = np.asarray(image.transform_index_to_physical_point(fhc_index + np.array([fhc_radius_index, 0, 0])), dtype=float)
    fhc_radius_mm = np.linalg.norm(radius_surface_phys - fhc_phys)
    ray_length_mm = ray_length_factor * fhc_radius_mm

    # cone axis: FHC -> acetabulum centroid, in physical space
    acetabulum_voxels = np.argwhere(acetabulum_mask)
    if acetabulum_voxels.size == 0:
        raise ValueError('Acetabulum mask is empty; cannot measure subchondral distance')
    acetabulum_centroid_phys = np.asarray(image.transform_index_to_physical_point(acetabulum_voxels.mean(axis=0)), dtype=float)
    axis = acetabulum_centroid_phys - fhc_phys
    if np.linalg.norm(axis) < 1e-6:
        raise ValueError('Femoral head center coincides with acetabulum centroid; cannot define a cone axis')

    directions = _sample_cone_directions(axis, n_rays, cone_angle)
    fhc_voxel = np.rint(fhc_index).astype(int)

    distances, exit_points, hit_points = [], [], []

    for d in directions:
        end_index = np.rint(image.transform_physical_point_to_index(fhc_phys + d * ray_length_mm)).astype(int)
        # bresenhamline returns (start, end] (start excluded); prepend the FHC so index 0 == FHC
        path = bresenhamline(fhc_voxel[np.newaxis, :], end_index[np.newaxis, :], max_iter=-1)
        path = np.vstack([fhc_voxel, path])

        # keep the in-bounds prefix (stop at the first voxel that leaves the volume)
        in_bounds = np.all((path >= 0) & (path < shape), axis=1)
        if not in_bounds[0]:
            continue  # FHC itself out of bounds
        if not in_bounds.all():
            path = path[:int(np.argmin(in_bounds))]

        fv = femur_mask[path[:, 0], path[:, 1], path[:, 2]]
        av = acetabulum_mask[path[:, 0], path[:, 1], path[:, 2]]

        # first voxel that is inside the femoral head (handles FHC landing just outside the mask)
        femur_hits = np.argwhere(fv == 1)
        if femur_hits.size == 0:
            continue  # ray never passes through the femur
        first_femur = int(femur_hits[0, 0])

        # femur exit = last femur voxel before the first outward 1->0 transition
        exit_i = None
        for j in range(first_femur, len(fv) - 1):
            if fv[j] == 1 and fv[j + 1] == 0:
                exit_i = j
                break
        if exit_i is None:
            continue  # ray never exits the femoral head within range

        # first acetabulum hit beyond the exit
        acetabulum_after = np.argwhere(av[exit_i + 1:] == 1)
        if acetabulum_after.size == 0:
            continue  # ray never reaches the acetabulum
        hit_i = exit_i + 1 + int(acetabulum_after[0, 0])

        exit_phys = np.asarray(image.transform_index_to_physical_point(path[exit_i]), dtype=float)
        hit_phys = np.asarray(image.transform_index_to_physical_point(path[hit_i]), dtype=float)
        distance = float(np.linalg.norm(hit_phys - exit_phys))
        if distance > ray_length_mm:
            continue  # generous sanity cap against tangential grazes

        distances.append(distance)
        exit_points.append(exit_phys)
        hit_points.append(hit_phys)

    if len(distances) == 0:
        raise ValueError('No valid rays connected the femoral head to the acetabulum')

    distances = np.array(distances)
    exit_points = np.array(exit_points)
    hit_points = np.array(hit_points)

    if plot is not False and plot is not None:
        if isinstance(plot, pv.Plotter):
            for exit_phys, hit_phys in zip(exit_points, hit_points):
                plot.add_lines(np.array([exit_phys, hit_phys]), color='red', width=2)
            plot.add_points(exit_points, color='blue', point_size=6, render_points_as_spheres=True)
            plot.add_points(hit_points, color='green', point_size=6, render_points_as_spheres=True)
        else:  # assume a matplotlib Axes; show a sagittal maximum-intensity projection
            plot.imshow(femur_mask.max(axis=1).T, cmap='gray')
            for exit_phys, hit_phys in zip(exit_points, hit_points):
                exit_idx = image.transform_physical_point_to_index(exit_phys)
                hit_idx = image.transform_physical_point_to_index(hit_phys)
                plot.plot([exit_idx[0], hit_idx[0]], [exit_idx[2], hit_idx[2]], 'r-', linewidth=0.5)
            plot.set_aspect('equal')
            plot.set_title(f'Subchondral distance: {np.nanmean(distances):.2f} mm ({len(distances)} rays)')

    return float(np.nanmean(distances)), float(np.nanstd(distances)), float(np.nanmin(distances)), float(np.nanmax(distances)), distances, exit_points, hit_points

def calculate_subchondral_distance_ray_tracing(image: Segmentation, side: str = 'left', femur_label: int = 1,
                                               acetabulum_label: int = 3, isotropic: bool = False,
                                               n_rays: int = 200, cone_angle: float = 45.0,
                                               plot: bool | plt.Axes | pv.Plotter = False) -> Tuple[float, float, float, float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Measure the subchondral femoral-head-to-acetabulum distance on an MRI hip segmentation.

    Rays are cast from the femoral head center within a cone aimed at the acetabulum;
    for each ray the femoral head contour exit and the first acetabulum contour hit are
    found and their physical (mm) distance recorded. See
    :func:`_subchondral_distance_ray_tracing` for the algorithm. The mask must already be
    in LPI orientation (call ``transform_coordinate_system`` first).
    :param image: A Segmentation of the (single-side) proximal-femur hip mask.
    :param side: Side of the image (not patient!), either 'left' or 'right'.
    :param femur_label: The label of the femur in the segmentation mask.
    :param acetabulum_label: The label of the acetabulum in the segmentation mask.
    :param isotropic: Whether the image has isotropic voxels.
    :param n_rays: The number of rays to cast within the cone.
    :param cone_angle: The half-angle of the cone in degrees.
    :param plot: A matplotlib Axes / PyVista Plotter to overlay the rays on, or False.
    :return: A tuple ``(mean, std, min, max, distances, exit_points, hit_points)`` in mm.
    """
    assert side in ['left', 'right'], 'Side must be either "left" or "right"'

    r_idx, fhc_idx = get_femoral_head_center(image.array, side=side, segmentation_label=femur_label, isotropic=isotropic)
    femur_mask = np.where(image.array == femur_label, 1, 0)
    acetabulum_mask = np.where(image.array == acetabulum_label, 1, 0)

    return _subchondral_distance_ray_tracing(image, fhc_idx, r_idx, femur_mask, acetabulum_mask,
                                             n_rays=n_rays, cone_angle=cone_angle, plot=plot)

def calculate_subchondral_distance_ray_tracing_ct(femur_image: Segmentation, side: str = 'left', femur_label: int = 1,
                                                  acetabulum_label: int = 7, n_rays: int = 200, cone_angle: float = 45.0,
                                                  plot: bool | plt.Axes | pv.Plotter = False) -> Tuple[float, float, float, float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Measure the subchondral femoral-head-to-acetabulum distance on a whole-leg CT segmentation.

    Identical to :func:`calculate_subchondral_distance_ray_tracing` but uses the CT femoral
    head center (returned in physical space and converted to index space here, mirroring
    ``get_proximal_reference_line_ct`` in femur.py). In whole-leg CT there is no distinct
    acetabulum label, so the hip bone label (7 by default) plays the acetabulum role. The
    mask must already be in LPI orientation.
    :param femur_image: A Segmentation of the (single-side) whole-leg mask.
    :param side: Side of the image (not patient!), either 'left' or 'right'.
    :param femur_label: The label of the femur in the segmentation mask.
    :param acetabulum_label: The label of the hip bone (acetabulum) in the segmentation mask.
    :param n_rays: The number of rays to cast within the cone.
    :param cone_angle: The half-angle of the cone in degrees.
    :param plot: A matplotlib Axes / PyVista Plotter to overlay the rays on, or False.
    :return: A tuple ``(mean, std, min, max, distances, exit_points, hit_points)`` in mm.
    """
    assert side in ['left', 'right'], 'Side must be either "left" or "right"'

    r_phys, fhc_phys = get_femoral_head_center_ct(femur_image, segmentation_label=femur_label, side=side)
    fhc_idx = femur_image.transform_physical_point_to_index(fhc_phys)
    tmp = femur_image.transform_physical_point_to_index(fhc_phys + np.array([r_phys, 0, 0]))
    r_idx = np.linalg.norm(fhc_idx - tmp)

    femur_mask = np.where(femur_image.array == femur_label, 1, 0)
    acetabulum_mask = np.where(femur_image.array == acetabulum_label, 1, 0)

    return _subchondral_distance_ray_tracing(femur_image, fhc_idx, r_idx, femur_mask, acetabulum_mask,
                                             n_rays=n_rays, cone_angle=cone_angle, plot=plot)

def calculate_cartilage_thickness_knn(segmentation_mask: np.ndarray, cartilage_label: int = 2) -> Tuple[float, float, float, float]:
    """
    Calculate the cartilage thickness using a k-nearest neighbors approach.
    :param segmentation_mask: A 3D segmentation mask of the hip.
    :param cartilage_label: The segmentation label of the cartilage.
    :return: The average thickness of the cartilage.
    """
    inner_surface, outer_surface = get_cartilage_inner_and_outer_surface_points(segmentation_mask, cartilage_label=cartilage_label)
    inner_tree = KDTree(inner_surface)
    distances = np.empty(len(outer_surface))

    for i, point in enumerate(outer_surface):
        distance, _ = inner_tree.query(point)
        distances[i] = distance

    return np.nanmean(distances), np.nanstd(distances), np.nanmin(distances), np.nanmax(distances)

def calculate_cartilage_thickness_ray_tracing(segmentation_mask: np.ndarray, cartilage_label: int = 2) -> float:
    """
    Calculate the cartilage thickness using ray tracing.
    :param segmentation_mask: A 3D segmentation mask of the hip.
    :param cartilage_label: The segmentation label of the cartilage.
    :return: The average thickness of the cartilage.
    """
    inner_surface, outer_surface = get_cartilage_inner_and_outer_surface_points(segmentation_mask, cartilage_label=cartilage_label)
    raise NotImplementedError('Ray tracing is not yet implemented for cartilage thickness calculation.')
    return 1.

def calculate_femoral_offset(hip_image: Image, knee_image: Optional[Image] = None, side: str = 'left', femur_label: int = 1, isotropic: bool = False, plot: pv.Plotter | bool = False) -> float:
    """
    Calculate the femoral offset, i.e. the distance between the femoral head center and the femoral shaft axis.
    :param hip_image: Image: A segmentation mask of the proximal femur.
    :param knee_image: Image: A segmentation mask of the knee (optional).
    :param side: str: Side of the image (not patient!), either 'left' or 'right'.
    :param femur_label: int: The label of the femur in the segmentation mask.
    :param isotropic: bool: Whether the image has isotropic voxels.
    :param plot: pv.Plotter: A PyVista plotter to visualize the femoral offset (optional).
    :return: The femoral offset in mm.
    """
    r, c = get_femoral_head_center(hip_image.array, side=side, segmentation_label=femur_label, isotropic=isotropic)
    start, end = get_femoral_shaft_axis(hip_image.array, knee_mask=(knee_image.array if knee_image is not None else None), femur_label=femur_label, isotropic=isotropic)

    _, projection_vector = get_vector_through_point_perpendicular_to_line(start, (end - start), c)

    c_world = hip_image.transform_index_to_physical_point(c)
    projection_vector_world = hip_image.transform_index_to_physical_point(projection_vector)
    femoral_offset = np.linalg.norm(c_world - projection_vector_world)

    if plot:
        femur_coords = np.argwhere(hip_image.array == femur_label).astype(np.float32)
        for i, coord in enumerate(femur_coords):
            femur_coords[i] = hip_image.transform_index_to_physical_point(coord)
            if side == 'right':
                femur_coords[i, 0] -= hip_image.shape[0]  # adjust for right side

        start_world = hip_image.transform_index_to_physical_point(start)
        end_world = hip_image.transform_index_to_physical_point(end)

        if side == 'right':
            c_world[0] -= hip_image.shape[0]  # adjust for right side
            projection_vector_world[0] -= hip_image.shape[0]
            start_world[0] -= hip_image.shape[0]
            end_world[0] -= hip_image.shape[0]

        plot.add_mesh(pv.PolyData(femur_coords), color=('g' if side == 'left' else 'y'), opacity=.1)
        plot.add_lines(np.array([start_world, end_world]), color='red', width=5)
        plot.add_lines(np.array([c_world, projection_vector_world]), color='red', width=5)

    return femoral_offset

def calculate_femoral_offset_projected(hip_image: Segmentation, knee_image: Optional[Image] = None, side: str = 'left', femur_label: int = 1, isotropic: bool = False, ct: bool = False) -> float:
    """
    Calculate the femoral offset, i.e. the distance between the femoral head center and the femoral shaft axis.
    Landmarks are projected to the coronal plane before final calculations.
    :param hip_image: Image: A segmentation mask of the proximal femur.
    :param knee_image: Image: A segmentation mask of the knee (optional).
    :param side: str: Side of the image (not patient!), either 'left' or 'right'.
    :param femur_label: int: The label of the femur in the segmentation mask.
    :param isotropic: bool: Whether the image has isotropic voxels.
    :param ct: bool: Whether the image is a CT scan.
    :return:
    """
    if ct:
        r, c = get_femoral_head_center_ct(hip_image, segmentation_label=femur_label, side=side)
        start, end = get_femoral_shaft_axis_ct(hip_image, segmentation_label=femur_label)
    else:
        r, c = get_femoral_head_center(hip_image.array, side=side, segmentation_label=femur_label, isotropic=isotropic)
        start, end = get_femoral_shaft_axis(hip_image.array,
                                            knee_mask=(knee_image.array if knee_image is not None else None),
                                            femur_label=femur_label, isotropic=isotropic)

    c = np.array([c[0], 0, c[2]])  # zero out coronal component
    start = np.array([start[0], 0, start[2]])
    end = np.array([end[0], 0, end[2]])

    _, projection_vector = get_vector_through_point_perpendicular_to_line(start, (end - start), c)

    femoral_offset = np.linalg.norm(c - projection_vector)

    return femoral_offset
