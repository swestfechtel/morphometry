import numpy as np
import pyvista as pv
import pandas as pd

from morphometry.utils import sphere_fit, get_contour_points, calculate_angle_between_vectors, \
    calculate_min_distance_between_point_clouds, get_vector_through_point_perpendicular_to_line, \
    get_minimum_distance_between_line_and_point, get_contour_points, num_connected_components, \
    extract_connected_components_2d, circumference_points, intersect_ndarrays, \
    sort_points_clockwise, sort_points_by_x, fit_circle_to_points
from morphometry.image_io import Image

from scipy.ndimage import center_of_mass, label, rotate
from scipy.spatial import KDTree
from scipy.stats import zscore
from typing import Tuple, Optional
from skimage.measure import find_contours
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt


def get_femoral_head_center(segmentation_mask: np.ndarray, side: str = 'left', segmentation_label: int = 1, return_layers: bool = False, x_ratio: float = 1., isotropic: bool = False) -> Tuple[float, np.ndarray] | Tuple[float, np.ndarray, int, int]:
    """
    Get the center of the femoral head from a segmentation mask.
    :param segmentation_mask: A segmentation mask of the proximal femur where the femur is 1 and everything else 0.
    :param side: Side of the image (not patient!), either 'left' or 'right'.
    :param segmentation_label: The label of the femur in the segmentation mask.
    :param return_layers: Whether to return layer_high and layer_low.
    :param x_ratio: Correction factor for slice thickness.
    :param isotropic: Whether the image has isotropic voxels.
    :return: The radius and location of the femoral head center.
    """
    assert side in ['left', 'right'], 'Side must be either "left" or "right"'
    segmentation_mask = np.where(segmentation_mask == segmentation_label, 1, 0)

    contour_pts = get_contour_points(segmentation_mask)

    # get highest layer with a mask point, its centroid
    # and lowest layer with mask point on this centroid
    layer_high = np.amin(contour_pts[:, 2])
    layer_sizes = np.zeros(segmentation_mask.shape[2])
    for i in range(segmentation_mask.shape[2]):
        layer = segmentation_mask[:, :, i]
        layer_sizes[i] = np.sum(layer)

    layer_zscores = np.abs(zscore(layer_sizes))
    while layer_zscores[layer_high] > 2:
        layer_high += 1  # go to the next layer if the current one has too few mask points, i.e. its size is more than 2 standard deviations away from the mean of all slices

    # if layer_high has two connected components, choose the larger one
    label_mask, num_features = label(segmentation_mask[:, :, layer_high])
    if num_features > 1:
        sizes = [len(np.argwhere(label_mask == i)) for i in range(1, num_features + 1)]
        largest_component = np.argmax(sizes) + 1
        segmentation_mask[:, :, layer_high] = np.where(label_mask == largest_component, 1, 0)

    com_high = center_of_mass(segmentation_mask[:, :, layer_high])
    com_high = (int(com_high[0]), int(com_high[1]))
    layer_low = layer_high + 1
    while segmentation_mask[com_high[0], com_high[1], layer_low] != 0:
        layer_low += 1

    point_cloud = get_contour_points(segmentation_mask)
    point_cloud = point_cloud[point_cloud[:, 2] >= layer_high]
    point_cloud = point_cloud[point_cloud[:, 2] <= layer_low]  # exclude everything that is not between layer_high and layer_low
    point_cloud = point_cloud.astype(np.float32) * np.array([1, 1, x_ratio])  # adjust for slice thickness

    # need to exclude lateral parts of the mask: compute distance between com and max medial point of femoral head,
    # then exclude everything that is farther away than this distance in the lateral direction
    middle_slice = (layer_high * x_ratio + (layer_low * x_ratio - layer_high * x_ratio) // 2)
    superior_half = point_cloud[point_cloud[:, 2] <= middle_slice]
    if side == 'left':
        max_s = np.max(point_cloud[:, 0]) if isotropic else np.max(superior_half[:, 0]) * 0.9  # only look at the superior half of the femoral head
        radius = max_s - com_high[0]
        min_s = com_high[0] - radius  # the most lateral point of the femoral head
        point_cloud = point_cloud[point_cloud[:, 0] >= min_s]
    else:
        min_s = np.min(point_cloud[:, 0]) if isotropic else np.min(superior_half[:, 0]) * 0.9 # only look at the superior half of the femoral head
        radius = com_high[0] - min_s
        max_s = com_high[0] + radius  # the most lateral point of the femoral head
        point_cloud = point_cloud[point_cloud[:, 0] <= max_s]

    min_c = np.min(point_cloud[:, 1])
    radius = com_high[1] - min_c
    max_c = com_high[1] + radius
    point_cloud = point_cloud[point_cloud[:, 1] <= max_c]

    # get center coordinates of the fitting sphere
    r, center = sphere_fit(point_cloud)

    # compensate pixel mm ratio between x, y and z axis
    center = np.array([center[0], center[1], center[2] / x_ratio])

    return (r, center) if not return_layers else (r, center, layer_high, layer_low)


def get_femoral_neck_center(segmentation_mask: np.ndarray, femoral_head_center: Tuple[float, np.ndarray], side: str = 'left', segmentation_label: int = 1, x_ratio: float = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get the endpoint of the femoral neck axis from a segmentation mask.
    :param segmentation_mask: A segmentation mask of the proximal femur where the femur is 1 and everything else 0.
    :param femoral_head_center: The radius and center of the femoral head.
    :param side: Side of the image (not patient!), either 'left' or 'right'.
    :param segmentation_label: The label of the femur in the segmentation mask.
    :param x_ratio: Correction factor for slice thickness.
    :return: The points constituting the femoral neck and the endpoint of the femoral neck axis.
    """
    assert side in ['left', 'right'], 'Side must be either "left" or "right"'
    segmentation_mask = np.where(segmentation_mask == segmentation_label, 1, 0)

    r, center = femoral_head_center
    point_cloud = np.argwhere(segmentation_mask)

    point_cloud = point_cloud * np.array([1, 1, x_ratio])  # adjust for slice thickness
    center = center * np.array([1, 1, x_ratio])

    # Get a sphere around the femoral head center with a radius of 1.2 times the femoral head radius
    # This sphere only includes the points between r and 1.2*r, i.e. is hollow
    solid_sphere = pv.SolidSphere(inner_radius=r, outer_radius=1.2 * r, center=center)

    # Get the points that are distal to the femoral head center
    points_i_want = np.array(solid_sphere.points)
    points_i_want = points_i_want[points_i_want[:, 2] > center[2]]

    # Get the points that are lateral to the femoral head center
    if side == 'left':
        points_i_want = points_i_want[points_i_want[:, 0] < center[0]]
    else:
        points_i_want = points_i_want[points_i_want[:, 0] > center[0]]

    com = KMeans(n_clusters=1).fit(points_i_want).cluster_centers_[0]
    distances = np.linalg.norm(points_i_want - com, axis=1)
    points_i_want = points_i_want[distances <= distances.mean() + 2 * distances.std()]  # remove outliers

    # Build a KDTree for the point cloud and the points we want
    pc_tree = KDTree(point_cloud)
    sphere_tree = KDTree(points_i_want)

    # Get the intersection between the two point clouds, i.e. the points that should constitute the
    # femoral neck
    # pairs = pc_tree.query_ball_tree(sphere_tree, 2)
    pairs = sphere_tree.query_ball_tree(pc_tree, 2)
    neck_points = list()
    for pair in pairs:
        if len(pair) > 0:
            for index in pair:
                # neck_points.append(sphere_tree.data[index])
                # if np.linalg.norm(point_cloud[index] - center) <= 1.2 * r:
                neck_points.append(point_cloud[index])

    neck_points = np.array(neck_points)

    # Get the center of mass of these points
    com = KMeans(n_clusters=1).fit(neck_points).cluster_centers_[0]
    com = np.array([com[0], com[1], com[2]])

    """
    p = pv.Plotter()
    p.add_mesh(pv.PolyData(point_cloud.astype(np.float32)), color='g', opacity=.5)
    p.add_mesh(pv.PolyData(neck_points.astype(np.float32)), color='r')
    # p.add_mesh(pv.PolyData(points_i_want * np.array([1, 1, 1])), color='b', opacity=.5)
    p.add_lines(np.array([center, com]), color='k')
    p.show()
    """

    neck_points = neck_points / np.array([1, 1, x_ratio])
    com = com / np.array([1, 1, x_ratio])

    return neck_points, com


def get_femoral_shaft_axis(hip_mask: np.ndarray, knee_mask: np.ndarray = None, femur_label: int = 1, isotropic: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get the femoral shaft axis from a segmentation mask.
    :param hip_mask: A segmentation mask of the hip.
    :param knee_mask: A segmentation mask of the knee.
    :param femur_label: The label of the femur in the segmentation mask.
    :param isotropic: Whether the image has isotropic voxels.
    :return: Start and end point of the vector representing the femoral shaft axis.
    """
    hip_mask = np.where(hip_mask == femur_label, 1, 0)

    point_cloud_hip = np.argwhere(hip_mask)

    offset = 5  # TODO derive this dynamically
    layer_low_hip = np.max(point_cloud_hip[:, 2]) - (offset if isotropic else 1)  # get the most distal layer with a mask point

    com_low_hip = center_of_mass(hip_mask[:, :, layer_low_hip])
    com_low_hip = (int(com_low_hip[0]), int(com_low_hip[1]), layer_low_hip)

    if knee_mask is not None:
        knee_mask = np.where(knee_mask == femur_label, 1, 0)
        point_cloud_knee = np.argwhere(knee_mask)
        layer_high_knee = np.min(point_cloud_knee[:, 2]) + (
            offset if isotropic else 1)  # get the most proximal layer with a mask point
        com_high_knee = center_of_mass(knee_mask[:, :, layer_high_knee])
        com_high_knee = (int(com_high_knee[0]), int(com_high_knee[1]), layer_high_knee)

        return np.array(com_high_knee), np.array(com_low_hip)

    # find the most proximal layer of the greater trochanter, which will serve as the high point of the femoral shaft axis
    connected_components = list()

    two_components_found = False
    components_large_enough = False
    for layer in range(hip_mask.shape[2]):
        connected_components = extract_connected_components_2d(hip_mask[:, :, layer])
        if len(connected_components) == 2:
            two_components_found = True

            if isotropic and (np.count_nonzero(connected_components[0]) < 10 or np.count_nonzero(connected_components[1]) < 10):  # avoid small components that belong to the femoral head
                continue

            components_large_enough = True

            break
    else:
        if not two_components_found:
            raise RuntimeError('No two connected components found in segmentation mask')
        elif not components_large_enough:
            raise RuntimeError('Connected components are too small in segmentation mask')
        else:
            raise RuntimeError('Tip of greater trochanter not found in segmentation mask')

    layer_high_hip = layer  # the layer with two connected components is the one we want
    # get the center of mass of the smaller component
    smaller_component = min(connected_components, key=lambda x: np.count_nonzero(x))
    com_high_hip = center_of_mass(smaller_component)

    com_high_hip = (int(com_high_hip[0]), int(com_high_hip[1]), layer_high_hip)

    return np.array(com_low_hip), np.array(com_high_hip)  # return the two points as the femoral shaft axis


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


def calculate_anteversion(segmentation_mask: Image, side: str = 'left', segmentation_label: int = 1, isotropic: bool = False, plot: Tuple[plt.axis, plt.axis] | bool = False) -> float | Tuple[float, plt.Figure]:
    """
    Calculate the anteversion of the femur.
    :param segmentation_mask: A segmentation mask of the proximal femur.
    :param side: Side of the image (not patient!), either 'left' or 'right'.
    :param segmentation_label: The label of the femur in the segmentation mask.
    :param isotropic: Whether the image has isotropic voxels.
    :param plot: Whether to plot the results.
    :return: The anteversion angle.
    """
    # TODO revise
    assert side in ['left', 'right'], 'Side must be either "left" or "right"'

    from morphometry.femur import get_proximal_reference_line
    landmarks = get_proximal_reference_line(segmentation_mask, side=side, method='murphy', segmentation_label=segmentation_label, isotropic=isotropic, x_ratio=1)

    hip_start = landmarks[0]
    fhc_layer = hip_start[2]
    hip_end = landmarks[1]
    hip_start[2] = hip_end[2]  # adjust axial coordinate to align layers

    proximal_line = hip_end - hip_start
    x = np.array([-1, 0, 0]) if side == 'left' else np.array(
        [1, 0, 0])  # need to distinguish between left and right image side
    proximal_angle = calculate_angle_between_vectors(proximal_line, x)

    if proximal_angle > 90:
        proximal_angle = 180 - proximal_angle

    proximal_orientation = hip_end[1] - hip_start[1]  # positive if hip_end is posterior to hip_start
    if proximal_orientation < 0:  # if hip_end is anterior to hip_start, the angle is negative
        proximal_angle = -proximal_angle

    distal_angle = 0

    angle = proximal_angle - distal_angle

    if plot:
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


def get_femoral_neck_transition(neck_points: np.ndarray, side: str = 'left') -> np.ndarray:
    """
    Get the point where the femoral neck transitions into the femoral head.
    :param neck_points: The points constituting the femoral neck.
    :param side: Side of the image (not patient!), either 'left' or 'right'
    :return: The point where the femoral neck transitions into the femoral head.
    """
    assert side in ['left', 'right'], 'Side must be either "left" or "right"'

    most_proximal_points = neck_points[neck_points[:, 2] == np.min(neck_points[:, 2])]  # get the most proximal points

    if side == 'left':
        most_proximal_medial_point = most_proximal_points[most_proximal_points[:, 0] == np.max(most_proximal_points[:, 0])]  # of the most proximal points, get the most medial one
    else:
        most_proximal_medial_point = most_proximal_points[most_proximal_points[:, 0] == np.min(most_proximal_points[:, 0])]  # of the most proximal points, get the most medial one

    return most_proximal_medial_point[0]  # this point is one possible transition point


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


def get_p1(acetabulum_array: np.ndarray, side: str = 'left', segmentation_label: int = 3, fhc_y_index: int = -1) -> np.ndarray:
    """
    Get the posterior acetabulum rim.
    :param acetabulum_array: A 2D segmentation mask of the acetabulum
    :param side: Side of the image (not patient!), either 'left' or 'right'.
    :param segmentation_label: The label of the acetabulum in the segmentation mask.
    :param fhc_y_index: The y index of the femoral head center, used to determine anterior and posterior.
    :return: The posterior acetabulum rim.
    """
    assert side in ['left', 'right'], 'Side must be either "left" or "right"'
    acetabulum_array = np.where(acetabulum_array == segmentation_label, 1, 0)

    if fhc_y_index == -1:
        fhc_y_index = acetabulum_array.shape[1] // 3  # fallback
    acetabulum_array[:, :fhc_y_index] = 0  # zero out the anterior one-third of the acetabulum

    # p1 = np.argwhere(acetabulum_array[:, acetabulum_array.shape[1] // 3:])  # only look at the posterior two-thirds of the acetabulum
    p1 = np.argwhere(acetabulum_array)
    p1 = p1[p1[:, 0].argmin()] if side == 'left' else p1[p1[:, 0].argmax()]

    return p1


def get_p2(acetabulum_array: np.ndarray, side: str = 'left', segmentation_label: int = 3, fhc_y_index: int = -1) -> np.ndarray:
    """
    Get the anterior acetabulum rim.
    :param acetabulum_array: A 2D segmentation mask of the acetabulum
    :param side: Side of the image (not patient!), either 'left' or 'right'.
    :param segmentation_label: The label of the acetabulum in the segmentation mask.
    :param fhc_y_index: The y index of the femoral head center, used to determine anterior and posterior.
    :return: The anterior acetabulum rim.
    """
    assert side in ['left', 'right'], 'Side must be either "left" or "right"'
    acetabulum_array = np.where(acetabulum_array == segmentation_label, 1, 0)

    if fhc_y_index == -1:
        fhc_y_index = acetabulum_array.shape[1] // 3  # fallback
    acetabulum_array[:, fhc_y_index:] = 0

    # p2 = np.argwhere(acetabulum_array[:, :acetabulum_array.shape[1] // 3])  # only look at the anterior one-third of the acetabulum
    p2 = np.argwhere(acetabulum_array)
    p2 = p2[p2[:, 0].argmin()] if side == 'left' else p2[p2[:, 0].argmax()]

    return p2


def calculate_acetabular_anteversion(segmentation_mask: np.ndarray, femur_label: int = 1, acetabulum_label: int = 3, isotropic: bool = False, plot: bool = False, fp: str = None) -> Tuple[float, float]:
    """
    Calculate the acetabular anteversion for both sides from a segmentation mask.
    :param segmentation_mask: A segmentation mask of the hip.
    :param femur_label: The label of the femur in the segmentation mask.
    :param acetabulum_label: The label of the acetabulum in the segmentation mask.
    :param isotropic: Whether the image has isotropic voxels.
    :param plot: Whether to plot the results.
    :param fp: File path to save the plot.
    :return: The acetabular anteversion for both sides.
    """
    left_mask = segmentation_mask[:segmentation_mask.shape[0] // 2]
    right_mask = segmentation_mask[segmentation_mask.shape[0] // 2:]

    _, left_fhc = get_femoral_head_center(left_mask, side='left', segmentation_label=femur_label, isotropic=isotropic)
    _, right_fhc = get_femoral_head_center(right_mask, side='right', segmentation_label=femur_label, isotropic=isotropic)

    slice_gap = abs(int(left_fhc[2]) - int(right_fhc[2]))
    correct_slice = min(int(left_fhc[2]), int(right_fhc[2])) + slice_gap // 2

    p1_left = get_p1(left_mask[:, :, correct_slice], side='left', segmentation_label=acetabulum_label, fhc_y_index=int(left_fhc[1]))
    p2_left = get_p2(left_mask[:, :, correct_slice], side='left', segmentation_label=acetabulum_label, fhc_y_index=int(left_fhc[1]))

    p1_right = get_p1(right_mask[:, :, correct_slice], side='right', segmentation_label=acetabulum_label, fhc_y_index=int(right_fhc[1]))
    p2_right = get_p2(right_mask[:, :, correct_slice], side='right', segmentation_label=acetabulum_label, fhc_y_index=int(right_fhc[1]))

    femur_array = np.where(segmentation_mask == femur_label, 1, 0)
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

    if plot:
        fig, ax = plt.subplots(figsize=(20, 10))
        ax.imshow(segmentation_mask[:, :, correct_slice].T, cmap='gray')
        ax.plot([p1_left[0], p2_left[0]], [p1_left[1], p2_left[1]], 'r-', label='Right Acetabulum Rim')
        ax.plot([p1_right[0] + left_femur.shape[0], p2_right[0] + left_femur.shape[0]], [p1_right[1], p2_right[1]], 'b-', label='Left Acetabulum Rim')
        ax.plot([left_fhc[0], right_fhc_adj[0]], [left_fhc[1], right_fhc_adj[1]], 'g-', label='Femoral Head Centers')
        # ax.plot([p1_left[0], p1_left[0] - s_left[0]], [p1_left[1], p1_left[1] - s_left[1]], 'c--', label='Right Perpendicular Vector')
        # ax.plot([p1_right[0] + left_femur.shape[0], p1_right[0] - s_right[0] + left_femur.shape[0]], [p1_right[1], p1_right[1] - s_right[1]], 'm--', label='Left Perpendicular Vector')
        ax.plot([p1_left[0], projection_left[0]], [p1_left[1], projection_left[1]], 'c--', label='Right Perpendicular Projection')
        ax.plot([p1_right[0] + left_femur.shape[0], projection_right[0] + left_femur.shape[0]], [p1_right[1], projection_right[1]], 'm--', label='Left Perpendicular Projection')
        ax.set_title(f'Right AA: {left_aa:.2f}°, Left AA: {right_aa:.2f}°')
        ax.set_aspect('equal')
        ax.legend()
        fig.savefig(fp)
        plt.close(fig)

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


def calculate_center_edge_angle(segmentation_mask: np.ndarray, femur_label: int = 1, acetabulum_label: int = 3, isotropic: bool = False, project: bool = False, plot: bool = False, fp: str = None) -> Tuple[float, float]:
    """
    Calculate the center edge angle for both sides from a segmentation mask.
    :param segmentation_mask: A segmentation mask of the proximal femur.
    :param femur_label: The label of the femur in the segmentation mask.
    :param acetabulum_label: The label of the acetabulum in the segmentation mask.
    :param isotropic: Whether the image has isotropic voxels.
    :param project: Whether to project all landmarks onto a 2D plane before calculating the angle.
    :param plot: Whether to plot the results.
    :param fp: File path to save the plot.
    :return: The center edge angle for both sides.
    """
    left_mask = segmentation_mask[:segmentation_mask.shape[0] // 2]
    right_mask = segmentation_mask[segmentation_mask.shape[0] // 2:]

    r_l, left_fhc = get_femoral_head_center(left_mask, side='left', segmentation_label=femur_label, isotropic=isotropic)
    r_r, right_fhc = get_femoral_head_center(right_mask, side='right', segmentation_label=femur_label, isotropic=isotropic)

    coronal_slice = int((right_fhc[1] + left_fhc[1]) // 2)

    if project:
        left_fhc[1] = 0
        right_fhc[1] = 0

    femur_array = np.where(segmentation_mask == femur_label, 1, 0)
    left_femur = femur_array[:femur_array.shape[0] // 2]
    right_femur = femur_array[femur_array.shape[0] // 2:]

    right_fhc_adj = right_fhc.copy()
    right_fhc_adj[0] += left_femur.shape[
        0]  # adjust the x coordinate of the right femoral head center to account for the splitting into left and right
    G = left_fhc - right_fhc_adj  # G is the vector connecting the left and right femoral head center

    def get_lateral_edge_point(fa: np.ndarray, aa: np.ndarray, side: str = 'left') -> np.ndarray:
        """
        Get the most lateral edge point of the acetabulum right above the femoral head.
        :param fa: A segmentation mask of the femur.
        :param aa: A segmentation mask of the acetabulum.
        :param side: Side of the image (not patient!), either 'left' or 'right'
        :return:
        """
        assert side in ['left', 'right'], 'Side must be either "left" or "right"'

        most_proximal_femur_slice = np.min(np.nonzero(fa)[2])
        tmp = aa.copy()
        ul = int((right_fhc[2] - 1.1 * r_r) if side == 'right' else (left_fhc[2] - 1.5 * r_l))
        tmp[:, :, :ul] = 0
        # tmp[:, :, most_proximal_femur_slice + 10:] = 0
        acetabulum_points = np.nonzero(tmp)

        # Find the most lateral sagittal coordinate
        lateral_s = np.min(acetabulum_points[0]) if side == 'left' else np.max(acetabulum_points[0])

        # Find all points with this sagittal coordinate
        candidates_mask = acetabulum_points[0] == lateral_s
        candidate_coronal = acetabulum_points[1][candidates_mask]
        candidate_transversal = acetabulum_points[2][candidates_mask]

        # Select the one with maximum transversal coordinate
        max_transversal_idx = np.argmax(candidate_transversal)
        lateral_edge_point = np.array(
            [lateral_s, candidate_coronal[max_transversal_idx], candidate_transversal[max_transversal_idx]])

        if project:
            lateral_edge_point[1] = 0

        return lateral_edge_point

    acetabulum_array = np.where(segmentation_mask == acetabulum_label, 1, 0)
    left_acetabulum = acetabulum_array[:acetabulum_array.shape[0] // 2]
    right_acetabulum = acetabulum_array[acetabulum_array.shape[0] // 2:]

    d = np.array([0, 0, -1])
    n = G / np.linalg.norm(G)
    d_perp = d - np.dot(d, n) * n  # d_perp is perpendicular to G and goes in proximal direction
    d_perp *= 100  # scale the perpendicular vector to a reasonable length
    s_right = right_fhc_adj + d_perp  # s is a point from the femoral head center with direction d_perp, just for visualisation
    lat_right = get_lateral_edge_point(right_femur, right_acetabulum, side='right')
    s2_right = lat_right - right_fhc
    cea_right = calculate_angle_between_vectors(d_perp, s2_right)

    s_left = left_fhc + d_perp
    lat_left = get_lateral_edge_point(left_femur, left_acetabulum)
    s2_left = lat_left - left_fhc
    cea_left = calculate_angle_between_vectors(d_perp, s2_left)

    if plot:
        fig, ax = plt.subplots(figsize=(20, 10))
        ax.imshow(segmentation_mask[:, coronal_slice].T, cmap='gray')
        ax.plot([left_fhc[0], right_fhc_adj[0]], [left_fhc[2], right_fhc_adj[2]], 'r-', label='G')

        ax.plot([left_fhc[0], s_left[0]], [left_fhc[2], s_left[2]], 'g--', label='Perpendicular Vector (right)')
        ax.plot([right_fhc_adj[0], s_right[0]], [right_fhc_adj[2], s_right[2]], 'b--', label='Perpendicular Vector (left)')

        ax.plot([left_fhc[0], lat_left[0]], [left_fhc[2], lat_left[2]], 'c-', label='Lateral Edge Point (right)')
        ax.plot([right_fhc_adj[0], lat_right[0] + left_mask.shape[0]], [right_fhc_adj[2], lat_right[2]], 'y-', label='Lateral Edge Point (left)')
        ax.set_title(f'Right CEA: {cea_left:.2f}°, Left CEA: {cea_right:.2f}°')
        ax.set_aspect('equal')
        ax.legend()
        fig.savefig(fp)
        plt.close(fig)

    return cea_left, cea_right


def calculate_center_edge_angle_2d(segmentation_mask: np.ndarray, femur_label: int = 1, acetabulum_label: int = 3, isotropic: bool = False, plot: bool = False, fp: str = None) -> Tuple[float, float]:
    """
    Calculate the center edge angle for both sides from a segmentation mask. Finds the landmarks on a 2D coronal plane.
    :param segmentation_mask: A segmentation mask of the proximal femur.
    :param femur_label: The label of the femur in the segmentation mask.
    :param acetabulum_label: The label of the acetabulum in the segmentation mask.
    :param isotropic: Whether the image has isotropic voxels.
    :param plot: Whether to plot the results.
    :param fp: File path to save the plot.
    :return: The center edge angle for both sides.
    """
    def get_lateral_edge_point(fa: np.ndarray, aa: np.ndarray, side: str = 'left') -> np.ndarray:
        """
        Get the most lateral edge point of the acetabulum right above the femoral head.
        :param fa: A segmentation mask of the femur.
        :param aa: A segmentation mask of the acetabulum.
        :param side: Side of the image (not patient!), either 'left' or 'right'
        :return:
        """
        assert side in ['left', 'right'], 'Side must be either "left" or "right"'

        tmp = aa.copy()
        ul = int((right_fhc[1] - 1.5 * r_r) if side == 'right' else (left_fhc[1] - 1.5 * r_l))
        tmp[:, :, :ul] = 0
        tmp = tmp[:, coronal_slice]

        acetabulum_points = np.nonzero(tmp)

        lateral_s = np.min(acetabulum_points[0]) if side == 'left' else np.max(acetabulum_points[0])

        # Find all points with this sagittal coordinate
        candidates_mask = acetabulum_points[0] == lateral_s
        candidate_transversal = acetabulum_points[1][candidates_mask]

        # Select the one with maximum transversal coordinate
        max_transversal_idx = np.argmax(candidate_transversal)
        lateral_edge_point = np.array(
            [lateral_s, candidate_transversal[max_transversal_idx]])

        return lateral_edge_point

    left_mask = segmentation_mask[:segmentation_mask.shape[0] // 2]
    right_mask = segmentation_mask[segmentation_mask.shape[0] // 2:]

    r_l, left_fhc = get_femoral_head_center(left_mask, side='left', segmentation_label=femur_label, isotropic=isotropic)
    r_r, right_fhc = get_femoral_head_center(right_mask, side='right', segmentation_label=femur_label, isotropic=isotropic)

    coronal_slice = int((left_fhc[1] + right_fhc[1]) // 2)
    left_fhc = np.array([left_fhc[0], left_fhc[2]])
    right_fhc = np.array([right_fhc[0], right_fhc[2]])

    femur_array = np.where(segmentation_mask == femur_label, 1, 0)
    left_femur = femur_array[:femur_array.shape[0] // 2]
    right_femur = femur_array[femur_array.shape[0] // 2:]

    right_fhc_adj = right_fhc.copy()
    right_fhc_adj[0] += left_femur.shape[
        0]  # adjust the x coordinate of the right femoral head center to account for the splitting into left and right
    G = left_fhc - right_fhc_adj  # G is the vector connecting the left and right femoral head center

    acetabulum_array = np.where(segmentation_mask == acetabulum_label, 1, 0)
    left_acetabulum = acetabulum_array[:acetabulum_array.shape[0] // 2]
    right_acetabulum = acetabulum_array[acetabulum_array.shape[0] // 2:]

    d = np.array([0, -1])
    n = G / np.linalg.norm(G)
    d_perp = d - np.dot(d, n) * n  # d_perp is perpendicular to G and goes in proximal direction
    d_perp *= 100  # scale the perpendicular vector to a reasonable length
    s_right = right_fhc_adj + d_perp  # s is a point from the femoral head center with direction d_perp, just for visualisation
    lat_right = get_lateral_edge_point(right_femur, right_acetabulum, side='right')
    s2_right = lat_right - right_fhc
    cea_right = calculate_angle_between_vectors(d_perp, s2_right)

    s_left = left_fhc + d_perp
    lat_left = get_lateral_edge_point(left_femur, left_acetabulum)
    s2_left = lat_left - left_fhc
    cea_left = calculate_angle_between_vectors(d_perp, s2_left)

    if plot:
        fig, ax = plt.subplots(figsize=(20, 10))
        ax.imshow(segmentation_mask[:, coronal_slice].T, cmap='gray')
        ax.plot([left_fhc[0], right_fhc_adj[0]], [left_fhc[1], right_fhc_adj[1]], 'r-', label='G')

        ax.plot([left_fhc[0], s_left[0]], [left_fhc[1], s_left[1]], 'g--', label='Perpendicular Vector (right)')
        ax.plot([right_fhc_adj[0], s_right[0]], [right_fhc_adj[1], s_right[1]], 'b--', label='Perpendicular Vector (left)')

        ax.plot([left_fhc[0], lat_left[0]], [left_fhc[1], lat_left[1]], 'c-', label='Lateral Edge Point (right)')
        ax.plot([right_fhc_adj[0], lat_right[0] + left_mask.shape[0]], [right_fhc_adj[1], lat_right[1]], 'y-', label='Lateral Edge Point (left)')
        ax.set_title(f'Right CEA: {cea_left:.2f}°, Left CEA: {cea_right:.2f}°')
        ax.set_aspect('equal')
        ax.legend()
        fig.savefig(fp)
        plt.close(fig)

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


def get_cartilage_inner_and_outer_surface_points(segmentation_mask: np.ndarray, cartilage_label: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get the inner and outer surface points of the cartilage.
    :param segmentation_mask: A 3D segmentation mask of the hip.
    :param cartilage_label: The segmentation label of the cartilage.
    :return: The inner and outer surface points.
    """
    cartilage = np.argwhere(segmentation_mask == cartilage_label)
    radius, center = sphere_fit(cartilage)  # fit a sphere to the cartilage

    # define one sphere that is "inside" of the cartilage and one that is "outside" of it
    resolution = int(np.sqrt(len(cartilage) * 2))  # resolution of the sphere; assuming ~half of the sphere is inside the cartilage, it needs to have twice the number of cartilage points
    inner_sphere = pv.Sphere(radius=0.8 * radius, center=center, theta_resolution=resolution, phi_resolution=resolution)
    outer_sphere = pv.Sphere(radius=1.2 * radius, center=center, theta_resolution=resolution, phi_resolution=resolution)

    # extract the "inner" and "outer" surface points of the cartilage
    inner_surface, outer_surface = dict(), dict()
    cartilage_tree = KDTree(cartilage)

    for point in inner_sphere.points:  # for every sphere point, find the closest cartilage point
        distance, index = cartilage_tree.query(point)
        if index in inner_surface.keys():
            if distance > inner_surface[index]:
                continue  # skip if the cartilage point has already been added with a smaller distance
        inner_surface[index] = distance  # save index and distance for the corresponding cartilage point

    for point in outer_sphere.points:
        distance, index = cartilage_tree.query(point)
        if index in outer_surface.keys():
            if distance > outer_surface[index]:
                continue
        outer_surface[index] = distance

    # need to filter out points that are too far away from the sphere points
    # this is necessary because ~half the sphere is not covered by the cartilage and thus these sphere points
    # should have no corresponding cartilage point
    inner_distances = pd.Series(inner_surface.values())
    outer_distances = pd.Series(outer_surface.values())

    # get the indices of the points that are within the 75th percentile of the distances
    inner_surface_indices = [k for k, v in inner_surface.items() if v < inner_distances.quantile(0.75)]
    outer_surface_indices = [k for k, v in outer_surface.items() if v < outer_distances.quantile(0.75)]

    inner_surface = cartilage[inner_surface_indices]
    outer_surface = cartilage[outer_surface_indices]

    return inner_surface, outer_surface


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


def calculate_femoral_offset_projected(hip_image: Image, knee_image: Optional[Image] = None, side: str = 'left', femur_label: int = 1, isotropic: bool = False, plot: pv.Plotter | bool = False) -> float:
    """
    Calculate the femoral offset, i.e. the distance between the femoral head center and the femoral shaft axis.
    Landmarks are projected to the coronal plane before final calculations.
    :param hip_image: Image: A segmentation mask of the proximal femur.
    :param knee_image: Image: A segmentation mask of the knee (optional).
    :param side: str: Side of the image (not patient!), either 'left' or 'right'.
    :param femur_label: int: The label of the femur in the segmentation mask.
    :param isotropic: bool: Whether the image has isotropic voxels.
    :param plot: pv.Plotter | bool: A PyVista plotter to visualize the femoral offset, or False if no plotting desired.
    :return:
    """
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
