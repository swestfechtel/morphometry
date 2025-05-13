import numpy as np
import pyvista as pv
import pandas as pd
import pyvista as pv

from morphometry.utils import sphere_fit, get_contour_points, calculate_angle_between_vectors, \
    calculate_min_distance_between_point_clouds, get_vector_through_point_perpendicular_to_line, \
    get_minimum_distance_between_line_and_point, get_contour_points
from morphometry.image_io import Image
from scipy.ndimage import center_of_mass, label
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
    layer_low = layer_high
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


def get_femoral_neck_center(segmentation_mask: np.ndarray, femoral_head_center: Tuple[float, np.ndarray], side: str = 'left', segmentation_label: int = 1, x_ratio: float = None) -> Tuple[np.ndarray, np.ndarray]:
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

    offset = 20 if isotropic else 5  # TODO derive this dynamically
    layer_low_hip = np.max(point_cloud_hip[:, 2]) - (offset if isotropic else 1)  # get the most distal layer with a mask point
    layer_high_hip = layer_low_hip - offset  # get a layer superior to that with some distance, distance depends on the resolution of the image

    com_low_hip = center_of_mass(hip_mask[:, :, layer_low_hip])
    com_low_hip = (int(com_low_hip[0]), int(com_low_hip[1]), layer_low_hip)
    com_high_hip = center_of_mass(hip_mask[:, :, layer_high_hip])
    com_high_hip = (int(com_high_hip[0]), int(com_high_hip[1]), layer_high_hip)  # get centers of mass of both layers

    if knee_mask is not None:
        knee_mask = np.where(knee_mask == femur_label, 1, 0)
        point_cloud_knee = np.argwhere(knee_mask)
        layer_high_knee = np.min(point_cloud_knee[:, 2]) + (
            offset if isotropic else 1)  # get the most proximal layer with a mask point
        com_high_knee = center_of_mass(knee_mask[:, :, layer_high_knee])
        com_high_knee = (int(com_high_knee[0]), int(com_high_knee[1]), layer_high_knee)

        return np.array(com_high_knee), np.array(com_high_hip)

    return np.array(com_low_hip), np.array(com_high_hip)  # return the two points as the femoral shaft axis


def calculate_ccd(hip_image: Image, knee_image: Image = None, side: str = 'left', segmentation_label: int = 1, isotropic: bool = False, x_ratio: float = 1., debug: bool = False, plot: bool = False) -> Tuple[float, float] | Tuple[float, float, plt.Figure]:
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

    hip_mask = hip_image.get_array()
    r, femoral_head_center = get_femoral_head_center(hip_mask, side=side, segmentation_label=segmentation_label, x_ratio=x_ratio, isotropic=isotropic)
    femoral_neck_points, femoral_neck_center = get_femoral_neck_center(hip_mask, (r, femoral_head_center), side=side, segmentation_label=segmentation_label, x_ratio=x_ratio)

    knee_mask = knee_image.get_array() if knee_image else None

    shaft_axis_low, shaft_axis_high = get_femoral_shaft_axis(hip_mask, knee_mask, femur_label=segmentation_label,
                                                             isotropic=isotropic)

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

    projected_ccd = 180 - projected_ccd if projected_ccd < 90 else projected_ccd

    if plot:
        fig, ax = plt.subplots()

        if knee_image is not None:
            ax.imshow(hip_mask[:, int(femoral_head_center_orig[1]), :].T)
            ax.plot([femoral_head_center_orig[0], femoral_neck_center_orig[0]], [femoral_head_center_orig[2], femoral_neck_center_orig[2]],
                    'r-')
            # ax.plot([shaft_axis_high_orig[0], shaft_axis_low_orig[0]], [shaft_axis_high_orig[2], shaft_axis_low_orig[2]], 'g-')
        else:
            ax.imshow(hip_mask[:, int(femoral_head_center[1]), :].T)
            ax.plot([shaft_axis_high[0], shaft_axis_low[0]], [shaft_axis_high[2], shaft_axis_low[2]], 'g-')
            ax.plot([femoral_head_center[0], femoral_neck_center[0]], [femoral_head_center[2], femoral_neck_center[2]],
                    'r-')

        ax.set_aspect(x_ratio)
        return ccd, projected_ccd, fig

    return ccd, projected_ccd


def calculate_anteversion(segmentation_mask: np.ndarray, side: str = 'left', segmentation_label: int = 1, isotropic: bool = False) -> float:
    """
    Calculate the anteversion of the femur.
    :param segmentation_mask: A segmentation mask of the proximal femur.
    :param side: Side of the image (not patient!), either 'left' or 'right'.
    :param segmentation_label: The label of the femur in the segmentation mask.
    :param isotropic: Whether the image has isotropic voxels.
    :return: The anteversion angle.
    """
    # TODO revise
    assert side in ['left', 'right'], 'Side must be either "left" or "right"'

    r, femoral_head_center = get_femoral_head_center(segmentation_mask, side=side, segmentation_label=segmentation_label, isotropic=isotropic)
    femoral_neck_points, femoral_neck_center = get_femoral_neck_center(segmentation_mask, (r, femoral_head_center),
                                                                       side=side, segmentation_label=segmentation_label)
    horizontal_axis = np.array([0, 0, (1 if side == 'left' else -1)]).astype('float32')  # the horizontal axis in the image
    at = calculate_angle_between_vectors(horizontal_axis[1:], femoral_neck_center[1:] - femoral_head_center[1:])
    return at if at < 90 else 180 - at


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


def calculate_alpha_angle(segmentation_mask: np.ndarray, side: str = 'left', segmentation_label: int = 1, isotropic: bool = False) -> float:
    """
    Calculate the alpha angle from the femoral head center and the femoral neck transition.
    :param segmentation_mask: A segmentation mask of the proximal femur.
    :param side: Side of the image (not patient!), either 'left' or 'right'.
    :param segmentation_label: The label of the femur in the segmentation mask.
    :param isotropic: Whether the image has isotropic voxels.
    :return: The alpha angle.
    """
    assert side in ['left', 'right'], 'Side must be either "left" or "right"'

    r, femoral_head_center = get_femoral_head_center(segmentation_mask, side=side, segmentation_label=segmentation_label, isotropic=isotropic)
    femoral_neck_points, femoral_neck_center = get_femoral_neck_center(segmentation_mask, (r, femoral_head_center), side=side, segmentation_label=segmentation_label)
    femoral_neck_transition = get_femoral_neck_transition(femoral_neck_points, side=side)

    # Calculate the angle between the femoral neck axis and the transition axis
    neck_vector = femoral_neck_center - femoral_head_center
    transition_vector = femoral_neck_transition - femoral_head_center
    alpha = calculate_angle_between_vectors(neck_vector, transition_vector)

    return alpha


def get_p1(acetabulum_array: np.ndarray, side: str = 'left', segmentation_label: int = 3) -> np.ndarray:
    """
    Get the posterior acetabulum rim.
    :param acetabulum_array: A 2D segmentation mask of the acetabulum
    :param side: Side of the image (not patient!), either 'left' or 'right'.
    :param segmentation_label: The label of the acetabulum in the segmentation mask.
    :return: The posterior acetabulum rim.
    """
    assert side in ['left', 'right'], 'Side must be either "left" or "right"'
    acetabulum_array = np.where(acetabulum_array == segmentation_label, 1, 0)

    p1 = np.argwhere(acetabulum_array)
    p1 = p1[p1[:, 1].argmin()] if side == 'left' else p1[p1[:, 1].argmax()]

    return p1


def get_p2(acetabulum_array: np.ndarray, side: str = 'left', segmentation_label: int = 3) -> np.ndarray:
    """
    Get the anterior acetabulum rim.
    :param acetabulum_array: A 2D segmentation mask of the acetabulum
    :param side: Side of the image (not patient!), either 'left' or 'right'.
    :param segmentation_label: The label of the acetabulum in the segmentation mask.
    :return: The anterior acetabulum rim.
    """
    assert side in ['left', 'right'], 'Side must be either "left" or "right"'
    acetabulum_array = np.where(acetabulum_array == segmentation_label, 1, 0)

    p2 = np.argwhere(acetabulum_array[:, :, acetabulum_array.shape[2] // 2])
    p2 = p2[p2[:, 1].argmin()] if side == 'left' else p2[p2[:, 1].argmax()]

    return p2


def calculate_acetabular_anteversion(segmentation_mask: np.ndarray, femur_label: int = 1, acetabulum_label: int = 3, isotropic: bool = False) -> Tuple[float, float]:
    """
    Calculate the acetabular anteversion for both sides from a segmentation mask.
    :param segmentation_mask: A segmentation mask of the hip.
    :param femur_label: The label of the femur in the segmentation mask.
    :param acetabulum_label: The label of the acetabulum in the segmentation mask.
    :param isotropic: Whether the image has isotropic voxels.
    :return: The acetabular anteversion for both sides.
    """
    left_mask = segmentation_mask[:segmentation_mask.shape[0] // 2]
    right_mask = segmentation_mask[segmentation_mask.shape[0] // 2:]

    _, left_fhc = get_femoral_head_center(left_mask, side='left', segmentation_label=femur_label, isotropic=isotropic)
    _, right_fhc = get_femoral_head_center(right_mask, side='right', segmentation_label=femur_label, isotropic=isotropic)

    slice_gap = abs(int(left_fhc[2]) - int(right_fhc[2]))
    correct_slice = min(int(left_fhc[2]), int(right_fhc[2])) + slice_gap // 2

    p1_left = get_p1(left_mask[:, :, correct_slice], side='left', segmentation_label=acetabulum_label)
    p2_left = get_p2(left_mask[:, :, correct_slice], side='left', segmentation_label=acetabulum_label)
    p1_right = get_p1(right_mask[:, :, correct_slice], side='right', segmentation_label=acetabulum_label)
    p2_right = get_p2(right_mask[:, :, correct_slice], side='right', segmentation_label=acetabulum_label)

    femur_array = np.where(segmentation_mask == femur_label, 1, 0)
    left_femur = femur_array[:femur_array.shape[0] // 2]

    right_fhc_adj = right_fhc.copy()
    right_fhc_adj[0] += left_femur.shape[0]  # adjust the x coordinate of the right femoral head center to account for the splitting into left and right
    G = left_fhc[:2] - right_fhc_adj[:2]  # G is the vector connecting the left and right femoral head center

    u = left_fhc[:2]  # the point of origin of the line
    v = G  # the direction of the line
    p = p1_left  # the point the perpendicular vector goes through
    s_left = get_vector_through_point_perpendicular_to_line(u, v, p)  # s is the vector that goes through p1 and is perpendicular to G (i.e. u + lambda * G)

    u = right_fhc_adj[:2]
    p = p1_right
    s_right = get_vector_through_point_perpendicular_to_line(u, v, p)

    v1 = (p1_left - p2_left).astype('float32')
    v2 = s_left.copy()
    left_aa = calculate_angle_between_vectors(v1, v2)

    v1 = (p1_right - p2_right).astype('float32')
    v2 = s_right.copy()
    right_aa = calculate_angle_between_vectors(v1, v2)

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

    p1 = get_p1(segmentation_mask[:, :, correct_slice], side=side, segmentation_label=acetabulum_label)
    p2 = get_p2(segmentation_mask[:, :, correct_slice], side=side, segmentation_label=acetabulum_label)

    ad = get_minimum_distance_between_line_and_point(p1, p2, fhc[:2])

    return ad


def calculate_center_edge_angle(segmentation_mask: np.ndarray, femur_label: int = 1, acetabulum_label: int = 3, isotropic: bool = False) -> Tuple[float, float]:
    """
    Calculate the center edge angle for both sides from a segmentation mask.
    :param segmentation_mask: A segmentation mask of the proximal femur.
    :param femur_label: The label of the femur in the segmentation mask.
    :param acetabulum_label: The label of the acetabulum in the segmentation mask.
    :param isotropic: Whether the image has isotropic voxels.
    :return: The center edge angle for both sides.
    """
    left_mask = segmentation_mask[:segmentation_mask.shape[0] // 2]
    right_mask = segmentation_mask[segmentation_mask.shape[0] // 2:]

    _, left_fhc = get_femoral_head_center(left_mask, side='left', segmentation_label=femur_label, isotropic=isotropic)
    _, right_fhc = get_femoral_head_center(right_mask, side='right', segmentation_label=femur_label, isotropic=isotropic)

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
        tmp[:, :, most_proximal_femur_slice - 20] = 0
        tmp[:, :, most_proximal_femur_slice + 10:] = 0
        acetabulum_points = np.nonzero(tmp)
        lateral_edge = np.argmin(acetabulum_points[0]) if side == 'left' else np.argmax(acetabulum_points[0])
        lateral_edge_point = np.array(
            [acetabulum_points[0][lateral_edge], acetabulum_points[1][lateral_edge],
             acetabulum_points[2][lateral_edge]])

        return lateral_edge_point

    acetabulum_array = np.where(segmentation_mask == acetabulum_label, 1, 0)
    left_acetabulum = acetabulum_array[:acetabulum_array.shape[0] // 2]
    right_acetabulum = acetabulum_array[acetabulum_array.shape[0] // 2:]

    d = np.array([0, 0, -1])
    n = G / np.linalg.norm(G)
    d_perp = d - np.dot(d, n) * n
    s = right_fhc_adj + d_perp  # s is perpendicular to G and goes in proximal direction
    s2 = right_fhc - get_lateral_edge_point(right_femur, right_acetabulum)
    cea_right = calculate_angle_between_vectors(np.abs(s), s2)

    s = left_fhc + d_perp
    s2 = left_fhc - get_lateral_edge_point(left_femur, left_acetabulum)
    cea_left = calculate_angle_between_vectors(np.abs(s), s2)

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


def calculate_cartilage_thickness_knn(segmentation_mask: np.ndarray, cartilage_label: int = 2) -> float:
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

    return np.nanmean(distances)


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
