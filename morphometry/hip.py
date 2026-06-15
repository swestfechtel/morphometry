import numpy as np
import pyvista as pv
import pandas as pd
import nibabel as nib

from morphometry.utils import sphere_fit, get_contour_points, calculate_angle_between_vectors, \
    calculate_min_distance_between_point_clouds, get_vector_through_point_perpendicular_to_line, \
    get_minimum_distance_between_line_and_point, get_contour_points, num_connected_components, \
    extract_connected_components_2d, circumference_points, intersect_ndarrays, \
    sort_points_clockwise, sort_points_by_x, fit_circle_to_points
from morphometry.image_io import Image, Segmentation
from morphometry.bresenham import bresenhamline
from morphometry import constants as C
from morphometry import geometry as G

from scipy.ndimage import center_of_mass, label, rotate
from scipy.spatial import KDTree
from scipy.stats import zscore
from typing import Tuple, Optional
from skimage.measure import find_contours
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
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

def get_femoral_head_center_ct(femur_image: Segmentation, segmentation_label: int = 1, side: str = 'left') -> Tuple[float, np.ndarray]:
    """
    Get the center of the femoral head from a whole-leg CT segmentation mask.
    :param femur_image: A segmentation mask of the whole leg.
    :param segmentation_label: The label of the femur in the segmentation mask.
    :param side: Side of the image (not patient!), either 'left' or 'right'.
    :return: The radius and location of the femoral head center.
    """
    mask = np.where(femur_image.array == segmentation_label, 1, 0)
    contour_points = get_contour_points(mask)

    layer_high = np.amin(contour_points[:, 2])
    com_high = center_of_mass(mask[:, :, layer_high])
    com_high = (int(com_high[0]), com_high[1])
    com_high = (int(com_high[0]), int(com_high[1]))
    layer_low = layer_high + 1

    while mask[com_high[0], com_high[1], layer_low] != 0:
        layer_low += 1

    contour_points = contour_points[contour_points[:, 2] >= layer_high]
    contour_points = contour_points[contour_points[:, 2] <= layer_low]

    # transform everything into physical coordinates
    contour_points = np.array([femur_image.transform_index_to_physical_point(x) for x in contour_points])
    layer_high = femur_image.transform_index_to_physical_point((0, 0, layer_high))[2]
    layer_low = femur_image.transform_index_to_physical_point((0, 0, layer_low))[2]
    com_high = femur_image.transform_index_to_physical_point((com_high[0], com_high[1], 0))[:2]

    middle_slice = (layer_high + (layer_low - layer_high) // 2)
    superior_half = contour_points[contour_points[:, 2] >= middle_slice]

    if side == 'left':
        min_s = np.min(superior_half[:, 0])
        radius = com_high[0] - min_s
        max_s = com_high[0] + radius
        contour_points = contour_points[contour_points[:, 0] <= max_s]
    else:
        max_s = np.max(superior_half[:, 0])
        radius = max_s - com_high[0]
        min_s = com_high[0] - radius
        contour_points = contour_points[contour_points[:, 0] >= min_s]

    min_c = np.min(contour_points[:, 1])
    radius = com_high[1] - min_c
    max_c = com_high[1] + radius
    contour_points = contour_points[contour_points[:, 1] <= max_c]

    return sphere_fit(contour_points)

def _femoral_neck_points_and_center(point_cloud: np.ndarray, r: float, center: np.ndarray, *,
                                    distal_greater_z: bool, lateral_greater_x: bool) -> Tuple[np.ndarray, np.ndarray]:
    """Extract femoral-neck points and their centroid from a femur point cloud.

    Samples a hollow sphere shell (radius ``r`` to ``FEMORAL_NECK_SPHERE_OUTER_FACTOR
    * r``) around the femoral head centre, keeps the distal/lateral quadrant,
    removes outliers, intersects the shell with the femur point cloud (KDTree
    radius 2), and returns the resulting neck points and their centroid. Works in
    whatever coordinate space the caller supplies (index space for MRI, physical mm
    for CT); the only modality difference is which half-space is distal/lateral.
    :param point_cloud: Femur coordinates in the working space (N x 3).
    :param r: Femoral head radius in the working space.
    :param center: Femoral head centre in the working space.
    :param distal_greater_z: True if distal is larger z (MRI), False if smaller (CT).
    :param lateral_greater_x: True if lateral is larger x for this side, else False.
    :return: The femoral neck points (M x 3) and their centroid (3,).
    """
    sphere = pv.SolidSphere(inner_radius=r, outer_radius=C.FEMORAL_NECK_SPHERE_OUTER_FACTOR * r, center=center)
    shell = np.array(sphere.points)
    shell = shell[shell[:, 2] > center[2]] if distal_greater_z else shell[shell[:, 2] < center[2]]
    shell = shell[shell[:, 0] > center[0]] if lateral_greater_x else shell[shell[:, 0] < center[0]]

    com = KMeans(n_clusters=1, random_state=0).fit(shell).cluster_centers_[0]
    distances = np.linalg.norm(shell - com, axis=1)
    shell = shell[distances <= distances.mean() + C.FEMORAL_NECK_OUTLIER_STD * distances.std()]

    pc_tree = KDTree(point_cloud)
    shell_tree = KDTree(shell)
    pairs = shell_tree.query_ball_tree(pc_tree, 2)
    neck_points = np.array([point_cloud[index] for pair in pairs for index in pair])

    com = KMeans(n_clusters=1, random_state=0).fit(neck_points).cluster_centers_[0]
    return neck_points, np.array(com)


def get_femoral_neck_center(segmentation_mask: np.ndarray, femoral_head_center: Tuple[float, np.ndarray], side: str = 'left', segmentation_label: int = 1, x_ratio: float = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get the endpoint of the femoral neck axis from an MRI proximal-femur segmentation.
    :param segmentation_mask: A segmentation mask of the proximal femur where the femur is 1 and everything else 0.
    :param femoral_head_center: The radius and center of the femoral head.
    :param side: Side of the image (not patient!), either 'left' or 'right'.
    :param segmentation_label: The label of the femur in the segmentation mask.
    :param x_ratio: Correction factor for slice thickness.
    :return: The points constituting the femoral neck and the endpoint of the femoral neck axis.
    """
    G.validate_side(side)
    segmentation_mask = np.where(segmentation_mask == segmentation_label, 1, 0)

    r, center = femoral_head_center
    scale = np.array([1, 1, x_ratio])  # adjust for slice thickness
    point_cloud = np.argwhere(segmentation_mask) * scale
    center = center * scale

    neck_points, com = _femoral_neck_points_and_center(
        point_cloud, r, center, distal_greater_z=True, lateral_greater_x=(side == 'right'))

    return neck_points / scale, com / scale


def get_femoral_neck_center_ct(femur_image: Segmentation, femoral_head_center: Tuple[float, np.ndarray], segmentation_label: int = 1, side: str = 'left') -> Tuple[np.ndarray, np.ndarray]:
    """
    Get the endpoint of the femoral neck axis from a whole-leg CT segmentation mask.

    Identical to :func:`get_femoral_neck_center` but works in physical (mm) space and
    with the CT distal/lateral half-space directions (z increases superiorly and the
    sagittal sign is inverted relative to the MRI image-side convention).
    :param femur_image: A segmentation mask of the whole leg.
    :param femoral_head_center: The radius and center of the femoral head (physical mm).
    :param segmentation_label: The label of the femur in the segmentation mask.
    :param side: The side of the image (not patient!), either 'left' or 'right'.
    :return: The points constituting the femoral neck and the endpoint of the femoral neck axis.
    """
    G.validate_side(side)
    mask = np.where(femur_image.array == segmentation_label, 1, 0)
    point_cloud = np.array([femur_image.transform_index_to_physical_point(x) for x in np.argwhere(mask)])
    r, center = femoral_head_center

    neck_points, com = _femoral_neck_points_and_center(
        point_cloud, r, center, distal_greater_z=False, lateral_greater_x=(side == 'left'))

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

def get_femoral_shaft_axis_ct(femur_image: Segmentation, segmentation_label: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get the femoral shaft axis from a whole-leg CT segmentation mask using PCA.
    :param femur_image: A segmentation mask of the whole leg.
    :param segmentation_label: The label of the femur in the segmentation mask.
    :return: Start and end point of the vector representing the femoral shaft axis.
    """

    cleaned_mask = np.where(femur_image.array == segmentation_label, 1, 0)

    # Extract shaft region (exclude proximal femur and distal condyles)
    point_cloud = np.argwhere(cleaned_mask)
    point_cloud = np.array([femur_image.transform_index_to_physical_point(x) for x in point_cloud])
    z_min, z_max = np.min(point_cloud[:, 2]), np.max(point_cloud[:, 2])
    z_range = z_max - z_min

    # Keep middle 60-70% to avoid complex geometry at ends
    shaft_mask = (point_cloud[:, 2] > z_min + 0.15 * z_range) & \
                 (point_cloud[:, 2] < z_max - 0.15 * z_range)
    shaft_points = point_cloud[shaft_mask]

    # Fit PCA to find principal axis
    pca = PCA(n_components=3)
    pca.fit(shaft_points)

    # First principal component is the shaft axis direction
    centroid = pca.mean_
    direction = pca.components_[0]

    # Project extremes onto the axis to get start/end points
    projections = np.dot(shaft_points - centroid, direction)
    t_min, t_max = np.min(projections), np.max(projections)

    start_point = centroid + t_min * direction
    end_point = centroid + t_max * direction

    return start_point.astype(np.int32), end_point.astype(np.int32)

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
