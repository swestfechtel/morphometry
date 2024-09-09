import math
import numpy as np
import pyvista as pv

from morphometry.utils import sphere_fit, get_contour_points
from scipy.ndimage import center_of_mass, binary_erosion
from scipy.spatial import KDTree
from typing import Tuple
from skimage.measure import find_contours
from sklearn.cluster import KMeans


def get_femoral_head_center(segmentation_mask: np.array, z_ratio: float) -> Tuple[float, Tuple[float, float, float]]:
    """
    Get the center of the femoral head from a segmentation mask.
    :param segmentation_mask: A segmentation mask of the proximal femur.
    :param z_ratio: The pixel to mm ratio of the z axis.
    :return: The radius and location of the femoral head center.
    """
    contour_pts = get_contour_points(segmentation_mask)

    # get highest layer with a mask point, its centroid
    # and lowest layer with mask point on this centroid
    layer_high = np.amin(contour_pts[:, 0])

    # com_high = get_centroid(mask[layer_high - 1])
    com_high = center_of_mass(segmentation_mask[layer_high + 1])
    com_high = (int(com_high[0]), int(com_high[1]))
    print(com_high)
    layer_low = layer_high
    while segmentation_mask[layer_low, com_high[0], com_high[1]] != 0:
        layer_low += 1

    point_cloud = list()
    for i in range(layer_high, layer_low):
        contours = find_contours(segmentation_mask[i], 0.8)
        for contour in contours:
            for coord in contour:
                point_cloud.append([i, coord[0], coord[1]])

    point_cloud = np.array(point_cloud)
    max_z = np.max(point_cloud[:, 2])
    radius = max_z - com_high[1]
    min_z = com_high[1] - radius

    min_y = np.min(point_cloud[:, 1])
    radius = com_high[0] - min_y
    max_y = com_high[0] + radius

    print(min_y, max_y)

    point_cloud = point_cloud[point_cloud[:, 2] >= min_z]
    point_cloud = point_cloud[point_cloud[:, 1] <= max_y]
    # need to exclude lateral parts of the mask: compute distance between com and max medial point of femoral head,
    # then exclude everything that is farther away than this distance in the lateral direction


    # get center coordinates of the fitting sphere
    r, center = sphere_fit(point_cloud)
    # compensate pixel mm ratio between x, y and z axis
    center = (int(center[0]), int(center[1]), int(center[2]))

    return r, center


def get_femoral_neck_axis(segmentation_mask: np.array, femoral_head_center: Tuple[float, Tuple[float, float, float]]) -> Tuple[float, float, float]:
    """
    Get the endpoint of the femoral neck axis from a segmentation mask.
    :param segmentation_mask: A segmentation mask of the proximal femur.
    :param femoral_head_center: The radius and center of the femoral head.
    :return: The endpoint of the femoral neck axis.
    """

    r, c = femoral_head_center
    point_cloud = np.argwhere(segmentation_mask)
    # Get a sphere around the femoral head center with a radius of 1.2 times the femoral head radius
    # This sphere only includes the points between r and 1.2*r, i.e. is hollow
    solid_sphere = pv.SolidSphere(inner_radius=r, outer_radius=1.2 * r, center=c)
    # Get the points that are distal to the femoral head center
    points_i_want = np.array(solid_sphere.points)
    points_i_want = points_i_want[points_i_want[:, 0] > c[0]]
    # Get the points that are lateral to the femoral head center
    # TODO: handle left and right femur
    points_i_want = points_i_want[points_i_want[:, 2] < c[2]]

    # Build a KDTree for the point cloud and the points we want
    pc_tree = KDTree(point_cloud)
    sphere_tree = KDTree(points_i_want)

    # Get the intersection between the two point clouds, i.e. the points that should constitute the
    # femoral neck
    pairs = pc_tree.query_ball_tree(sphere_tree, 2)
    neck_points = list()
    for pair in pairs:
        if len(pair) > 0:
            for index in pair:
                neck_points.append(sphere_tree.data[index])

    neck_points = np.array(neck_points)

    # Get the center of mass of these points
    com = KMeans(n_clusters=1).fit(neck_points).cluster_centers_[0]
    return com[0], com[1], com[2]



def get_femoral_shaft_axis(segmentation_mask: np.array) -> Tuple[np.array, np.array]:
    """
    Get the femoral shaft axis from a segmentation mask.
    :param segmentation_mask: A segmentation mask of the proximal femur.
    :return: Start and end point of the vector representing the femoral shaft axis.
    """
    point_cloud = np.argwhere(segmentation_mask)
    layer_low = np.max(point_cloud[:, 0]) - 1
    layer_high = layer_low - 20
    com_low = center_of_mass(segmentation_mask[layer_low])
    com_low = (layer_low, int(com_low[0]), int(com_low[1]))
    com_high = center_of_mass(segmentation_mask[layer_high])
    com_high = (layer_high, int(com_high[0]), int(com_high[1]))

    return np.array(com_low), np.array(com_high)


def calc_ccd(femoral_head_center: Tuple[float, float, float], femoral_neck_axis: Tuple[float, float, float], femoral_shaft_axis: Tuple[np.array, np.array]) -> float:
    """
    Calculate the CCD angle from the femoral head center, femoral neck axis and femoral shaft axis.
    :param femoral_head_center: The center of the femoral head.
    :param femoral_neck_axis: The endpoint of the femoral neck axis.
    :param femoral_shaft_axis: Start and end point of the vector representing the femoral shaft axis.
    :return: The CCD angle.
    """
    # Calculate the angle between the femoral neck axis and the femoral shaft axis
    neck_vector = np.array(femoral_neck_axis) - np.array(femoral_head_center)
    shaft_vector = femoral_shaft_axis[1] - femoral_shaft_axis[0]
    neck_vector = neck_vector / np.linalg.norm(neck_vector)
    shaft_vector = shaft_vector / np.linalg.norm(shaft_vector)
    ccd = math.degrees(np.arccos(np.dot(neck_vector, shaft_vector)))

    return ccd
