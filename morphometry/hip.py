import math
import numpy as np
import pyvista as pv

from morphometry.utils import sphere_fit, get_contour_points, calc_angle_between_vectors
from scipy.ndimage import center_of_mass, binary_erosion
from scipy.spatial import KDTree
from typing import Tuple
from skimage.measure import find_contours
from sklearn.cluster import KMeans


def get_femoral_head_center(segmentation_mask: np.array) -> Tuple[float, np.array]:
    """
    Get the center of the femoral head from a segmentation mask.
    :param segmentation_mask: A segmentation mask of the proximal femur.
    :return: The radius and location of the femoral head center.
    """
    contour_pts = get_contour_points(segmentation_mask)

    # get highest layer with a mask point, its centroid
    # and lowest layer with mask point on this centroid
    layer_high = np.amin(contour_pts[:, 0]) + 1  # exclude the most proximal layer because it's tiny

    # com_high = get_centroid(mask[layer_high - 1])
    com_high = center_of_mass(segmentation_mask[layer_high + 1])
    com_high = (int(com_high[0]), int(com_high[1]))
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

    point_cloud = point_cloud[point_cloud[:, 2] >= min_z]
    point_cloud = point_cloud[point_cloud[:, 1] <= max_y]
    # need to exclude lateral parts of the mask: compute distance between com and max medial point of femoral head,
    # then exclude everything that is farther away than this distance in the lateral direction


    # get center coordinates of the fitting sphere
    r, center = sphere_fit(point_cloud)
    # compensate pixel mm ratio between x, y and z axis
    center = np.array([center[0], center[1], center[2]]).T[0]  # not sure why this is necessary, returns a column vector otherwise

    return r, center


def get_femoral_neck_center(segmentation_mask: np.array, femoral_head_center: Tuple[float, np.array]) -> Tuple[np.array, np.array]:
    """
    Get the endpoint of the femoral neck axis from a segmentation mask.
    :param segmentation_mask: A segmentation mask of the proximal femur.
    :param femoral_head_center: The radius and center of the femoral head.
    :return: The points constituting the femoral neck and the endpoint of the femoral neck axis.
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

    return neck_points, np.array([com[0], com[1], com[2]])


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


def calc_ccd(segmentation_mask: np.array) -> float:
    """
    Calculate the CCD angle from the femoral head center, femoral neck axis and femoral shaft axis.
    :param segmentation_mask: A segmentation mask of the proximal femur.
    :return: The CCD angle.
    """
    r, femoral_head_center = get_femoral_head_center(segmentation_mask)
    femoral_neck_points, femoral_neck_center = get_femoral_neck_center(segmentation_mask, (r, femoral_head_center))
    femoral_shaft_axis = get_femoral_shaft_axis(segmentation_mask)

    # Calculate the angle between the femoral neck axis and the femoral shaft axis
    neck_vector = femoral_neck_center - femoral_head_center
    shaft_vector = femoral_shaft_axis[1] - femoral_shaft_axis[0]
    ccd = calc_angle_between_vectors(neck_vector, shaft_vector)

    return ccd


def get_femoral_neck_transition(neck_points: np.array) -> np.array:
    """
    Get the point where the femoral neck transitions into the femoral head.
    :param neck_points: The points constituting the femoral neck.
    :return: The point where the femoral neck transitions into the femoral head.
    """
    most_proximal_points = neck_points[neck_points[:, 0] == np.min(neck_points[:, 0])]  # get the most proximal points
    most_proximal_lateral_point = most_proximal_points[most_proximal_points[:, 2] == np.max(most_proximal_points[:, 2])]  # of the most proximal points, get the most lateral one
    return most_proximal_lateral_point[0]  # this point is one possible transition point


def calc_alpha_angle(segmentation_mask: np.array) -> float:
    """
    Calculate the alpha angle from the femoral head center and the femoral neck transition.
    :param segmentation_mask: A segmentation mask of the proximal femur.
    :return: The alpha angle.
    """
    r, femoral_head_center = get_femoral_head_center(segmentation_mask)
    femoral_neck_points, femoral_neck_center = get_femoral_neck_center(segmentation_mask, (r, femoral_head_center))
    femoral_neck_transition = get_femoral_neck_transition(femoral_neck_points)

    # Calculate the angle between the femoral neck axis and the transition axis
    neck_vector = femoral_neck_center - femoral_head_center
    transition_vector = femoral_neck_transition - femoral_head_center
    alpha = calc_angle_between_vectors(neck_vector, transition_vector)

    return alpha


def calc_acetabular_anteversion(segmentation_mask: np.array) -> Tuple[float, float]:
    """
    Calculate the acetabular anteversion for both sides from a segmentation mask.
    :param segmentation_mask: A segmentation mask of the proximal femur.
    :return: The acetabular anteversion for both sides.
    """
    femur_array = np.where(segmentation_mask == 1, 1, 0)
    acetabulum_array = np.where(segmentation_mask == 3, 3, 0)

    left_femur = femur_array[:, :femur_array.shape[2] // 2]
    right_femur = femur_array[:, femur_array.shape[2] // 2:]

    left_acetabulum = acetabulum_array[:, :acetabulum_array.shape[2] // 2]
    right_acetabulum = acetabulum_array[:, acetabulum_array.shape[2] // 2:]

    _, left_fhc = get_femoral_head_center(left_femur)
    _, right_fhc = get_femoral_head_center(right_femur)

    slice_gap = abs(int(left_fhc[0]) - int(right_fhc[0]))
    correct_slice = min(int(left_fhc[0]), int(right_fhc[0])) + slice_gap // 2

    p1_left = np.argwhere(left_acetabulum[correct_slice])
    p1_left = p1_left[p1_left[:, 1].argmin()]  # p1 is the posterior acetabulum rim

    p1_right = np.argwhere(right_acetabulum[correct_slice])
    p1_right = p1_right[p1_right[:, 1].argmax()]  # p1 is the posterior acetabulum rim

    p2_left = np.argwhere(left_acetabulum[correct_slice][:left_acetabulum.shape[1] // 3])
    p2_left = p2_left[p2_left[:, 1].argmin()]  # p2 is the anterior acetabulum rim

    p2_right = np.argwhere(right_acetabulum[correct_slice][:right_acetabulum.shape[1] // 3])
    p2_right = p2_right[p2_right[:, 1].argmax()]  # p2 is the anterior acetabulum rim

    right_fhc_adj = right_fhc.copy()
    right_fhc_adj[2] += left_femur.shape[2]  # adjust the x coordinate of the right femoral head center to account for the splitting into left and right
    G = left_fhc[1:] - right_fhc_adj[1:]  # G is the vector connecting the left and right femoral head center

    def calc_s(u, v, p) -> np.array:
        p_ = np.dot(np.dot((p - u), v) / np.dot(v, v), v) + u
        return p - p_

    u = left_fhc[1:]
    v = G
    p = p1_left
    s_left = calc_s(u, v, p)  # s is the vector that goes through p1 and is perpendicular to G

    u = right_fhc_adj[1:]
    p = p1_right
    s_right = calc_s(u, v, p)

    v1 = (p1_left - p2_left).astype('float32')
    v2 = s_left.copy()
    left_aa = calc_angle_between_vectors(v1, v2)

    v1 = (p1_right - p2_right).astype('float32')
    v2 = s_right.copy()
    right_aa = calc_angle_between_vectors(v1, v2)

    return left_aa, right_aa