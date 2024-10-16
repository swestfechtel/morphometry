import numpy as np
import pyvista as pv

from morphometry.utils import sphere_fit, get_contour_points, calc_angle_between_vectors, \
    calc_min_distance_between_point_clouds, get_vector_through_point_perpendicular_to_line, \
    get_minimum_distance_between_line_and_point, get_contour_points
from scipy.ndimage import center_of_mass
from scipy.spatial import KDTree
from typing import Tuple, Optional
from skimage.measure import find_contours
from sklearn.cluster import KMeans


"""
All functions assume axis ordering is x = axial, y = coronal, z = sagittal.
"""


def get_femoral_head_center(segmentation_mask: np.ndarray, side: str = 'left', segmentation_label: int = 1, return_layers: bool = False, x_ratio: float = 1., isotropic: bool = False) -> Tuple[float, np.ndarray, Optional[int], Optional[int]]:
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
    layer_high = np.amin(contour_pts[:, 0]) + 1  # exclude the most proximal layer because it's tiny

    # com_high = get_centroid(mask[layer_high - 1])
    com_high = center_of_mass(segmentation_mask[layer_high + 1])
    com_high = (int(com_high[0]), int(com_high[1]))
    layer_low = layer_high
    while segmentation_mask[layer_low, com_high[0], com_high[1]] != 0:
        layer_low += 1

    """
    point_cloud = list()
    for i in range(layer_high, layer_low):
        contours = find_contours(segmentation_mask[i], 0.8)
        for contour in contours:
            for coord in contour:
                point_cloud.append([i * x_ratio, coord[0], coord[1]])
    """
    point_cloud = get_contour_points(segmentation_mask)
    point_cloud = point_cloud[point_cloud[:, 0] >= layer_high]
    point_cloud = point_cloud[point_cloud[:, 0] <= layer_low]
    point_cloud = point_cloud.astype(np.float32) * np.array([x_ratio, 1, 1])

    # need to exclude lateral parts of the mask: compute distance between com and max medial point of femoral head,
    # then exclude everything that is farther away than this distance in the lateral direction
    middle_slice = (layer_high * x_ratio + (layer_low * x_ratio - layer_high * x_ratio) // 2)
    superior_half = point_cloud[point_cloud[:, 0] <= middle_slice]
    if side == 'left':
        max_z = np.max(point_cloud[:, 2]) if isotropic else np.max(superior_half[:, 2]) * 0.9  # only look at the superior half of the femoral head
        radius = max_z - com_high[1]
        min_z = com_high[1] - radius  # the most lateral point of the femoral head
        point_cloud = point_cloud[point_cloud[:, 2] >= min_z]
    else:
        min_z = np.min(point_cloud[:, 2]) if isotropic else np.min(superior_half[:, 2]) * 0.9 # only look at the superior half of the femoral head
        radius = com_high[1] - min_z
        max_z = com_high[1] + radius  # the most lateral point of the femoral head
        point_cloud = point_cloud[point_cloud[:, 2] <= max_z]

    min_y = np.min(point_cloud[:, 1])
    radius = com_high[0] - min_y
    max_y = com_high[0] + radius
    point_cloud = point_cloud[point_cloud[:, 1] <= max_y]

    # return point_cloud

    # get center coordinates of the fitting sphere
    r, center = sphere_fit(point_cloud)
    print(center)
    # compensate pixel mm ratio between x, y and z axis
    center = np.array([center[0] / x_ratio, center[1], center[2]]).T[0]  # not sure why this is necessary, returns a column vector otherwise

    return (r, center) if not return_layers else (r, center, layer_high, layer_low)


def get_femoral_neck_center(segmentation_mask: np.ndarray, femoral_head_center: Tuple[float, np.ndarray], side: str = 'left', segmentation_label: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get the endpoint of the femoral neck axis from a segmentation mask.
    :param segmentation_mask: A segmentation mask of the proximal femur where the femur is 1 and everything else 0.
    :param femoral_head_center: The radius and center of the femoral head.
    :param side: Side of the image (not patient!), either 'left' or 'right'.
    :param segmentation_label: The label of the femur in the segmentation mask.
    :return: The points constituting the femoral neck and the endpoint of the femoral neck axis.
    """
    assert side in ['left', 'right'], 'Side must be either "left" or "right"'
    segmentation_mask = np.where(segmentation_mask == segmentation_label, 1, 0)

    r, c = femoral_head_center
    point_cloud = np.argwhere(segmentation_mask)

    # Get a sphere around the femoral head center with a radius of 1.2 times the femoral head radius
    # This sphere only includes the points between r and 1.2*r, i.e. is hollow
    solid_sphere = pv.SolidSphere(inner_radius=r, outer_radius=1.2 * r, center=c)

    # Get the points that are distal to the femoral head center
    points_i_want = np.array(solid_sphere.points)
    points_i_want = points_i_want[points_i_want[:, 0] > c[0]]

    # Get the points that are lateral to the femoral head center
    if side == 'left':
        points_i_want = points_i_want[points_i_want[:, 2] < c[2]]
    else:
        points_i_want = points_i_want[points_i_want[:, 2] > c[2]]

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


def get_femoral_shaft_axis(segmentation_mask: np.ndarray, segmentation_label: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get the femoral shaft axis from a segmentation mask.
    :param segmentation_mask: A segmentation mask of the proximal femur where the femur is 1 and everything else 0.
    :param segmentation_label: The label of the femur in the segmentation mask.
    :return: Start and end point of the vector representing the femoral shaft axis.
    """
    segmentation_mask = np.where(segmentation_mask == segmentation_label, 1, 0)
    point_cloud = np.argwhere(segmentation_mask)

    layer_low = np.max(point_cloud[:, 0]) - 1  # get the most distal layer with a mask point
    layer_high = layer_low - 20  # get a layer superior to that with some distance, distance depends on the resolution of the image

    com_low = center_of_mass(segmentation_mask[layer_low])
    com_low = (layer_low, int(com_low[0]), int(com_low[1]))
    com_high = center_of_mass(segmentation_mask[layer_high])
    com_high = (layer_high, int(com_high[0]), int(com_high[1]))  # get centers of mass of both layers

    return np.array(com_low), np.array(com_high)  # return the two points as the femoral shaft axis


def calc_ccd(segmentation_mask: np.ndarray, side: str = 'left', segmentation_label: int = 1) -> float:
    """
    Calculate the CCD angle from the femoral head center, femoral neck axis and femoral shaft axis.
    :param segmentation_mask: A segmentation mask of the proximal femur.
    :param side: Side of the image (not patient!), either 'left' or 'right'.
    :param segmentation_label: The label of the femur in the segmentation mask.
    :return: The CCD angle.
    """
    assert side in ['left', 'right'], 'Side must be either "left" or "right"'

    r, femoral_head_center = get_femoral_head_center(segmentation_mask, side=side, segmentation_label=segmentation_label)
    femoral_neck_points, femoral_neck_center = get_femoral_neck_center(segmentation_mask, (r, femoral_head_center), side=side, segmentation_label=segmentation_label)
    femoral_shaft_axis = get_femoral_shaft_axis(segmentation_mask, segmentation_label=segmentation_label)

    # Calculate the angle between the femoral neck axis and the femoral shaft axis
    neck_vector = femoral_neck_center - femoral_head_center
    shaft_vector = femoral_shaft_axis[1] - femoral_shaft_axis[0]
    ccd = calc_angle_between_vectors(neck_vector.astype('float32'), shaft_vector.astype('float32'))

    return ccd


def calc_anteversion(segmentation_mask: np.ndarray, side: str = 'left', segmentation_label: int = 1) -> float:
    """
    Calculate the anteversion of the femur.
    :param segmentation_mask: A segmentation mask of the proximal femur.
    :param side: Side of the image (not patient!), either 'left' or 'right'.
    :param segmentation_label: The label of the femur in the segmentation mask.
    :return: The anteversion angle.
    """
    assert side in ['left', 'right'], 'Side must be either "left" or "right"'

    r, femoral_head_center = get_femoral_head_center(segmentation_mask, side=side, segmentation_label=segmentation_label)
    femoral_neck_points, femoral_neck_center = get_femoral_neck_center(segmentation_mask, (r, femoral_head_center),
                                                                       side=side, segmentation_label=segmentation_label)
    horizontal_axis = np.array([0, 0, (1 if side == 'left' else -1)]).astype('float32')  # the horizontal axis in the image
    return calc_angle_between_vectors(horizontal_axis[1:], femoral_neck_center[1:] - femoral_head_center[1:])


def get_femoral_neck_transition(neck_points: np.ndarray, side: str = 'left') -> np.ndarray:
    """
    Get the point where the femoral neck transitions into the femoral head.
    :param neck_points: The points constituting the femoral neck.
    :param side: Side of the image (not patient!), either 'left' or 'right'
    :return: The point where the femoral neck transitions into the femoral head.
    """
    assert side in ['left', 'right'], 'Side must be either "left" or "right"'

    most_proximal_points = neck_points[neck_points[:, 0] == np.min(neck_points[:, 0])]  # get the most proximal points

    if side == 'left':
        most_proximal_medial_point = most_proximal_points[most_proximal_points[:, 2] == np.max(most_proximal_points[:, 2])]  # of the most proximal points, get the most medial one
    else:
        most_proximal_medial_point = most_proximal_points[most_proximal_points[:, 2] == np.min(most_proximal_points[:, 2])]  # of the most proximal points, get the most medial one

    return most_proximal_medial_point[0]  # this point is one possible transition point


def calc_alpha_angle(segmentation_mask: np.ndarray, side: str = 'left', segmentation_label: int = 1) -> float:
    """
    Calculate the alpha angle from the femoral head center and the femoral neck transition.
    :param segmentation_mask: A segmentation mask of the proximal femur.
    :param side: Side of the image (not patient!), either 'left' or 'right'.
    :param segmentation_label: The label of the femur in the segmentation mask.
    :return: The alpha angle.
    """
    assert side in ['left', 'right'], 'Side must be either "left" or "right"'

    r, femoral_head_center = get_femoral_head_center(segmentation_mask, side=side, segmentation_label=segmentation_label)
    femoral_neck_points, femoral_neck_center = get_femoral_neck_center(segmentation_mask, (r, femoral_head_center), side=side, segmentation_label=segmentation_label)
    femoral_neck_transition = get_femoral_neck_transition(femoral_neck_points, side=side)

    # Calculate the angle between the femoral neck axis and the transition axis
    neck_vector = femoral_neck_center - femoral_head_center
    transition_vector = femoral_neck_transition - femoral_head_center
    alpha = calc_angle_between_vectors(neck_vector, transition_vector)

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

    p2 = np.argwhere(acetabulum_array[:acetabulum_array.shape[0] // 2])
    p2 = p2[p2[:, 1].argmin()] if side == 'left' else p2[p2[:, 1].argmax()]

    return p2


def calc_acetabular_anteversion(segmentation_mask: np.ndarray, femur_label: int = 1, acetabulum_label: int = 3) -> Tuple[float, float]:
    """
    Calculate the acetabular anteversion for both sides from a segmentation mask.
    :param segmentation_mask: A segmentation mask of the hip.
    :param femur_label: The label of the femur in the segmentation mask.
    :param acetabulum_label: The label of the acetabulum in the segmentation mask.
    :return: The acetabular anteversion for both sides.
    """
    left_mask = segmentation_mask[:, :, :segmentation_mask.shape[2] // 2]
    right_mask = segmentation_mask[:, :, segmentation_mask.shape[2] // 2:]

    _, left_fhc = get_femoral_head_center(left_mask, side='left', segmentation_label=femur_label)
    _, right_fhc = get_femoral_head_center(right_mask, side='right', segmentation_label=femur_label)

    slice_gap = abs(int(left_fhc[0]) - int(right_fhc[0]))
    correct_slice = min(int(left_fhc[0]), int(right_fhc[0])) + slice_gap // 2

    p1_left = get_p1(left_mask[correct_slice], side='left', segmentation_label=acetabulum_label)
    p2_left = get_p2(left_mask[correct_slice], side='left', segmentation_label=acetabulum_label)
    p1_right = get_p1(right_mask[correct_slice], side='right', segmentation_label=acetabulum_label)
    p2_right = get_p2(right_mask[correct_slice], side='right', segmentation_label=acetabulum_label)

    femur_array = np.where(segmentation_mask == femur_label, 1, 0)
    left_femur = femur_array[:, :, :femur_array.shape[2] // 2]

    right_fhc_adj = right_fhc.copy()
    right_fhc_adj[2] += left_femur.shape[2]  # adjust the x coordinate of the right femoral head center to account for the splitting into left and right
    G = left_fhc[1:] - right_fhc_adj[1:]  # G is the vector connecting the left and right femoral head center

    u = left_fhc[1:]  # the point of origin of the line
    v = G  # the direction of the line
    p = p1_left  # the point the perpendicular vector goes through
    s_left = get_vector_through_point_perpendicular_to_line(u, v, p)  # s is the vector that goes through p1 and is perpendicular to G (i.e. u + lambda * G)

    u = right_fhc_adj[1:]
    p = p1_right
    s_right = get_vector_through_point_perpendicular_to_line(u, v, p)

    v1 = (p1_left - p2_left).astype('float32')
    v2 = s_left.copy()
    left_aa = calc_angle_between_vectors(v1, v2)

    v1 = (p1_right - p2_right).astype('float32')
    v2 = s_right.copy()
    right_aa = calc_angle_between_vectors(v1, v2)

    return left_aa, right_aa


def calc_acetabular_depth(segmentation_mask: np.ndarray, femur_label: int = 1, acetabulum_label: int = 3) -> Tuple[float, float]:
    """
    Get the minimum distance between the line connecting the anterior and posterior acetabulum rim and the femoral head center.
    :param segmentation_mask: A segmentation mask of the hip.
    :param femur_label: The label of the femur in the segmentation mask.
    :param acetabulum_label: The label of the acetabulum in the segmentation mask.
    :return: The acetabular depth for both sides.
    """
    left_mask = segmentation_mask[:, :, :segmentation_mask.shape[2] // 2]
    right_mask = segmentation_mask[:, :, segmentation_mask.shape[2] // 2:]

    _, left_fhc = get_femoral_head_center(left_mask, side='left', segmentation_label=femur_label)
    _, right_fhc = get_femoral_head_center(right_mask, side='right', segmentation_label=femur_label)

    slice_gap = abs(int(left_fhc[0]) - int(right_fhc[0]))
    correct_slice = min(int(left_fhc[0]), int(right_fhc[0])) + slice_gap // 2

    p1_left = get_p1(left_mask[correct_slice], side='left', segmentation_label=acetabulum_label)
    p2_left = get_p2(left_mask[correct_slice], side='left', segmentation_label=acetabulum_label)
    p1_right = get_p1(right_mask[correct_slice], side='right', segmentation_label=acetabulum_label)
    p2_right = get_p2(right_mask[correct_slice], side='right', segmentation_label=acetabulum_label)

    # right_fhc_adj = right_fhc.copy()
    # right_fhc_adj[2] += left_femur.shape[2]  # adjust the x coordinate of the right femoral head center to account for the splitting into left and right

    left_ad = get_minimum_distance_between_line_and_point(p1_left, p2_left, left_fhc[1:])
    right_ad = get_minimum_distance_between_line_and_point(p1_right, p2_right, right_fhc[1:])

    return left_ad, right_ad


def calc_center_edge_angle(segmentation_mask: np.ndarray, femur_label: int = 1, acetabulum_label: int = 3) -> Tuple[float, float]:
    """
    Calculate the center edge angle for both sides from a segmentation mask.
    :param segmentation_mask: A segmentation mask of the proximal femur.
    :param femur_label: The label of the femur in the segmentation mask.
    :param acetabulum_label: The label of the acetabulum in the segmentation mask.
    :return: The center edge angle for both sides.
    """
    left_mask = segmentation_mask[:, :, :segmentation_mask.shape[2] // 2]
    right_mask = segmentation_mask[:, :, segmentation_mask.shape[2] // 2:]

    _, left_fhc = get_femoral_head_center(left_mask, side='left', segmentation_label=femur_label)
    _, right_fhc = get_femoral_head_center(right_mask, side='right', segmentation_label=femur_label)

    femur_array = np.where(segmentation_mask == femur_label, 1, 0)
    left_femur = femur_array[:, :, :femur_array.shape[2] // 2]
    right_femur = femur_array[:, :, femur_array.shape[2] // 2:]

    right_fhc_adj = right_fhc.copy()
    right_fhc_adj[2] += left_femur.shape[
        2]  # adjust the x coordinate of the right femoral head center to account for the splitting into left and right
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

        most_proximal_femur_slice = np.min(np.nonzero(fa)[0])
        acetabulum_points = np.nonzero(aa[most_proximal_femur_slice - 10:most_proximal_femur_slice + 10])
        lateral_edge = np.argmin(acetabulum_points[2]) if side == 'left' else np.argmax(acetabulum_points[2])
        lateral_edge_point = np.array(
            [acetabulum_points[0][lateral_edge] + most_proximal_femur_slice, acetabulum_points[1][lateral_edge],
             acetabulum_points[2][lateral_edge]])

        return lateral_edge_point


    acetabulum_array = np.where(segmentation_mask == acetabulum_label, 1, 0)
    left_acetabulum = acetabulum_array[:, :, :acetabulum_array.shape[2] // 2]
    right_acetabulum = acetabulum_array[:, :, acetabulum_array.shape[2] // 2:]

    u = right_fhc
    v = G
    p = right_fhc + np.array([-1, 0, 0])
    s = get_vector_through_point_perpendicular_to_line(u, v,
                                                           p)  # s is perpendicular to G and goes in proximal direction
    s2 = right_fhc - get_lateral_edge_point(right_femur, right_acetabulum)
    cea_right = calc_angle_between_vectors(np.abs(s), s2)

    u = left_fhc
    p = left_fhc + np.array([-1, 0, 0])
    s = get_vector_through_point_perpendicular_to_line(u, v, p)
    s2 = left_fhc - get_lateral_edge_point(left_femur, left_acetabulum)
    cea_left = calc_angle_between_vectors(np.abs(s), s2)

    return cea_left, cea_right


def get_min_distance_between_femoral_head_and_acetabulum(segmentation_mask: np.ndarray, side: str = 'left', femur_label: int = 1, acetabulum_label: int = 3) -> float:
    """
    Get the minimum distance between the femoral head and the acetabulum.
    :param segmentation_mask: A segmentation mask for the hip.
    :param side: Side of the image (not patient!), either 'left' or 'right'.
    :param femur_label: The label of the femur in the segmentation mask.
    :param acetabulum_label: The label of the acetabulum in the segmentation mask.
    :return: The minimum distance between the femoral head and the acetabulum.
    """
    assert side in ['left', 'right'], 'Side must be either "left" or "right"'

    r, c = get_femoral_head_center(segmentation_mask, side=side, segmentation_label=femur_label)
    femur_mask = np.where(segmentation_mask == femur_label, 1, 0)
    point_cloud = np.argwhere(femur_mask)
    solid_sphere = pv.SolidSphere(inner_radius=0, outer_radius=1.05 * r, center=c)

    # Get the points that are distal to the femoral head center
    points_i_want = np.array(solid_sphere.points)
    points_i_want = points_i_want[points_i_want[:, 0] > c[0]]

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

    return calc_min_distance_between_point_clouds(femoral_head_points, acetabulum_points)
