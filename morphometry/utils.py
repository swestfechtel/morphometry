import math

import numpy as np

from typing import Tuple
from scipy.ndimage import binary_erosion
from scipy.spatial import KDTree


def sphere_fit(point_cloud: np.array) -> Tuple[float, Tuple[float, float, float]]:
    """
    Fit a sphere to a 3D point cloud.
    :param point_cloud: A 3D point cloud with shape (n, 3).
    :return: The radius and center of the fitted sphere.
    """
    #   Assemble the A matrix
    sp_x = point_cloud[:, 0]
    sp_y = point_cloud[:, 1]
    sp_z = point_cloud[:, 2]
    A = np.zeros((len(sp_x), 4))
    A[:, 0] = sp_x * 2
    A[:, 1] = sp_y * 2
    A[:, 2] = sp_z * 2
    A[:, 3] = 1

    #   Assemble the f matrix
    f = np.zeros((len(sp_x), 1))
    f[:, 0] = (sp_x * sp_x) + (sp_y * sp_y) + (sp_z * sp_z)
    C, residuals, rank, singval = np.linalg.lstsq(A, f)

    #   solve for the radius
    t = (C[0] * C[0]) + (C[1] * C[1]) + (C[2] * C[2]) + C[3]
    radius = math.sqrt(t)

    return radius, (C[0], C[1], C[2])


def get_contour_points(segmentation_mask: np.array) -> np.array:
    """
    Get the contour points of a segmentation mask.
    :param segmentation_mask: A segmentation mask.
    :return: The contour points.
    """
    eroded_mask = binary_erosion(segmentation_mask)
    diff_mask = segmentation_mask - eroded_mask
    contour_pts = np.argwhere(diff_mask == 1)
    return contour_pts


def calc_angle_between_vectors(v1: np.array, v2: np.array) -> float:
    """
    Calculate the angle (in degrees) between two vectors.
    :param v1:
    :param v2:
    :return: The angle (in degrees) between v1 and v2.
    """
    v1 /= np.linalg.norm(v1)
    v2 /= np.linalg.norm(v2)
    return math.degrees(np.arccos(np.dot(v1, v2)))


def calc_min_distance_between_point_clouds(pc1: np.array, pc2: np.array) -> float:
    """
    Calculate the minimum distance between two point clouds.
    :param pc1: A Nx3 array representing the first point cloud.
    :param pc2: A Nx3 array representing the second point cloud.
    :return: The minimum distance between the two point clouds.
    """
    tree = KDTree(pc1)
    distances, _ = tree.query(pc2)
    min_distance = np.min(distances)

    return min_distance


def get_minimum_distance_between_line_and_point(p1: np.array, p2: np.array, p0: np.array) -> float:
    """
    Get the minimum distance between a line that passes through two points p1 and p2 and a point p0.
    https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line
    :param p1: The first point the line passes through.
    :param p2: The second point the line passes through.
    :param p0: The point.
    :return: The minimum distance between the line and the point.
    """
    numerator = abs(
        (p2[1] - p1[1]) * p0[0] - (p2[0] - p1[0]) * p0[1] + p2[0] * p1[1] - p2[1] *
        p1[0])
    denominator = math.sqrt((p2[1] - p1[1]) ** 2 + (p2[0] - p1[0]) ** 2)
    return numerator / denominator


def get_vector_through_point_perpendicular_to_line(u: np.array, v: np.array, p: np.array) -> np.array:
    """
    Get a vector that goes through a point p and is perpendicular to the line defined by u + lambda * v.
    https://math.stackexchange.com/questions/1398634/finding-a-perpendicular-vector-from-a-line-to-a-point
    :param u: The origin vector of the line.
    :param v: The directional vector of the line.
    :param p: The point the perpendicular vector goes through.
    :return: A vector that goes through p and is perpendicular to the line defined by u + lambda * v.
    """
    p_ = np.dot(np.dot((p - u), v) / np.dot(v, v), v) + u
    return p - p_