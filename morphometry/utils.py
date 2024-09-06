import math

import numpy as np

from typing import Tuple
from scipy.ndimage import binary_erosion


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