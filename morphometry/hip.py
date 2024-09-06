import math
import numpy as np

from morphometry.utils import sphere_fit, get_contour_points
from scipy.ndimage import center_of_mass, binary_erosion
from typing import Tuple
from skimage.measure import find_contours


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


def get_femoral_neck_axis(segmentation_mask: np.array):
    """
    Get the endpoint of the femoral neck axis from a segmentation mask.
    :param segmentation_mask:
    :return:
    """
    # approximate through axis femoral head center - trochanter major for now
    point_cloud = np.argwhere(segmentation_mask)
    return min(point_cloud[:, 2])


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
