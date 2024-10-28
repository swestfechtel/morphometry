import math

import numpy as np
import SimpleITK as sitk

from typing import Tuple, Optional
from scipy.ndimage import binary_erosion
from scipy.spatial import KDTree
from skimage.transform import rotate
from skimage.measure import regionprops, label
from morphometry.bresenham import bresenhamline


def sphere_fit(point_cloud: np.ndarray) -> Tuple[float, np.ndarray]:
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

    return radius, np.array([C[0], C[1], C[2]]).T[0]  # not sure why this is necessary, returns a column vector otherwise


def points_on_circle_(mask: np.ndarray, r: float, center: np.ndarray) -> bool:
    """
    Check if there are any non-zero mask points on the circumference of a circle
    with radius `r` and center `center`
    :param mask: A 2D mask.
    :param r: The radius of the circle.
    :param center: The center of the circle.
    :return: True if there are any non-zero points on the circumference of the circle, False otherwise.
    """
    rt = False
    for z in range(max(0, center[1] - int(r) - 2), min(mask.shape[1], center[1] + int(r) + 2)):  # all relevant z coordinates
        temp = r**2 - (z - center[1])**2
        if temp > 0 and (
                mask[max(0, int(round(center[0] - math.sqrt(temp)))), z] != 0
                or mask[min(int(round(center[0] + math.sqrt(temp))), mask.shape[0]-1), z] != 0):
            rt = True
            break
    if rt:
        return rt
    else:
        for y in range(max(0, center[0] - int(r) - 2), min(mask.shape[0], center[0] + int(r) + 2)):  # all relevant y coordinates
            temp = r**2 - (y - center[0])**2
            if temp > 0 and (
                    mask[y, min(0, int(round(center[1] - math.sqrt(temp))))] != 0
                    or mask[y, max(mask.shape[1]-1, int(round(center[1] + math.sqrt(temp))))] != 0):
                rt = True
                break
    return rt


def points_on_circle(mask: np.ndarray, r: float, center: np.ndarray) -> np.ndarray:
    """
    Find all points that lie on the circumference of a circle and have a value of 1 in the segmentation mask.

    :param mask: A 2D segmentation mask.
    :param center: The center of the circle (y, z).
    :param r: The radius of the circle.
    :return: An array of points (y, z) that lie on the circle's circumference and have a value of 1 in the mask.
    """
    points = []
    y_center, z_center = center[0], center[1]
    for angle in range(360):
        theta = math.radians(angle)
        y = int(round(y_center + r * math.sin(theta)))
        z = int(round(z_center + r * math.cos(theta)))
        if 0 <= y < mask.shape[0] and 0 <= z < mask.shape[1] and mask[y, z] == 1:
            points.append((y, z))

    return np.array(points) > 0


def get_contour(segmentation_mask: np.array) -> np.array:
    """
    Get the contour of a segmentation mask. Returns a binary mask where contour coordinates are set to 1.

    Example::

        >>> mask = np.array([[0, 0, 0, 0, 0],
        ...                  [0, 1, 1, 1, 0],
        ...                  [0, 1, 1, 1, 0],
        ...                  [0, 1, 1, 1, 0],
        ...                  [0, 0, 0, 0, 0]])
        >>> get_contour(mask)
        array([[0, 0, 0, 0, 0],
               [0, 1, 1, 1, 0],
               [0, 1, 0, 1, 0],
               [0, 1, 1, 1, 0],
               [0, 0, 0, 0, 0]])

    :param segmentation_mask: A segmentation mask with shape S.
    :return: A binary mask with shape S, where contour coordinates are set to 1.
    """
    eroded_mask = binary_erosion(segmentation_mask)
    return segmentation_mask - eroded_mask


def get_contour_points(segmentation_mask: np.ndarray) -> np.ndarray:
    """
    Get the contour points of a segmentation mask. Returns an array of contour points.

    Example::

        >>> mask = np.array([[0, 0, 0, 0, 0],
        ...                  [0, 1, 1, 1, 0],
        ...                  [0, 1, 1, 1, 0],
        ...                  [0, 1, 1, 1, 0],
        ...                  [0, 0, 0, 0, 0]])
        >>> get_contour_points(mask)
        array([[1, 1],
               [1, 2],
               [1, 3],
               [2, 1],
               [2, 3],
               [3, 1],
               [3, 2],
               [3, 3]])

    :param segmentation_mask: A segmentation mask.
    :return: A Nx2 or Nx3 array of contour points, where points are defined by (y, z) or (x, y, z).
    """
    return np.argwhere(get_contour(segmentation_mask))


def calculate_angle_between_vectors(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Calculate the angle (in degrees) between two vectors.
    :param v1:
    :param v2:
    :return: The angle (in degrees) between v1 and v2.
    """
    v1 /= np.linalg.norm(v1)
    v2 /= np.linalg.norm(v2)
    return math.degrees(np.arccos(np.dot(v1, v2)))


def calculate_min_distance_between_point_clouds(pc1: np.ndarray, pc2: np.ndarray) -> float:
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


def get_minimum_distance_between_line_and_point(p1: np.ndarray, p2: np.ndarray, p0: np.ndarray) -> float:
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


def get_vector_through_point_perpendicular_to_line(u: np.ndarray, v: np.ndarray, p: np.ndarray) -> np.ndarray:
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


def rotate_point(origin: np.ndarray, point: np.ndarray, angle: float, deg: bool = True) -> np.ndarray:
    """
    Rotate a point on a layer (2D) counterclockwise by a given angle around a given origin.
    Points are given as (y, z) where y is the coronal axis and z is the sagittal axis.
    :param origin: The origin of the rotation, given as (y, z).
    :param point: The point to rotate, given as (y, z).
    :param angle: The angle of rotation in degrees or radians.
    :param deg: Whether the angle is given in degrees (True) or radians (False).
    """
    if deg:
        angle = np.deg2rad(angle)

    oy, oz = origin[0], origin[1]
    py, pz = point[0], point[1]

    qz = oz + math.cos(angle) * (pz - oz) - math.sin(angle) * (-py + oy)
    qy = oy - math.sin(angle) * (pz - oz) - math.cos(angle) * (-py + oy)
    return np.array([qy, qz])


def transform_point(point: np.ndarray, origin: np.ndarray, angle: float, offset: np.ndarray = None) -> np.ndarray:
    """
    Transform a point on a layer (2D) into the rotated mask.
    Points are given as (y, z) where y is the coronal axis and z is the sagittal axis.
    :param point: The point to transform, given as (y, z).
    :param origin: The origin of the rotation, given as (y, z).
    :param angle: The angle of rotation in degrees.
    :param offset: The offset to apply to the point after rotation, given as (y, z).
    """
    new_point = rotate_point(origin, point, angle)

    if offset is not None:
        new_point = new_point + offset

    return new_point


def angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Calculate the angle in radians between vectors 'v1' and 'v2'.
    :param v1: The first vector.
    :param v2: The second vector.
    :return: The angle in radians between vectors 'v1' and 'v2'.
    """
    return math.acos(np.vdot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))


def rotate_mask_vec_parallel(mask: np.ndarray, vec1: np.ndarray, vec2: np.ndarray, return_angle: bool = True) -> Tuple[np.ndarray, Optional[float]]:
    """
    Rotate a 2D mask by the angle between two vectors so that the given vectors are parallel and return the rotated mask
    and optionally the rotation angle.
    :param mask: The 2D mask to rotate.
    :param vec1: The first vector.
    :param vec2: The second vector.
    :param return_angle: Whether to return the rotation angle.
    :return: The rotated mask and optionally the rotation angle.
    """
    angle = np.rad2deg(angle_between(vec1, vec2))

    # need to rotate clockwise or counterclockwise?
    if np.rad2deg(angle_between(rotate_point(np.array([0, 0]), vec1, angle), vec2)) != 0:
        angle = -angle

    rotated_mask = rotate(mask, angle, resize=True, preserve_range=True) > 0
    rotated_mask = rotated_mask.astype(np.uint8)

    if return_angle:
        return rotated_mask, angle
    else:
        return rotated_mask


def get_dorsal_mask_point(mask: np.ndarray) -> np.ndarray:
    """
    Return the most dorsal point of a 2D mask, i.e. the point with the greatest y value.
    :param mask: The 2D mask.
    :return: The point with the greatest y value, given as (y, z), where y is the coronal axis and z is the sagittal axis.
    """
    points = get_contour_points(mask)
    indices = np.argsort(points[:, 0])

    return np.array([points[:, 0][indices[-1]], points[:, 1][indices[-1]]])


def rotate_mask_dorsal_points(mask: np.ndarray, thresh_point: np.ndarray) -> Tuple[np.ndarray, Optional[float]]:
    """
    Rotate the mask so that a line between the most dorsal point right and left (on the sagittal axis)
    of the notch would be parallel to the sagittal axis.
    :param mask: The 2D mask.
    :param thresh_point: A threshold point, given as (y, z), where y is the coronal axis and z is the sagittal axis.
    :return: The rotated mask and optionally the rotation angle.
    """
    start = get_dorsal_mask_point(mask)
    if start[1] < thresh_point[1]:
        end = get_dorsal_mask_point(mask[:, thresh_point[1] + 3:])
        end = (end[0], end[1] + thresh_point[1])
    else:
        end = start
        start = get_dorsal_mask_point(mask[:, :thresh_point[1] - 3])

    return rotate_mask_vec_parallel(mask, np.array(end) - np.array(start), np.array([0, 1]))


def get_side_contour_points(mask: np.ndarray, y: int) -> Tuple[int, int]:
    """
    Return the smallest and largest z (i.e. sagittal) coordinate of the contour for a given y (i.e. coronal) in a 2D mask.
    :param mask: The 2D mask.
    :param y: The y coordinate.
    :return: The smallest and largest z coordinate of the contour.
    """
    nonzero = np.nonzero(mask[y])[0]
    return nonzero.min(), nonzero.max()


def determine_min_y(mask: np.ndarray, percentage: float = 0.5) -> int:
    """
    Calculate a min y (i.e. coronal) cut-off value to just have a view on the dorsal part of the mask.
    :param mask: The 2D mask.
    :param percentage: The percentage of the dorsal part to consider.
    :return: The min y value.
    """
    contour_points = get_contour_points(mask)

    num_points = int(len(contour_points[:, 0]) * percentage)  # max number of points to consider

    sorted_y = np.sort(contour_points[:, 0])
    return sorted_y[-num_points]


def calculate_discontinuity(mask: np.ndarray, y: int) -> int:
    """
    Find a discontinuity in the mask for a given y (i.e. coronal) coordinate
    :param mask: The 2D mask.
    :param y: The y coordinate.
    :return x: The z (i.e. sagittal) coordinate of the discontinuity.
    """
    side_points = get_side_contour_points(mask, y)
    return np.nonzero(1-mask[y, side_points[0]:side_points[1]])[0] + side_points[0]


def find_notch(mask: np.ndarray, min_y: int = None, percentage: float = 0.5, thresh: float = 0, break_after_first: bool = False) -> np.ndarray:
    """
    Find a notch by decreasing the y (i.e. coronal) value (up to the value of min_y)
    -> moving to ventral to find the notch.
    :param mask: The 2D mask.
    :param min_y: The minimum y value.
    :param percentage: The percentage of the dorsal part to consider.
    :param thresh: The threshold for the discontinuity.
    :param break_after_first: Whether to break after the first notch was found.
    :return: The notch.
    """
    # min_y is the threshold for the most ventral point where the notch can be
    if min_y is None:
        min_y = determine_min_y(mask, percentage)
    y, _ = get_dorsal_mask_point(mask)
    y = int(y)

    return_allowed = False
    notch = None

    # lowest mask pt >= y coord of notch >= min_y
    while y >= min_y:
        discontinuity = calculate_discontinuity(mask, y)

        # discontinuity found, return is now possible
        if isinstance(discontinuity, np.ndarray) and len(discontinuity) > thresh:
            return_allowed = True

        # no more discontinuity was found -> calculate discontinuity of
        # previous level and return its center
        else:
            if return_allowed:
                new_discontinuity = calculate_discontinuity(mask, y + 1)
                notch = np.array([y+1, new_discontinuity[int(len(new_discontinuity)/2)]])
                return_allowed = False
                if break_after_first:
                    break
        y -= 1

    if notch is None:
        raise ValueError("No notch found.")
    return notch


def get_layer_with_biggest_convex_area(mask: np.ndarray) -> int:
    """
    Get the index of the layer with the biggest convex area.
    :param mask: A 3D segmentation mask.
    :return: The index of the layer with the biggest convex area.
    """
    area = np.zeros(mask.shape[0])
    # save diameters of the layers
    for k in range(len(mask)):
        if len(np.nonzero(mask[k])[0]) != 0:
            props = regionprops(label(mask[k]))
            if props.__len__() > 1:
                i_biggest = 0
                for i in range(props.__len__()):
                    if props[i].convex_area > props[i_biggest].convex_area:
                        i_biggest = i
                area[k] = props[i_biggest].convex_area
            else:
                area[k] = props[0].convex_area

    # find index of the layer with the biggest diameter
    indices = np.argsort(area)
    return indices[-1]


def num_mask_points_on_line(mask: np.ndarray, start: np.ndarray, end: np.ndarray, thresh_pt: np.ndarray = None) -> int:
    """
    Calculate the number of mask points on the line between start and end.
    For a given thresh_pt it calculates only the mask points on the line
    with an x coordinate between the x coordinates of thresh_pt and end.
    :param mask: A 2D mask.
    :param start: The start point of the line.
    :param end: The end point of the line.
    :param thresh_pt: The threshold point.
    :return: The number of mask points on the line.
    """
    pts_on_line = bresenhamline([start], end, -1).astype(np.uint16)

    # just look at points on the other side of the threshold
    if thresh_pt is not None:
        if start[1] < thresh_pt[1]:
            pts_on_line = np.array(
                [pt for pt in pts_on_line if pt[1] >= thresh_pt[1]])
        else:
            pts_on_line = np.array(
                [pt for pt in pts_on_line if pt[1] <= thresh_pt[1]])

    sum_val = 0
    for pt in pts_on_line:
        sum_val += mask[pt[0], pt[1]]

    return sum_val


def shrink_points_to_mask(mask: np.ndarray, start_pt: np.ndarray, end_pt: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Shrink a line so that it does not go beyond the mask.
    :param mask: A 2D mask.
    :param start_pt: The start point of the line.
    :param end_pt: The end point of the line.
    :return: The start and end point of the line after shrinking.
    """

    points_on_line = bresenhamline([start_pt], end_pt, -1)

    shrinked_start, shrinked_end = None, None

    # start at one end of the line and go along until the first point of the line that is on the mask is found
    for pt in points_on_line:
        if mask[int(pt[0]), int(pt[1])]:
            shrinked_start = pt
            break

    # do the same but in opposite direction
    for pt in points_on_line[::-1]:
        if mask[int(pt[0]), int(pt[1])]:
            shrinked_end = pt
            break

    return shrinked_start, shrinked_end


def translate_image_coord_to_world_coord(image_coord: np.ndarray, reference_image: sitk.Image) -> np.ndarray:
    """
    Transform image coordinates to world coordinates.
    Can also be achieved with reference_image.TransformIndexToPhysicalPoint(image_coord).
    https://itk.org/pipermail/insight-users/2010-June/037400.html
    :param image_coord: A point in image coordinates to transform.
    :param reference_image: A reference SimpleITK image.
    :return: The transformed coordinates.
    """
    image_coord = np.array(list(reversed(image_coord)))  # because sitk and numpy ordering is flipped
    D_ = reference_image.GetDirection()
    D = np.empty((3,3))
    D[0] = np.array(D_[0:3])
    D[1] = np.array(D_[3:6])
    D[2] = np.array(D_[6:])
    S = reference_image.GetSpacing()
    O = reference_image.GetOrigin()
    S2 = np.zeros((3,3))
    np.fill_diagonal(S2, S)

    world_coordinates = D @ S2 @ image_coord.T + O
    return np.array(list(reversed(world_coordinates)))