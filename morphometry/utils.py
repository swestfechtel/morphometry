import math

import numpy as np
import nibabel as nib
import nibabel.orientations as nio

from typing import Tuple, Optional
from scipy.ndimage import binary_erosion, center_of_mass
from scipy.ndimage import label as scipy_label
from scipy.spatial import KDTree
from scipy import optimize
from skimage.transform import rotate
from skimage.measure import regionprops, label
from morphometry.bresenham import bresenhamline
from tempfile import NamedTemporaryFile


def calculate_angle_between_vectors(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Calculate the angle (in degrees) between two vectors.
    :param v1:
    :param v2:
    :return: The angle (in degrees) between v1 and v2.
    """
    v1 = v1.astype(np.float64)
    v2 = v2.astype(np.float64)
    v1 /= np.linalg.norm(v1)
    v2 /= np.linalg.norm(v2)
    return math.degrees(np.arccos(np.dot(v1, v2)))


def calculate_discontinuity(mask: np.ndarray, c: int) -> int:
    """
    Find a discontinuity in the mask for a given coronal coordinate
    :param mask: A 2D segmentation mask.
    :param c: The coronal coordinate.
    :return: The sagittal coordinates of the discontinuity.
    """
    side_points = get_side_contour_points(mask, c)
    return np.nonzero(1 - mask[side_points[0]:side_points[1], c])[0] + side_points[0]  # all sagittal coordinates where the mask is 0 between the two side points (?)


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


def circle_fit(mask: np.array) -> Tuple[np.array, float]:
    """Fit a circle to an arbitrary 2D point cloud.

    :param mask: A 2D Segmentation mask.
    :return: The center and radius of the fitted circle.
    """
    point_cloud = np.argwhere(mask != 0)
    y = point_cloud[:, 0]
    x = point_cloud[:, 1]

    y_m = np.mean(y)
    x_m = np.mean(x)

    def calc_r(xc, yc):
        return np.sqrt((x - xc) ** 2 + (y - yc) ** 2)

    def f_2(c):
        ri = calc_r(*c)
        return ri - ri.mean()

    center_estimate = y_m, x_m
    center, _ = optimize.leastsq(f_2, center_estimate)
    radius = calc_r(*center).mean()
    return np.flip(center), radius


def circumference_points(array: np.ndarray, r: float, c: tuple, tol: float = 0.5):
    """
    Find indices of array that lie on the circumference of a circle.

    Parameters:
        array: 2D numpy array (shape only used)
        r: radius of the circle
        c: center of the circle (tuple: (row, col))
        tol: tolerance for how close a point must be to the circumference

    Returns:
        indices: Nx2 numpy array of (row, col) indices
    """
    rows, cols = array.shape
    rr, cc = np.ogrid[:rows, :cols]
    dist = np.sqrt((rr - c[0]) ** 2 + (cc - c[1]) ** 2)
    mask = np.abs(dist - r) <= tol
    return np.column_stack(np.nonzero(mask))


def combine_masks(mask1: np.ndarray, mask2: np.ndarray) -> np.ndarray:
    """
    Combine two masks.
    :param mask1: The first mask.
    :param mask2: The second mask.
    :return: The combined mask.
    """
    return np.concatenate((mask1, mask2), 0)


def draw_circle(mask: np.ndarray, layer: int, r: float, center: np.ndarray, color_label=5) -> np.ndarray:
    """
    Draw a circle on a segmentation mask.
    :param mask: A 3D segmentation mask.
    :param layer: The mask layer to draw the circle on.
    :param r: The radius of the circle.
    :param center: The centre of the circle.
    :param color_label: The label to draw the circle with.
    :return: A copy of the input mask with the circle drawn on it.
    """
    mask = mask.copy()
    for s in range(center[0] - int(r) - 2,
                   center[0] + int(r) + 2):  # all relevant sagittal coordinates
        if r**2 - (s - center[0])**2 >= 0:
            c = int(round(math.sqrt(r**2 - (s - center[0])**2)))
            if c < mask.shape[1]:
                mask[s, center[1] + c, layer] = color_label
                mask[s, center[1] - c, layer] = color_label

    for c in range(center[1] - int(r) - 2,
                   center[1] + int(r) + 2):  # all relevant y coordinates
        if r**2 - (c - center[1])**2 >= 0:
            s = int(round(math.sqrt(r**2 - (c - center[1])**2)))
            if s < mask.shape[0]:
                mask[center[2] + s, c, layer] = color_label
                mask[center[2] - s, c, layer] = color_label

    return mask


def draw_line(mask: np.ndarray, layer: int, start: np.ndarray, end: np.ndarray, color_label: int = 5) -> np.ndarray:
    """
    Draw a line on a segmentation mask.
    :param mask: A 3D segmentation mask.
    :param layer: The mask layer to draw the line on.
    :param start: Start coordinates of the line.
    :param end: End coordinates of the line.
    :param color_label: The label to draw the line with.
    :return: A copy of the input mask with the line drawn on it.
    """
    mask = mask.copy()
    line = bresenhamline([start[:2]], [end[:2]], -1).astype(np.uint16)
    for u in line:
        mask[u[0], u[1], layer] = color_label

    return mask


def find_notch(mask: np.ndarray, most_ventral: int = None, percentage: float = 0.5, thresh: float = 0, break_after_first: bool = False) -> np.ndarray:
    """
    Find a notch by shifting the coronal coordinate (up to the value of most_ventral) and calculating the discontinuity.
    :param mask: The 2D mask.
    :param most_ventral: Threshold for the most ventral coordinate where the notch can be.
    :param percentage: The percentage of the dorsal part to consider.
    :param thresh: The threshold for the discontinuity.
    :param break_after_first: Whether to break after the first notch was found.
    :return: The notch (sagittal, coronal).
    """
    # most_ventral is the threshold for the most ventral (anterior) point where the notch can be
    if most_ventral is None:
        contour_points = get_contour_points(mask)

        num_points = int(len(contour_points[:, 1]) * percentage)  # max number of points to consider

        sorted_coronal = np.sort(contour_points[:, 1])
        most_ventral = sorted_coronal[-num_points]

    most_dorsal = get_dorsal_mask_point(mask)
    most_dorsal = most_dorsal[1]
    most_dorsal = int(most_dorsal)

    return_allowed = False
    notch = None

    # lowest mask pt >= dorsal coord of notch >= min_y
    while most_dorsal >= most_ventral:  # if A->P,
        discontinuity = calculate_discontinuity(mask, most_dorsal)

        # discontinuity found, return is now possible
        if isinstance(discontinuity, np.ndarray) and len(discontinuity) > thresh:
            return_allowed = True

        # no more discontinuity was found -> calculate discontinuity of
        # previous level and return its center
        else:
            if return_allowed:
                new_discontinuity = calculate_discontinuity(mask, most_dorsal + 1)
                notch = np.array([np.median(new_discontinuity), most_dorsal + 1])  # median of the discontinuity is the sagittal center of the notch
                return_allowed = False
                if break_after_first:
                    break

        most_dorsal = most_dorsal - 1  # shift in ventral (anterior) direction

    if notch is None:
        raise RuntimeError('No notch found.')

    return notch  # notch is always in form (sagittal, coronal)


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


def get_dorsal_mask_point(mask: np.ndarray, knee: bool = False) -> np.ndarray:
    """
    Return the most dorsal (posterior) point of a 2D mask, i.e. the point with the greatest coronal value.
    :param mask: The 2D mask.
    :param knee: Workaround for the knee mask, where I need to get the most dorsal point with the maximum sagittal value.
    :return: The point with the greatest coronal value, given as (x, y), where x is the sagittal coordinate and y is the coronal coordinate.
    """
    points = get_contour_points(mask)

    indices = np.argsort(points[:, 1])
    most_dorsal_index = indices[-1]  # if indices increase from ventral (anterior) to dorsal (posterior), take the last (= largest) index
    coronal_coord = points[:, 1][most_dorsal_index]
    sagittal_coord = points[:, 0][most_dorsal_index]

    most_dorsal_point_old = np.array([sagittal_coord, coronal_coord])  # make sure to always return (sagittal, coronal)

    most_dorsal_coordinate = np.max(points[:, 1])
    indices = np.nonzero(points[:, 1] == most_dorsal_coordinate)
    most_dorsal_points = points[indices]

    median_sagittal = np.median(most_dorsal_points[:, 0])
    mean_sagittal = int(np.mean(most_dorsal_points[:, 0]))
    max_sagittal = int(np.max(most_dorsal_points[:, 0]))
    most_dorsal_point = np.array([max_sagittal, most_dorsal_coordinate]) if knee else np.array([mean_sagittal, most_dorsal_coordinate])  # TODO figure out a better way for this

    return most_dorsal_point


def get_layer_with_biggest_convex_area(mask: np.ndarray) -> int:
    """
    Get the index of the layer with the biggest convex area.
    :param mask: A 3D segmentation mask.
    :return: The index of the layer with the biggest convex area.
    """
    area = np.zeros(mask.shape[2])
    # save diameters of the layers
    for i in range(mask.shape[2]):
        if len(np.nonzero(mask[:, :, i])[0]) != 0:
            props = regionprops(label(mask[:, :, i]))
            if props.__len__() > 1:
                i_biggest = 0
                for j in range(props.__len__()):
                    if props[j].convex_area > props[i_biggest].convex_area:
                        i_biggest = j
                area[i] = props[i_biggest].convex_area
            else:
                area[i] = props[0].convex_area

    # find index of the layer with the biggest diameter
    indices = np.argsort(area)
    return indices[-1]


def get_minimum_distance_between_line_and_point_(p1: np.ndarray, p2: np.ndarray, p0: np.ndarray) -> float:
    """
    Get the minimum distance between a line that passes through two points p1 and p2 and a point p0.
    Only defined for 2D points.
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


def get_minimum_distance_between_line_and_point(p1: np.ndarray, p2: np.ndarray, p0: np.ndarray) -> float:
    """
    Get the minimum distance between a line that passes through two points p1 and p2 and a point p0.
    Only defined for 2D points.
    https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line
    :param p1: The first point the line passes through.
    :param p2: The second point the line passes through.
    :param p0: The point.
    :return: The minimum distance between the line and the point.
    """
    ap = p0 - p1
    u = p2 - p1
    return np.linalg.norm(np.cross(ap, u)) / np.linalg.norm(u)


def get_point_orientation_to_vertical_line(p1: np.ndarray, p2: np.ndarray, p0: np.ndarray) -> str:
    """
    Get the orientation of a point p0 relative to a vertical line defined by p1 and p2.
    Determines whether p0 is to the left or right of the line.
    :param p1: The origin vector of the line.
    :param p2: The end vector of the line.
    :param p0: The point.
    :return: The orientation of p0 relative to the line, either 'left' or 'right'.
    """
    p1_p2 = p2 - p1
    cross = np.cross(p1_p2, p0)
    return 'left' if cross > 0 else 'right'


def get_side_contour_points(mask: np.ndarray, c: int) -> Tuple[int, int]:
    """
    Return the smallest and largest sagittal coordinate of the contour for a given coronal coordinate in a 2D mask.
    :param mask: The 2D mask.
    :param c: The coronal coordinate.
    :return: The smallest and largest sagittal coordinate of the contour.
    """
    nonzero_sagittal_coordinates = np.nonzero(mask[:, c])[0]  # returns a 1-tuple, so need to index 0 at the end
    return nonzero_sagittal_coordinates.min(), nonzero_sagittal_coordinates.max()


def get_vector_through_point_perpendicular_to_line(u: np.ndarray, v: np.ndarray, p: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get a vector that goes through a point p and is perpendicular to the line defined by u + lambda * v.
    https://math.stackexchange.com/questions/1398634/finding-a-perpendicular-vector-from-a-line-to-a-point
    :param u: The origin vector of the line.
    :param v: The directional vector of the line.
    :param p: The point the perpendicular vector goes through.
    :return: A vector that goes through p and is perpendicular to the line defined by u + lambda * v, and the projection of p onto the line (i.e. the start of the vector).
    """
    p_ = np.dot(np.dot((p - u), v) / np.dot(v, v), v) + u
    return p - p_, p_


def intersect_ndarrays(a: np.ndarray, b: np.ndarray):
    """
    Returns the intersecting elements of two n-dimensional arrays.
    For 1D arrays, returns common elements.
    For nD arrays, returns common rows (tuples of values).

    :param a: First n-dimensional array.
    :param b: Second n-dimensional array.
    """
    if a.ndim == 1 and b.ndim == 1:
        return np.intersect1d(a, b)

    a_ = a.reshape(-1, a.shape[-1])
    b_ = b.reshape(-1, b.shape[-1])
    dtype = np.dtype((np.void, a_.dtype.itemsize * a_.shape[1]))
    a_view = np.ascontiguousarray(a_).view(dtype)
    b_view = np.ascontiguousarray(b_).view(dtype)
    intersected = np.intersect1d(a_view, b_view)
    return intersected.view(a_.dtype).reshape(-1, a_.shape[1])


def num_connected_components(x: np.ndarray) -> int:
    """
    Counts the number of connected components in a 2D image.
    :param x: A 2D numpy array where each pixel is either 0 or 1.
    :return: The number of connected components in the image.
    """
    visited = np.zeros_like(x, dtype=bool)
    nrows, ncols = x.shape
    count = 0

    def neighbors(r, c):
        # 8-connectivity: include diagonals
        for dr, dc in [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < nrows and 0 <= nc < ncols:
                yield nr, nc

    for i in range(nrows):
        for j in range(ncols):
            if x[i, j] and not visited[i, j]:
                # Start BFS/DFS
                stack = [(i, j)]
                visited[i, j] = True
                while stack:
                    r, c = stack.pop()
                    for nr, nc in neighbors(r, c):
                        if x[nr, nc] and not visited[nr, nc]:
                            visited[nr, nc] = True
                            stack.append((nr, nc))
                count += 1
    return count


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
        if start[0] < thresh_pt[0]:
            pts_on_line = np.array(
                [pt for pt in pts_on_line if pt[0] >= thresh_pt[0]])
        else:
            pts_on_line = np.array(
                [pt for pt in pts_on_line if pt[0] <= thresh_pt[0]])

    sum_val = 0
    for pt in pts_on_line:
        sum_val += mask[pt[0], pt[1]]

    return sum_val


def points_on_circle(mask: np.ndarray, r: float, center: np.ndarray) -> np.ndarray:
    """
    Find all points that lie on the circumference of a circle and have a value of 1 in the segmentation mask.

    :param mask: A 2D segmentation mask.
    :param center: The center of the circle (y, z).
    :param r: The radius of the circle.
    :return: An array of points (y, z) that lie on the circle's circumference and have a value of 1 in the mask.
    """
    points = []
    c_center, s_center = center[1], center[0]
    for angle in range(360):
        theta = math.radians(angle)
        c = int(round(c_center + r * math.sin(theta)))
        s = int(round(s_center + r * math.cos(theta)))
        if 0 <= c < mask.shape[1] and 0 <= s < mask.shape[0] and mask[s, c] == 1:
            points.append((s, c))

    return np.array(points) > 0


def rotate_mask_dorsal_points(mask: np.ndarray, thresh_point: np.ndarray) -> Tuple[np.ndarray, Optional[float]]:
    """
    Rotate the mask so that a line between the most dorsal (posterior) point right and left (on the sagittal axis)
    of the notch would be parallel to the sagittal axis.
    :param mask: The 2D mask.
    :param thresh_point: A threshold point, given as (x, y), where x is the sagittal axis and y is the coronal axis.
    :return: The rotated mask and optionally the rotation angle.
    """
    start = get_dorsal_mask_point(mask)
    if start[0] < thresh_point[0]:  # start is 2D with (sagittal, coronal), thresh_point is 3D
        end = get_dorsal_mask_point(mask[thresh_point[0] + 3:])
        end = (end[0] + thresh_point[0], end[1])
    else:
        end = start
        start = get_dorsal_mask_point(mask[:thresh_point[0] - 3])

    return rotate_mask_vec_parallel(mask, np.array(end) - np.array(start), np.array([1, 0]))


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
    angle = calculate_angle_between_vectors(vec1, vec2)

    # need to rotate clockwise or counterclockwise?
    if calculate_angle_between_vectors(rotate_point(np.array([0, 0]), vec1, angle), vec2) != 0:  # TODO was != 0 before, but that failed with new code, why?
        angle = -angle

    rotated_mask = rotate(mask, -angle, resize=True, preserve_range=True) > 0  # TODO need to invert the sign now, why?
    rotated_mask = rotated_mask.astype(np.uint8)

    if return_angle:
        return rotated_mask, angle
    else:
        return rotated_mask


def rotate_point(origin: np.ndarray, point: np.ndarray, angle: float, deg: bool = True) -> np.ndarray:
    """
    Rotate a point on a layer (2D) counterclockwise by a given angle around a given origin.
    Points are given as (x, y) where x is the sagittal axis and y is the coronal axis.
    :param origin: The origin of the rotation, given as (sagittal, coronal).
    :param point: The point to rotate, given as (sagittal, coronal).
    :param angle: The angle of rotation in degrees or radians.
    :param deg: Whether the angle is given in degrees (True) or radians (False).
    """
    if deg:
        angle = np.deg2rad(angle)

    oc, os = origin[1], origin[0]
    pc, ps = point[1], point[0]

    qs = os + math.cos(angle) * (ps - os) - math.sin(angle) * (-pc + oc)
    qc = oc - math.sin(angle) * (ps - os) - math.cos(angle) * (-pc + oc)
    return np.array([qs, qc])


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


def split_masks(segmentation_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split a segmentation mask into two separate masks, one for the left and one for the right bone.
    :param segmentation_mask: A 3D segmentation mask.
    :return:
    """
    # Step 1: Label connected components
    labels, num_features = scipy_label(segmentation_mask)

    # Step 2: Compute properties of each component
    components = []
    for i in range(1, num_features + 1):
        # Compute volume
        volume = np.sum(labels == i)
        # Compute centroid
        centroid = center_of_mass(segmentation_mask, labels, i)
        components.append({'label': i, 'volume': volume, 'centroid': centroid})

    # Step 3: Identify the two largest components
    components.sort(key=lambda x: x['volume'], reverse=True)
    bone_1 = components[0]
    bone_2 = components[1]

    # Step 4: Determine which bone is left and which is right
    # Adjust the index [2] if the left-right axis is not the first axis
    if bone_1['centroid'][2] < bone_2['centroid'][2]:
        left_label = bone_1['label']
        right_label = bone_2['label']
    else:
        left_label = bone_2['label']
        right_label = bone_1['label']

    # Step 5: Create separate masks
    left_mask = (labels == left_label)
    right_mask = (labels == right_label)

    return left_mask.astype(np.uint8), right_mask.astype(np.uint8)


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


def extract_connected_components_2d(mask: np.ndarray) -> list:
    """
    Extract all connected components from a 2D binary segmentation mask.

    :param mask: 2D numpy array (binary mask)
    :return: List of 2D numpy arrays (binary masks), one per connected component
    """
    labeled, num = label(mask, connectivity=1, return_num=True)
    components = []
    for i in range(1, num + 1):
        comp_mask = (labeled == i).astype(np.uint8)
        components.append(comp_mask)
    return components