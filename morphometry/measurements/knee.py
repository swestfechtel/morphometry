"""Knee measurement functions: knee rotation angle and joint-line convergence angle.

The posterior-condylar reference lines are computed by ``get_knee_reference_line``
in ``morphometry.knee``; this module combines them into angles. "left"/"right"
refers to the image side, not the patient side.
"""
import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import center_of_mass

from morphometry.image_io import Segmentation
from morphometry.knee import get_knee_reference_line
from morphometry.utils import calculate_angle_between_vectors, extract_connected_components_2d
from morphometry import geometry as G

#: **The one place the knee-rotation sign convention is decided.** ``+1`` means positive =
#: external rotation of the tibia relative to the femur (the tibial condylar line turned from
#: lateral towards posterior relative to the femoral one), which is the same rotation sense as
#: positive tibial torsion in :func:`morphometry.measurements.tibia._tibial_torsion_angle` and
#: positive femoral antetorsion. **This direction is PROVISIONAL**: it was chosen for internal
#: consistency with the two torsion measurements, *not* because the available data prefer it
#: (backlog task #16; the evidence, such as it is, is summarised in CLAUDE.md, Core data flow
#: item 3). Flipping the convention is this constant plus a re-capture of the
#: ``knee_rotation_angle_left`` golden -- nothing else reads the sign, and the magnitude,
#: the rotation invariance and the mirror equivariance are all unaffected by it.
KNEE_ROTATION_POSITIVE_SIGN = +1


def _knee_rotation_angle(femur_start: np.ndarray, femur_end: np.ndarray,
                         tibia_start: np.ndarray, tibia_end: np.ndarray, side: str) -> float:
    """Compose the knee rotation angle from the two posterior condylar lines.

    The angle is the signed angle, in the axial (x, y) plane, from the posterior FEMORAL
    condylar line to the posterior TIBIAL condylar line, positive when the tibia is rotated
    externally relative to the femur (its lateral aspect turned posteriorly, equivalently its
    medial aspect turned anteriorly) and negative for internal rotation. This is the same
    rotation sense as positive (external) tibial torsion, where the distal reference turns from
    lateral towards posterior relative to the proximal condylar line
    (:func:`morphometry.geometry.signed_axial_angle`, shared with both torsions).

    Both lines are undirected, so the angle is folded into (-90, 90] and neither endpoint order
    matters. Only the two in-plane directions enter, so the result is invariant to an in-plane
    rotation of the whole knee in the scanner. In LPI a left image half holds a patient-right
    knee (medial = +x, anterior = -y); ``side='right'`` holds the mirror-image knee, so the x
    components are negated first and a knee and its mirror twin get the same angle.

    This replaces a composition that folded each line to an acute angle against the image x-axis
    and added or subtracted the two by the signs of their y-offsets. That returned
    ``sign(femoral line y-extent)`` times the signed angle above, i.e. the negated angle whenever
    the medial end of the FEMORAL condylar line lay anterior to its lateral end in the image --
    an arbitrary image-frame coordinate sign, not an anatomical one -- and, past 90 degrees, the
    supplement instead of the folded angle. It also ignored ``side``: negating both y-offsets
    leaves their relative sign unchanged, so the right-side branch cancelled out.

    **Which direction counts as positive is provisional** and is decided in exactly one place,
    :data:`KNEE_ROTATION_POSITIVE_SIGN`; see that constant and backlog task #16. Everything else
    here -- the magnitude, the rotation invariance, the mirror equivariance -- is independent of
    that choice.

    The angle is computed from voxel indices, so it assumes equal in-plane x/y spacing (as did the
    former composition; true for the Augsburg MRI and typical CT).
    :param femur_start: One posterior femoral condyle reference point.
    :param femur_end: The other posterior femoral condyle reference point.
    :param tibia_start: One posterior tibial condyle reference point.
    :param tibia_end: The other posterior tibial condyle reference point.
    :param side: Image side (not patient!), 'left' or 'right'.
    :return: The signed knee rotation angle in degrees, in (-90, 90]; with the provisional
        convention, positive = external tibial rotation relative to the femur.
    """
    femoral_line = np.asarray(femur_end, dtype=float) - np.asarray(femur_start, dtype=float)
    tibial_line = np.asarray(tibia_end, dtype=float) - np.asarray(tibia_start, dtype=float)
    return KNEE_ROTATION_POSITIVE_SIGN * G.signed_axial_angle(femoral_line, tibial_line, side)


def calculate_knee_rotation_angle(segmentation_mask: np.ndarray, femur_label: int, tibia_label: int,
                                  side: str = 'left', plot: bool | plt.Axes = False) -> float:
    """
    Calculate the knee rotation angle.

    The signed axial-plane angle from the line connecting the posterior femoral condyles to the
    line connecting the posterior tibial condyles, with the **provisional** convention positive =
    external rotation of the tibia relative to the femur (see :func:`_knee_rotation_angle` for the
    sign derivation and the convention it replaced, and :data:`KNEE_ROTATION_POSITIVE_SIGN` for the
    single place the direction is decided -- backlog task #16).
    :param segmentation_mask: A 3D segmentation mask of the knee.
    :param femur_label: The segmentation label of the femur.
    :param tibia_label: The segmentation label of the tibia.
    :param side: Side of the image (not patient!), either 'left' or 'right'. A right half holds the
        mirror-image knee, so it is composed in the mirrored convention and a knee and its mirror
        twin give the same angle.
    :param plot: A sequence of two matplotlib Axes to draw the reference lines on, or False.
    :return: The knee rotation angle in degrees, in (-90, 90].
    """
    G.validate_side(side)

    femur_mask = np.where(segmentation_mask == femur_label, 1, 0)
    proximal_layer, femur_start, femur_end = get_knee_reference_line(femur_mask, 'femur')
    femur_start, femur_end = G.order_condylar_points(femur_start, femur_end)

    tibia_mask = np.where(segmentation_mask == tibia_label, 1, 0)
    distal_layer, tibia_start, tibia_end = get_knee_reference_line(tibia_mask, 'tibia')
    tibia_start, tibia_end = G.order_condylar_points(tibia_start, tibia_end)

    angle = _knee_rotation_angle(femur_start, femur_end, tibia_start, tibia_end, side)

    if plot is not False:
        plot[0].imshow(segmentation_mask[:, :, proximal_layer].T)
        plot[0].plot([femur_start[0], femur_end[0]], [femur_start[1], femur_end[1]], color='red')
        plot[0].set_title(f'Angle: {angle:.2f}°')
        plot[1].imshow(segmentation_mask[:, :, distal_layer].T)
        plot[1].plot([tibia_start[0], tibia_end[0]], [tibia_start[1], tibia_end[1]], color='red')
    return angle


def calculate_joint_line_convergence_angle(segmentation_mask: Segmentation, femur_label: int, tibia_label: int,
                                           side: str = 'left', plot: bool | plt.Axes = False) -> float:
    """
    Calculate the joint line convergence angle (JLCA).

    Builds a femoral line between the two condyle centroids and a tibial line between
    the medial/lateral plateau apices, projects both to the coronal plane and returns
    the acute angle between them.
    :param segmentation_mask: A 3D Segmentation of the knee.
    :param femur_label: The segmentation label of the femur.
    :param tibia_label: The segmentation label of the tibia.
    :param side: Side of the image (not patient!), either 'left' or 'right'.
    :param plot: A sequence of two matplotlib Axes to draw the reference lines on, or False.
    :return: The joint line convergence angle in degrees.
    """
    # --- Femoral reference line: line between the two condyle centroids ---
    array = np.where(segmentation_mask.array == femur_label, 1, 0)

    saw_two = False
    split_layer = None
    for i in range(array.shape[2]):
        components = extract_connected_components_2d(array[:, :, i])
        if len(components) == 2:
            saw_two = True
        if saw_two and len(components) == 1:
            split_layer = i
            break
    if not (saw_two and split_layer is not None):
        raise RuntimeError('Could not find condyles!')

    components = None
    condyle_1 = None
    for i in range(array.shape[2] - 1, -1, -1):
        components = extract_connected_components_2d(array[:, :, i])
        if len(components) == 2:
            smaller = components[0] if np.count_nonzero(components[0]) < np.count_nonzero(components[1]) else components[1]
            c = center_of_mass(smaller)
            condyle_1 = np.array([c[0], c[1], i])
            break
    if condyle_1 is None:
        raise RuntimeError('Could not find condyles!')

    condyle_2 = None
    for i in range(array.shape[2] - 1, -1, -1):
        components = extract_connected_components_2d(array[:, :, i])
        if len(components) == 1:
            c = center_of_mass(components[0])
            condyle_2 = np.array([c[0], c[1], i])
            break
    if condyle_2 is None:
        raise RuntimeError('Could not find condyles!')

    proximal_reference_line = condyle_2 - condyle_1

    # --- Tibial reference line: line between medial/lateral plateau apices ---
    array = np.where(segmentation_mask.array == tibia_label, 1, 0)
    if np.count_nonzero(array) == 0:
        raise ValueError('No tibia found!')

    first_layer = next(i for i in range(array.shape[2]) if np.count_nonzero(array[:, :, i]) > 0)
    eminence = center_of_mass(array[:, :, first_layer])
    eminence = np.array([eminence[0], eminence[1], first_layer])

    points = np.argwhere(array == 1)
    leftmost = points[points[:, 0].argmin()]
    leftmost = np.array([leftmost[0], leftmost[1], first_layer])
    rightmost = points[points[:, 0].argmax()]
    rightmost = np.array([rightmost[0], rightmost[1], first_layer])

    d_1 = np.linalg.norm(leftmost - eminence)
    d_2 = np.linalg.norm(rightmost - eminence)

    # restrict to points well lateral / medial of the eminence (>2/3 of the apex distance)
    left_points = points[points[:, 0] < (eminence[0] - d_1 * 2 / 3)]
    right_points = points[points[:, 0] > (eminence[0] + d_2 * 2 / 3)]

    leftmost_point = left_points[left_points[:, 2].argmin()]
    rightmost_point = right_points[right_points[:, 2].argmin()]
    distal_reference_line = leftmost_point - rightmost_point

    # project to the coronal plane
    proximal_reference_line[1] = 0
    distal_reference_line[1] = 0

    angle = G.fold_to_acute(calculate_angle_between_vectors(proximal_reference_line, distal_reference_line))

    if plot is not False:
        femur = np.where(segmentation_mask.array == femur_label, 1, 0)
        plot[0].imshow(femur[:, (int(condyle_1[1]) + int(condyle_2[1])) // 2].T)
        plot[0].plot([condyle_1[0], condyle_2[0]], [condyle_1[2], condyle_2[2]], 'g-')
        plot[0].set_title('Femoral JLCA line')
        tibia = np.where(segmentation_mask.array == tibia_label, 1, 0)
        plot[1].imshow(tibia[:, (int(leftmost_point[1]) + int(rightmost_point[1])) // 2].T)
        plot[1].plot([leftmost_point[0], rightmost_point[0]], [leftmost_point[2], rightmost_point[2]], 'g-')
        plot[1].set_title('Tibial JLCA line')

    return angle
