"""Knee measurement functions: knee rotation angle and joint-line convergence angle.

The posterior-condylar reference lines are computed by ``get_knee_reference_line``
in ``morphometry.knee``; this module combines them into angles. "left"/"right"
refers to the image side, not the patient side.
"""
from typing import Optional, Sequence

import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import center_of_mass

from morphometry.image_io import Image, Segmentation, mask_and_spacing
from morphometry.knee import get_knee_reference_line
from morphometry.utils import calculate_angle_between_vectors, extract_connected_components_2d
from morphometry import geometry as G

#: **The one place the knee-rotation sign convention is decided.** ``+1`` means positive =
#: external rotation of the tibia relative to the femur (the tibial condylar line turned from
#: lateral towards posterior relative to the femoral one), which is the same rotation sense as
#: positive tibial torsion in :func:`morphometry.measurements.tibia._tibial_torsion_angle` and
#: positive femoral antetorsion, so all three agree. **This is the published convention** for
#: this measurement, which the literature calls the tibiofemoral rotation (TFR) angle -- "this
#: measurement is positive if the tibia is externally rotated compared to the femur" (Anzillotti
#: et al., J Exp Orthop 2026;13(2):e70708, following the standard technique of Diederichs et al.,
#: Am J Sports Med 2013;41:51-57) -- and the literature, not the internal consistency, is the
#: reason for it (backlog task #16, settled 2026-10-29; the evidence that could NOT settle it is
#: in CLAUDE.md, Core data flow item 3). Only the *sign* carries over from that literature, never
#: the magnitude: the published technique picks both of its slices differently (femoral line
#: through the posterior condylar CARTILAGE at the widest A-P diameter, tibial line at the first
#: slice below the menisci), and the same paper measures a 6.18 degree mean gap between CT and
#: MRI on one set of knees. Flipping the convention would still be this constant plus a
#: re-capture of the ``knee_rotation_angle_left`` golden -- nothing else reads the sign, and the
#: magnitude, the rotation invariance and the mirror equivariance are all unaffected by it.
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

    Which direction counts as positive is decided in exactly one place,
    :data:`KNEE_ROTATION_POSITIVE_SIGN`, and follows the convention this measurement is published
    under as the tibiofemoral rotation angle: positive = the tibia externally rotated relative to
    the femur. See that constant for the citation and for why only the sign, not the magnitude,
    is comparable with the published values. Everything else here -- the magnitude, the rotation
    invariance, the mirror equivariance -- is independent of that choice.

    The angle is computed from voxel indices, so it assumes equal in-plane x/y spacing (as did the
    former composition; true for the Augsburg MRI and typical CT, but **not** for the
    ``dfg_knees`` knee MRI, whose LPI in-plane spacing is 0.75 x 0.3125 mm). The precondition is
    checked by :func:`morphometry.geometry.validate_in_plane_isotropy` whenever the caller passes
    ``spacing`` to :func:`calculate_knee_rotation_angle`, which is the only level that can know it
    -- this function is handed bare landmark points.
    :param femur_start: One posterior femoral condyle reference point.
    :param femur_end: The other posterior femoral condyle reference point.
    :param tibia_start: One posterior tibial condyle reference point.
    :param tibia_end: The other posterior tibial condyle reference point.
    :param side: Image side (not patient!), 'left' or 'right'.
    :return: The signed knee rotation angle in degrees, in (-90, 90]; positive = external tibial
        rotation relative to the femur (:data:`KNEE_ROTATION_POSITIVE_SIGN`).
    """
    femoral_line = np.asarray(femur_end, dtype=float) - np.asarray(femur_start, dtype=float)
    tibial_line = np.asarray(tibia_end, dtype=float) - np.asarray(tibia_start, dtype=float)
    return KNEE_ROTATION_POSITIVE_SIGN * G.signed_axial_angle(femoral_line, tibial_line, side)


def calculate_knee_rotation_angle(segmentation_mask: Image | np.ndarray, femur_label: int, tibia_label: int,
                                  side: str = 'left', plot: bool | plt.Axes = False,
                                  spacing: Optional[Sequence[float]] = None) -> float:
    """
    Calculate the knee rotation angle.

    The signed axial-plane angle from the line connecting the posterior femoral condyles to the
    line connecting the posterior tibial condyles, positive = external rotation of the tibia
    relative to the femur, which is how the published tibiofemoral rotation (TFR) angle is signed
    (see :func:`_knee_rotation_angle` for the sign derivation and the composition it replaced, and
    :data:`KNEE_ROTATION_POSITIVE_SIGN` for the single place the direction is decided, with the
    citation and the caveat that only the sign is comparable with published values).

    **Mirror-equivariant** (backlog task #17): a RIGHT image half is measured as the reflection of
    a left one -- the mask is reversed along x, both condylar searches run on it, and the angle is
    composed there with the left-side conventions -- so a knee and its mirror image give the same
    value. The composition was already symmetric (negating the x components, which is what
    ``side='right'`` does, is algebraically the same as mirroring both lines), so the asymmetry was
    entirely in ``knee.get_knee_reference_line``: it rotates the section about the image centre and
    rounds, and is not invariant under ``x -> n - 1 - x``. Measured on the 100 Augsburg
    reader-study halves, a half and its mirror disagreed by a median 1.16 degrees, p90 3.06 and max
    6.21, exact on only 5.

    This also removes an inconsistency *within* one cohort row rather than only between the two
    sides. ``measurements/femur.py::_mri_torsion`` and
    ``measurements/tibia.py::_tibial_torsion_points`` already measured a right half as a
    reflection, so before this change a right-side row's knee rotation angle rested on a
    **different** posterior condylar line from the femoral and tibial torsion in the same row: over
    the 50 right halves the tibial line's orientation differed by a median 1.23 degrees (max 4.28,
    equal on only 8) and the femoral line's by a median 0.62 (max 2.70, equal on 10), while on left
    halves both consumers agreed exactly on all 50.
    :param segmentation_mask: A 3D segmentation mask of the knee, either as an ``Image`` /
        ``Segmentation`` (preferred: its voxel spacing is then checked against the index-space
        angle's equal-in-plane-spacing precondition without any further argument) or as a bare
        ``np.ndarray`` (then see ``spacing``).
    :param femur_label: The segmentation label of the femur.
    :param tibia_label: The segmentation label of the tibia.
    :param side: Side of the image (not patient!), either 'left' or 'right'. A right half holds the
        mirror-image knee and is measured as the reflection of a left one (see above), so a knee
        and its mirror twin give the same angle -- which before task #17 was true of the
        composition alone and not of the landmark search it rests on.
    :param plot: A sequence of two matplotlib Axes to draw the reference lines on, or False. On a
        right image half the overlay is drawn on the **reflected** mask, so the knee appears
        mirrored; the lines drawn are the ones the returned angle was composed from, which is the
        point of the overlay, and ``measurements.hip.calculate_ccd_ct`` documents the same
        consequence of the same construction.
    :param spacing: Voxel spacing of ``segmentation_mask`` (a bare array carries no geometry), e.g.
        ``knee_segmentation.spacing``. Only needed when ``segmentation_mask`` is a bare array:
        pass the ``Segmentation`` instead and the spacing is taken from it. Given either way, it is
        checked for the equal in-plane x/y spacing that the index-space angle composition assumes;
        a bare array with no ``spacing`` leaves it unchecked and the precondition then rests on the
        caller. An explicit ``spacing`` wins over an ``Image``'s own. **Worth supplying**: the
        ``dfg_knees`` knee MRI in this project's test data has 0.75 x 0.3125 mm in-plane voxels
        after the LPI reorientation, where removing the shear moves this angle by a median 3.75
        degrees (p90 14.87, and four of the 95 flexed scans land on the other side of the
        (-90, 90] fold seam). No such scan has ever been measured with it -- the shear is what
        *would* happen; see :func:`morphometry.geometry.validate_in_plane_isotropy`.
    :raises ValueError: If the resolved spacing is in-plane anisotropic or malformed (see
        :func:`morphometry.geometry.validate_in_plane_isotropy`).
    :return: The knee rotation angle in degrees, in (-90, 90].
    """
    G.validate_side(side)
    segmentation_mask, image_spacing = mask_and_spacing(segmentation_mask)
    resolved_spacing = spacing if spacing is not None else image_spacing
    G.validate_in_plane_isotropy(resolved_spacing, 'knee rotation angle segmentation')

    if side == 'right':
        # Measure the right half as the reflection of a left one, so both condylar searches run on
        # one canonical orientation -- the rule the two torsions already follow, and the reason a
        # right-side row no longer carries a different condylar line from them. Only the angle is
        # returned, so no landmark has to be mapped back; the spacing is forwarded already resolved
        # so the recursion checks the same value instead of losing it with the Image wrapper.
        return calculate_knee_rotation_angle(segmentation_mask[::-1], femur_label, tibia_label,
                                             side='left', plot=plot, spacing=resolved_spacing)

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
