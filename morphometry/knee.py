import logging

import numpy as np
from scipy.ndimage import center_of_mass
from sklearn.decomposition import PCA
from morphometry.utils import get_layer_with_biggest_convex_area, find_notch, \
    dorsal_points_rotation_angle, extract_connected_components_2d
from morphometry.image_io import Segmentation
from typing import Tuple
from matplotlib import pyplot as plt

# Warnings (a section whose intercondylar gap is not resolved) go to the 'morphometry' logger;
# attach a handler to logging.getLogger('morphometry') to collect them. A library must not
# configure logging itself.
logger = logging.getLogger(__name__)


def get_condylar_divider(mask_layer: np.ndarray) -> np.ndarray:
    """
    Separate the two condyles of an axial knee section when the intercondylar gap is not resolved.

    :func:`morphometry.utils.find_notch` locates the intercondylar notch, but the *only* thing
    :func:`get_knee_reference_line` uses it for is its sagittal coordinate: the divider that tells
    the posterior-most point of one condyle from the posterior-most point of the other (see
    :func:`morphometry.utils.dorsal_points_rotation_angle` and
    :func:`get_condylar_tangent_points`). On a section that does not resolve the
    intercondylar gap at all — a solid, roughly convex outline — there is no notch to find, and the
    section's centre of mass takes its place.

    Why that is enough, measured rather than argued: forcing this fallback on the 89 Augsburg
    reader-study tibial halves whose notch search *does* succeed moves the divider by a median -3.0
    voxels (p5 -9.6, p95 +4.6, max |delta| 11.6, i.e. up to 17% of the section width) but changes
    the tibial torsion by a median 0.00 degrees — 93% of halves within 1 degree, 98% within 3, max
    3.27 (PA000012 image-left 22.59 -> 25.86) — and all 89 still resolve. The divider only decides
    which side of the section the second posterior point is taken from, so its exact position
    barely matters on a real plateau, and the fallback is not a different measurement from the
    notch path. The centroid is mass-weighted and therefore robust to a thin protrusion (the tibial
    tuberosity, a fibular remnant) in a way an outline-extent midpoint is not.

    Nothing about the frame is inferred here: the caller's mask is assumed to be in the LPI index
    frame, where the coronal axis increases posteriorly, so "posterior" is given and no 180 degree
    ambiguity arises. That premise is shared with the notch path
    (:func:`morphometry.utils.get_dorsal_mask_point` takes the largest coronal coordinate) and it
    does **not** degrade gracefully: on a synthetic plateau turned so that its anterior-posterior
    axis is no longer the image's (past about +45 degrees, or +-90), the reported line comes out
    80-82 degrees wrong. Reorient masks to LPI before measuring.

    :param mask_layer: A 2D binary axial section of the distal femur or proximal tibia. Its
        centroid must satisfy ``3 < x < mask_layer.shape[0] - 3``, because
        :func:`morphometry.utils.dorsal_points_rotation_angle` slices the mask 3 voxels either side
        of the divider; that holds for any section of a whole bone (the centroid is interior).
    :return: The divider as a continuous ``(sagittal, coronal)`` index pair; only the sagittal
        coordinate is load-bearing.
    """
    return np.asarray(center_of_mass(mask_layer), dtype=float)


#: Two mask points count as equally posterior in the condylar frame when their coronal coordinates
#: differ by less than this (in voxels). The frame *approximately* levels the two posterior
#: cortices, so the posterior-most point of the section is a near-tie between the two condyles --
#: only near, because the frame is built from the two points
#: :func:`morphometry.utils.get_dorsal_mask_point` reports with ``knee=False``, whose sagittal
#: coordinate is ``int(mean(x))`` over the tied voxels and so need not be a mask voxel at all. The
#: window is therefore **not** merely breaking an otherwise arbitrary tie: it admits a band in which
#: the far condyle may be genuinely more posterior than the chosen start point (it is, by up to
#: 0.499 voxels, on 90 of the 200 Augsburg reader-study sections), and the start is then taken on
#: the ``+x`` side as the voxel-grid tie-break of the replaced implementation did. Widening or
#: narrowing it changes which condyle the ray is swept from.
CONDYLAR_POSTERIOR_TIE_VOXELS = 0.5

#: Two far-side mask points count as lying on the same sweep ray when their bearings from the start
#: point differ by less than this (in degrees). Only exactly collinear points are meant; the
#: tolerance absorbs the last bits of the rotation, nothing more (a tenth of a voxel at the width of
#: a knee subtends ~0.05 degrees, nine orders of magnitude above this). It has to be a tolerance
#: rather than an equality test: collinear integer coordinates stay collinear under an exact
#: rotation, but their floating-point bearings differ by ~1e-14 degrees, and resolving such a tie by
#: whichever came out smaller would reintroduce an index-frame dependence (see the tie-break in
#: :func:`get_condylar_tangent_points`).
CONDYLAR_TANGENT_TIE_DEG = 1e-9


def rotate_indices(points: np.ndarray, origin: np.ndarray, angle: float) -> np.ndarray:
    """
    Rotate 2D index coordinates about an origin, in the sense of :func:`morphometry.utils.rotate_point`.

    The vectorised equivalent of calling :func:`morphometry.utils.rotate_point` on every row, and
    the coordinate-space equivalent of :func:`morphometry.utils.rotate_mask_by_angle`: a mask point
    ``p`` of an image rotated by ``angle`` lands at ``R(angle) @ (p - c)``, with
    ``R(a) = [[cos a, sin a], [-sin a, cos a]]`` and ``c`` the array centre. Unlike the image
    rotation this is exact and loses nothing, and the result is returned relative to ``origin``
    rather than re-indexed into an array of a new shape.

    :param points: An Nx2 array of (sagittal, coronal) coordinates.
    :param origin: The centre of the rotation, as (sagittal, coronal). The result is expressed
        relative to it, so ``origin`` itself maps to ``(0, 0)``.
    :param angle: The rotation angle in degrees.
    :return: An Nx2 float array of rotated coordinates, relative to ``origin``.
    """
    radians = np.deg2rad(angle)
    cos, sin = np.cos(radians), np.sin(radians)
    rotation = np.array([[cos, sin], [-sin, cos]])
    offsets = np.asarray(points, dtype=float) - np.asarray(origin, dtype=float)
    return offsets @ rotation.T


def get_condylar_tangent_points(mask_layer: np.ndarray, divider: np.ndarray, angle: float) -> Tuple[
        np.ndarray, np.ndarray]:
    """
    Find the two endpoints of the posterior condylar tangent of an axial knee section.

    The construction is the one :func:`get_knee_reference_line` has always described: in the frame
    in which the line between the two dorsal condylar points is parallel to the sagittal axis
    (``angle``, from :func:`morphometry.utils.dorsal_points_rotation_angle`), take the posterior-most
    mask point, then sweep a ray out of it towards the other condyle until the ray grazes the mask
    on the far side of ``divider``. It is carried out **in coordinates**: the section is never
    resampled, the ray is never rasterised and the result is a pair of actual mask points, returned
    in the caller's own (unrotated) index frame.

    **It is not a strict tangent, and that is deliberate.** The ray starts along the condylar
    baseline and sweeps anteriorly, so a far-side point lying *more* posterior than the start point
    would have to be reached by sweeping the other way; the ranking below places such a point last
    (a sweep of nearly 360 degrees) rather than first, which is what the replaced implementation did
    -- its sweep also only ever turned anteriorly. Those points exist, within the half-voxel window
    of :data:`CONDYLAR_POSTERIOR_TIE_VOXELS` (see there). Measured over the 200 Augsburg
    reader-study sections: reading them as reachable instead -- a strict posterior tangent -- picks
    a different endpoint on 85 of 200 and turns the line by a median 0.598, p90 1.353, max 1.759
    degrees. Scored by whether *any* far-side mask point lies posterior of the reported line, the
    replaced implementation is "non-clean" on 112 of 200 (by at most 0.868 voxels) and this one on
    85 of 200 (at most 0.940), while the strict reading is clean on all 200 by construction. Keeping
    the anterior-only sweep therefore preserves the pre-change construction rather than silently
    adding a third semantic change to the two declared below; "tangent" in the name and in the
    prose is shorthand for that sweep, not a claim that no mask point lies behind the line.

    **The result is a function of the anatomy alone.** The rotation is taken about ``divider``,
    which moves with the anatomy, and every selection below — posterior-most, side of the divider,
    extreme bearing — is invariant under translating the frame. Since ``divider`` is an integer
    index, ``points - divider`` is exact integer arithmetic, so padding or cropping the array
    reproduces the rotated coordinates **bitwise** and shifts the reported points by exactly the
    shift of the array. **Do not "simplify" the origin to the array centre**: that is what made the
    old construction position-dependent in the first place, and a non-integer origin would make the
    invariance hold only to floating-point noise, which the exact collinear ties below sit inside of.
    The replaced implementation instead read its landmarks off a resampled,
    re-binarised, re-shaped rotated image and transformed them back through shape-derived offsets
    and a ``uint16`` cast, which made them depend on where the bone sat in the array (padding a
    section by (4, 3) voxels moved the reported endpoints by up to 51 voxels relative to the pad on
    the 200 Augsburg reader-study sections).

    **A second consequence of that read-out, and the plainest argument for replacing it: the
    reported landmark was routinely not on the bone it is defined as lying on.** The back-rotation
    plus the final ``uint16`` truncation moved the point off the posterior cortex on more than half
    of all endpoints, on sound sections as much as on the no-notch ones -- 202 of the 399 scorable
    endpoints of the Augsburg reader study ended up a voxel *behind* the posterior cortex of their
    own column, i.e. outside the mask, and one (PA000717 image-right's tibial start) landed in a
    column holding no bone voxel at all. By the repo's own cue (nearest point of the per-column
    posterior cortex, ``tests/unit/test_knee_reference_line.py::_cortex_distances``) only 195 of 400
    endpoints lay on the cortex, median 1.00 and max 2.00 voxels off. Every endpoint this function
    returns is a mask voxel, which is the property to rely on; on this cohort all 400 also land
    exactly on the posterior cortex, but that follows from its condylar frames being turned by only
    a few degrees, not from the construction -- "posterior-most" is posterior-most in the *rotated*
    frame, and at the -76 degree frames of the flexed ``dfg_knees`` femurs the reported point can
    sit 1-2 voxels anterior of its own column's cortex (measured: 2 voxels on K5_90_unbelastet, 1 on
    K9_90_unbelastet, 0 on the -34 degree K1_0_unbelastet).

    Two deliberate semantic changes come with the exact geometry (both are discussed on
    :func:`get_knee_reference_line`): the old sweep stopped only once the rasterised ray held at
    least two mask points, where an exact ray touches one, and it quantised the ray's bearing to one
    degree. Both are gone. Measured against the old implementation on the 200 Augsburg reader-study
    sections, the line's orientation moves by a median 0.567, p90 1.753 and at most 3.995 degrees,
    and its endpoints by a median 4.1 (max 13.0) voxels.

    :param mask_layer: A 2D binary axial section of the distal femur or proximal tibia.
    :param divider: The sagittal divider between the two condyles, as an integer ``(sagittal,
        coronal)`` index pair — the intercondylar notch, or the section centroid where the notch is
        not resolved (see :func:`get_condylar_divider`). Only its sagittal coordinate is
        load-bearing, as the side test and the rotation origin.
    :param angle: The rotation (in degrees) that takes the line between the two dorsal condylar
        points onto the sagittal axis, as returned by
        :func:`morphometry.utils.dorsal_points_rotation_angle`.
    :raises RuntimeError: If the section holds no mask points, or none on the far side of the
        divider (a section that lies entirely on one side of it has no second condyle to graze).
    :return: ``(start, tangent)``, the posterior-most point of the frame and the first far-side
        point the anterior sweep reaches, as float ``(sagittal, coronal)`` index pairs in
        ``mask_layer``'s own frame. Both are mask points, so they are exact voxel indices and lie on
        bone by construction.
    """
    points = np.argwhere(mask_layer > 0).astype(float)
    if points.size == 0:
        raise RuntimeError('The condylar reference section holds no mask points.')

    # The rotated frame has the divider at its origin, so "the far side of the divider" is simply
    # the sign of the rotated sagittal coordinate.
    rotated = rotate_indices(points, np.asarray(divider)[:2], angle)

    # The posterior-most point of the section. The frame only levels the two posterior cortices
    # approximately (CONDYLAR_POSTERIOR_TIE_VOXELS explains why), so this is a near-tie between the
    # two condyles and the tie-break (maximum sagittal coordinate, i.e. the +x condyle) is what
    # decides which condyle the sweep starts from -- deterministically, where the resampled-image
    # version left it to interpolation noise.
    posterior = rotated[:, 1] >= rotated[:, 1].max() - CONDYLAR_POSTERIOR_TIE_VOXELS
    tied = np.flatnonzero(posterior)
    start_index = int(tied[np.argmax(rotated[tied, 0])])
    start = rotated[start_index]

    if start[0] < 0:
        # Start point medial of the divider: the ray leaves along +x and sweeps anteriorly, which is
        # a positive rotation in rotate_point's sense (it decreases the bearing).
        far_side = rotated[:, 0] >= 0.0
        initial_bearing, sweep_sign = 0.0, 1.0
    else:
        far_side = rotated[:, 0] <= 0.0
        initial_bearing, sweep_sign = 180.0, -1.0

    far_indices = np.flatnonzero(far_side)
    offsets = rotated[far_indices] - start
    # The start point itself (and anything coincident with it) has no bearing.
    separated = np.linalg.norm(offsets, axis=1) > 0
    far_indices, offsets = far_indices[separated], offsets[separated]
    if far_indices.size == 0:
        raise RuntimeError('No mask points on the far side of the condylar divider; the section '
                           'does not hold two condyles.')

    # How far the ray has to sweep ANTERIORLY to reach each far-side point; the modulo puts a point
    # that is more posterior than the start (possible inside the half-voxel window above) at ~360
    # degrees, i.e. last, exactly as the old anterior-only sweep did. The first point reached is the
    # extreme bearing, so no step size and no rasterised hit count are involved.
    bearings = np.degrees(np.arctan2(offsets[:, 1], offsets[:, 0]))
    sweep = np.mod(sweep_sign * (initial_bearing - bearings), 360.0)
    # Among points on the same ray (a posterior cortex lying exactly along it) take the farthest.
    # This is REQUIRED for the invariance, not just a reproduction of the replaced implementation
    # (which walked the rasterised ray and kept the *last* mask point on it): exact collinear ties
    # are common on a real cortex -- 3 of the 200 Augsburg reader-study sections have two or more
    # cortex voxels on the tangent ray, e.g. PA000295 image-left's femur, where the displacements
    # from the start point are (36, 9) and (40, 10) -- and their sweep angles differ only in the
    # last bits, so an argmin over them would be settled by floating-point noise and the reported
    # point would depend on the index frame after all. Taking the farthest is well defined.
    on_first_ray = sweep <= sweep.min() + CONDYLAR_TANGENT_TIE_DEG
    candidates = np.flatnonzero(on_first_ray)
    tangent_index = int(far_indices[candidates[np.argmax(np.linalg.norm(offsets[candidates], axis=1))]])

    return points[start_index], points[tangent_index]


def get_knee_reference_line(mask: np.ndarray, bone: str, segmentation_label=1, thresh: int = 2, step_size: int = 1) -> Tuple[
    int, np.ndarray, np.ndarray]:
    """
    Get the distal reference line of a segmentation mask for calculating the femoral torsion, or the proximal
    reference line for calculating tibial torsion.

    The notch (or, where the section does not resolve the intercondylar gap, the section centroid —
    see :func:`get_condylar_divider`) is used only as the sagittal divider between the two condyles:
    the section is rotated so that the line between the posterior-most mask point on either side of
    the divider is parallel to the sagittal axis, and the reference line is then swept out of the
    posterior-most point of that frame until it grazes the mask on the far side of the divider.

    **The construction runs in coordinates, so the line is a function of the anatomy** (backlog task
    #5). The endpoints are mask points of the chosen section, found by rotating the section's point
    coordinates (:func:`get_condylar_tangent_points`) rather than by reading landmarks off a
    resampled rotated image and transforming them back; padding or cropping the array therefore
    shifts them by exactly that shift, bitwise, and changes no derived angle. Up to and including
    task #4 they were read off a ``skimage.transform.rotate`` resampling of the section, mapped back
    through offsets derived from the rotated array's *shape* and truncated to ``uint16``, all of
    which depend on where the bone happens to sit in the index grid: padding the 200 Augsburg
    reader-study sections by (4, 3) voxels moved the reported endpoints by up to 51 voxels relative
    to the pad, and left them correct on only 46 of 200. Everything upstream of the endpoints — the
    layer choice, the notch search, the divider and the rotation angle — was already exactly
    equivariant and is unchanged.

    Two **deliberate semantic changes** come with the exact geometry, both of which move values:

    * The sweep used to stop only once at least ``thresh`` (2) *rasterised* voxels of the ray lay on
      the far-side mask. An exact tangent touches one point; the two-voxel rule existed because a
      rasterised line through a single voxel is unreliable, and with exact geometry it has no
      purpose. It is gone.
    * The sweep quantised the ray's bearing to ``step_size`` (1) degree. The tangent is now exact.
      That removes a noise source, and moves values with it.

    Measured over the 200 Augsburg reader-study sections (50 patients x 2 image halves x
    femur/tibia), the two together move the line's orientation by a median 0.567, p90 1.753 and at
    most 3.995 degrees, and its endpoints by a median 4.1 (max 13.0) voxels. The orientation change
    is quoted as the difference of the two tilts folded into (-90, 90], i.e. the **undirected** angle
    between the two lines; the directed difference is identical except on the six sections whose
    endpoint order swaps, where it reads ~178 degrees (on those six the undirected change is 1.07 to
    2.76, i.e. *above* the all-200 median: when the start jumps to the other condyle the far-side
    point the sweep reaches changes with it, so the undirected line genuinely moves -- it is not the
    same line). Those swaps -- pairing each reported point with the nearer of the two old
    ones, on six of 200 each new point is nearer the *other* old one -- happen because the frame only
    nearly levels the two posterior cortices (see :data:`CONDYLAR_POSTERIOR_TIE_VOXELS`), so the
    posterior-most point of the section is a near-tie between the two condyles which the resampled
    image used to break by interpolation noise and which is now broken deterministically. The swap
    itself is invisible downstream whatever the line does: all five call sites order the pair with
    :func:`morphometry.geometry.order_condylar_points` before returning it, so neither the
    compositions nor the reported landmarks (and therefore neither ``landmarks.json`` nor the UI
    overlay) depend on which endpoint the search called ``start``.

    :param mask: A 3D segmentation mask of the femur or tibia.
    :param bone: Either 'femur' or 'tibia'.
    :param segmentation_label: The segmentation label of the femur/tibia.
    :param thresh: **Obsolete.** The minimum number of rasterised mask points the swept ray had to
        hit; the exact tangent touches one point by definition. Only the default (2) is accepted, so
        that a call site carrying a tuned value fails instead of being silently ignored.
    :param step_size: **Obsolete.** The sweep's step size in degrees; the tangent is now exact. Only
        the default (1) is accepted, for the same reason.
    :raises ValueError: If ``thresh`` or ``step_size`` is given a non-default value.
    :raises RuntimeError: If the chosen section holds no mask points on one side of the divider (see
        :func:`get_condylar_tangent_points`).
    :return: The layer and start and end points of the reference line. The two points are **float**
        ``(sagittal, coronal, layer)`` index triples now (they used to be ``uint16``); they are
        exact voxel indices of mask points, so they always lie on bone.
    """
    assert bone in ['femur', 'tibia'], 'Bone must be either "femur" or "tibia"'
    if thresh != 2 or step_size != 1:
        raise ValueError(
            'thresh and step_size are obsolete: the posterior condylar tangent is now found exactly, '
            'in coordinates, so there is no rasterised hit count to threshold and no sweep to step '
            f'(got thresh={thresh!r}, step_size={step_size!r}). Drop both arguments.')

    mask = np.where(mask == segmentation_label, 1, 0)

    layer_index = get_layer_with_biggest_convex_area(mask)
    mask_layer = mask[:, :, layer_index]  # mask_layer is 2D mask of the selected layer

    if bone == 'femur':
        percentage = 0.7
        notch_threshold = 1
    else:
        percentage = 0.5
        notch_threshold = 2

    notch = find_notch(mask_layer, percentage=percentage, thresh=notch_threshold)
    if notch is None:
        # The section does not resolve the intercondylar gap; its centroid divides the condyles.
        # This replaces a fallback that rotated the section so that its longest contour chord
        # pointed along +x -- taking the chord in its -x representative, which turned the section
        # by ~140-181 degrees and so put the points it reported as "dorsal" on the ANTERIOR cortex.
        notch = get_condylar_divider(mask_layer)
        logger.warning('No intercondylar notch resolved on the %s reference layer %d; using the section '
                       'centroid (%.1f, %.1f) as the divider between the condyles.',
                       bone, int(layer_index), float(notch[0]), float(notch[1]))
    notch = notch.astype(np.int16)

    # The frame in which the two dorsal condylar points lie on one sagittal line. Only the angle is
    # needed: the tangent search below rotates coordinates, so the section is never resampled.
    angle = dorsal_points_rotation_angle(mask_layer, notch)

    start_pt, end_pt = get_condylar_tangent_points(mask_layer, notch, angle)

    # lift the two in-plane points back into the 3D index frame of the input mask
    start_pt_orig = np.array([start_pt[0], start_pt[1], layer_index], dtype=float)
    end_pt_orig = np.array([end_pt[0], end_pt[1], layer_index], dtype=float)

    return layer_index, start_pt_orig, end_pt_orig

def get_knee_center(segmentation_mask: np.ndarray) -> np.ndarray:
    """
    Get the center point of the knee.
    :param segmentation_mask: A 3D segmentation mask of the knee, where both the femur and tibia should be labeled 1
    and everything else 0.
    :return: The center point of the knee.
    """
    return np.array(center_of_mass(segmentation_mask))
