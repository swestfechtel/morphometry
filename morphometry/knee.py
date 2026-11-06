import logging
import math

import numpy as np
from scipy.ndimage import center_of_mass
from sklearn.decomposition import PCA
from morphometry import constants as C
from morphometry.utils import get_layer_with_biggest_convex_area, find_notch, \
    dorsal_points_rotation_angle, extract_connected_components_2d
from morphometry.geometry import rotate_indices, validate_in_plane_isotropy
from morphometry.image_io import Segmentation, mask_and_spacing
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


#: Adjectival flag prefix per bone, so that the two condylar lines of one leg stay distinguishable
#: in the single ``qc_flags`` cell they share (``femoral_condylar_line_short`` against
#: ``tibial_condylar_line_short``). It doubles as the set of bones
#: :func:`knee_condylar_line_qc` accepts, which keeps the accepted names and the flag names from
#: drifting apart. The adjectives match the package's own naming (``femoral_torsion``,
#: ``tibial_torsion``), not :func:`get_knee_reference_line`'s noun ``bone`` argument.
_BONE_FLAG_PREFIX = {'femur': 'femoral', 'tibia': 'tibial'}


def knee_condylar_line_qc(start_point: np.ndarray, end_point: np.ndarray,
                          spacing: Tuple[float, ...] | None = None, bone: str = 'femur') -> dict:
    """
    Describe how trustworthy one posterior condylar reference line is, without changing it.

    :func:`get_knee_reference_line` returns two posterior-condylar endpoints whose in-plane
    **direction** is the entire anatomical content of three measurements -- the femoral torsion
    (``measurements.femur.calculate_femoral_torsion``), the tibial torsion
    (``measurements.tibia.calculate_tibial_torsion``) and the knee rotation angle
    (``measurements.knee.calculate_knee_rotation_angle``). Nothing else about the pair is read:
    the endpoint order is normalised away by :func:`morphometry.geometry.order_condylar_points`,
    the layer only says where the section was taken, and each composition folds the undirected
    line's bearing into a signed axial angle. A **short** line is therefore badly conditioned in
    the one quantity that is used, and the angle it produces looks perfectly ordinary: one voxel
    of landmark error turns a 9.5-voxel line by 6.0 degrees and a 44-voxel line by 1.3.

    This function reports that conditioning so a cohort run can mark the row. **Nothing is
    repaired and nothing is refused**: a flagged leg keeps its torsion and its knee rotation
    angle, and the endpoints returned by the search are the ones the measurement used. The one
    flag, in the ``name:value`` / ``;``-joined format of ``scripts/process_nmdid.py``'s
    ``qc_flags`` column:

    * ``femoral_condylar_line_short:<voxels>`` / ``tibial_condylar_line_short:<voxels>`` -- the
      in-plane length of the line is below ``1 / tan(C.CONDYLAR_LINE_ANGLE_TOLERANCE_DEG)``
      = 14.30 in-plane voxels, i.e. one in-plane voxel of landmark error would tilt it by more than
      that tolerance. See :data:`morphometry.constants.CONDYLAR_LINE_ANGLE_TOLERANCE_DEG` for the
      derivation, the Augsburg prevalence survey, and the disclosure of what the number does and
      does not rest on (in short: the construction is inherited from ``murphy_axis_short``, the
      *evidence* for 4 degrees is not, and on the tibia the threshold falls in a continuum where no
      value is distinguished). ``<voxels>`` is the tested quantity itself, rounded to two decimals
      while the comparison is not -- read it as a label, not as the number the gate saw.

    **The value is in in-plane voxels, and this deliberately departs from ``murphy_axis_short``,
    which prints millimetres.** The departure is not a style preference; it is forced by a measured
    property of the cohort. The verdict is a voxel count, because the threshold means "one voxel of
    landmark error", while the Augsburg large cohort carries four in-plane spacings (1.09375,
    1.145833, 1.302083, 1.640625 mm) -- so in millimetres the printed value would **not be monotone
    in the verdict**: 51 of its *unflagged* lines are shorter in millimetres (16.41 to 21.24 mm)
    than its longest *flagged* one (``PA000435`` image-right tibia, 13.04 voxels = 21.39 mm at
    1.64 mm spacing), and all 51 are longer in voxels. A ``qc_flags`` column sorted on that number
    would interleave flagged and sound lines, and a flagged ``21.39`` read against "the threshold is
    15.6 mm on Augsburg" would look like the gate misfiring. In voxels every flagged value is below
    14.30 and every unflagged one at or above it, on every cohort, with no spacing needed to
    interpret it. ``murphy_axis_short`` has the same exposure in principle but far less of it in
    practice: it prints millimetres and this gate does not, so **the two numbers in one
    ``qc_flags`` cell are in different units** -- which is the cost of this choice, paid once in
    documentation rather than once per reader of a sorted column. ``line_length_mm`` is in the
    returned dict for anyone who wants the physical figure.

    **The flag name carries the bone, and that is load-bearing rather than decorative.** One leg
    has two condylar lines, on two different sections of two different bones, and both land in the
    *same* ``qc_flags`` cell (:func:`get_condylar_line_qc` is called once per bone). An unprefixed
    ``condylar_line_short`` would therefore be ambiguous exactly where it matters, and the prefix
    also says which measurements to distrust: the femoral line feeds the femoral torsion and the
    knee rotation angle, the tibial line the tibial torsion and the knee rotation angle. The
    threshold itself is **not** per bone -- the two length distributions differ substantially
    (Augsburg median 45.10 voxels femoral against 29.65 tibial) but a landmark error of one voxel is
    the same error on both, so a per-bone threshold would be fitting to a distribution, which is
    what this gate's construction exists to avoid.

    **A count of in-plane voxels is only a length where the two in-plane voxel edges agree**, which
    is why a known ``spacing`` is validated here rather than trusted (backlog task #10). On the
    ``dfg_knees`` knee MRI, 0.75 x 0.3125 mm in plane, a "voxel" is 2.4 times longer one way than
    the other and the same line would be flagged or not depending on how the knee happens to lie in
    the slice -- so no rate from that cohort is quoted for this gate anywhere, the comparison being
    void on units. It is the same precondition the three consuming angles already enforce on
    themselves, so a dataset this gate refuses is one on which none of them returns a value either.

    **Three things that are deliberately NOT flagged**, recorded so the absences are not read as
    oversights:

    * *The no-notch centroid fallback.* It fires on 15.8% of Augsburg tibial sections, and a gate
      at that rate would mark a sixth of the cohort. It would also be marking something measured to
      be nearly harmless: forcing the fallback on the 89 Augsburg tibial halves whose notch search
      succeeds moves the tibial torsion by a median 0.00 degrees (93% within 1 degree, max 3.27).
      The fallback is a divider choice, not a frame flip, since the 180-degree ``rotate_tibia``
      fallback was removed in task #4.
    * *A section on reference layer 0*, i.e. at the edge of the field of view. This is the
      condition that produced the motivating leg (PA000877's femoral search), but it occurs on
      exactly 2 of 1904 Augsburg femoral halves -- both halves of that one patient -- and a gate
      cannot be justified on its single motivating case. The short-line gate catches the half whose
      line is actually degenerate (the image-right one, **5.39** voxels, which is what the flag
      prints -- its unreflected length of 9.49 is a different reading, see
      :func:`get_condylar_line_qc`) and leaves the other alone at 41.44,
      which is the verdict the evidence supports. ``layer`` is reported in the dict so a caller can
      test it, and a layer-at-the-FOV-edge gate needs a prevalence survey of its own first.
    * *Whether the two endpoints sit on opposite condyles.* They are a near-tie away from each
      other by construction (see :data:`CONDYLAR_POSTERIOR_TIE_VOXELS`), and this gate's whole
      point is the length, which is low precisely when they do not separate.

    :param start_point: The first endpoint of the reference line, as returned by
        :func:`get_knee_reference_line` -- an index triple ``(sagittal, coronal, layer)``. The
        order of the two endpoints does not matter: every quantity here is symmetric in them.
    :param end_point: The second endpoint, same frame.
    :param spacing: The voxel spacing of the image the line was found in, as ``(sx, sy, sz)``, or
        ``None``. ``None`` means the caller holds no geometry (the condylar searches historically
        take bare arrays): ``line_length_mm`` is then ``None`` and the in-plane isotropy
        precondition is **not** checked, resting on that caller instead -- the trust model of
        :func:`morphometry.femur.lee_neck_axis_qc`. The **flag string is byte-identical either
        way**, since it prints voxels: an ungeometried call loses the precondition and the
        millimetre figure, never the verdict or its units. (An earlier draft printed millimetres
        when a spacing was given and voxels when not, i.e. two unit systems under one flag key
        selected by an optional argument; printing voxels removes that without making ``spacing``
        required.) Prefer passing the spacing all the same;
        :func:`get_condylar_line_qc` does whenever its caller has one.
    :param bone: ``'femur'`` or ``'tibia'`` -- which bone's line this is. It names the flag
        (``femoral_`` / ``tibial_``, see above) and the image in the isotropy error message, and is
        reported in the dict; it changes no threshold.
    :return: ``{'bone', 'layer', 'line_length_voxels', 'line_length_mm', 'angle_per_voxel_deg',
        'flags'}``, where ``angle_per_voxel_deg`` is how far one in-plane voxel of landmark error
        would tilt the line (``atan2(1, length in voxels)``, the quantity the threshold is derived
        from) and ``flags`` is the ``;``-joined flag string, ``''`` when the line looks sound.
        Every entry is invariant under the left-right mirror of a *given* line, the length and the
        layer being preserved by a reflection; the *search* that produces the line is only
        approximately mirror-equivariant, which is why :func:`get_condylar_line_qc` reflects a
        right half exactly as the measurements do instead of relying on this.
    :raises ValueError: If ``bone`` is not ``'femur'`` or ``'tibia'``, if the two endpoints are not
        readable index triples, or if a given ``spacing`` has unequal in-plane entries (see
        :func:`morphometry.geometry.validate_in_plane_isotropy`). The gate never raises on a
        degenerate *line*: a line of length zero is flagged, not refused, because it is still the
        line the measurement used.
    """
    if bone not in _BONE_FLAG_PREFIX:
        raise ValueError(f"bone must be either 'femur' or 'tibia', got {bone!r}.")
    validate_in_plane_isotropy(spacing, f'knee image ({bone} condylar-line QC)')

    try:
        start = np.asarray(start_point, dtype=float).reshape(-1)
        end = np.asarray(end_point, dtype=float).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise ValueError('The condylar reference line endpoints must be numeric index triples, '
                         f'got {start_point!r} and {end_point!r}.') from exc
    if start.size < 3 or end.size < 3:
        raise ValueError('The condylar reference line endpoints must be (sagittal, coronal, layer) '
                         f'index triples, got sizes {start.size} and {end.size}.')
    if not (np.all(np.isfinite(start[:3])) and np.all(np.isfinite(end[:3]))):
        raise ValueError('The condylar reference line endpoints must be finite, got '
                         f'{start[:3]!r} and {end[:3]!r}.')

    # in-plane voxels: equal x/y spacing is guaranteed above wherever the spacing is known, so this
    # is a length in voxel edges. Only the in-plane direction of the line enters the three angles,
    # so the slice coordinate is deliberately not part of the length -- the two endpoints come from
    # one section and share it anyway.
    length_voxels = float(np.hypot(start[0] - end[0], start[1] - end[1]))
    metrics = {
        'bone': bone,
        'layer': int(round(float(start[2]))),
        'line_length_voxels': length_voxels,
        'line_length_mm': None if spacing is None else length_voxels * float(abs(spacing[0])),
        # how far one in-plane voxel of landmark error turns the line; the threshold below is this
        # quantity read backwards
        'angle_per_voxel_deg': float(np.degrees(np.arctan2(1.0, length_voxels))),
    }

    flags = []
    if length_voxels < 1.0 / math.tan(math.radians(C.CONDYLAR_LINE_ANGLE_TOLERANCE_DEG)):
        # in-plane VOXELS, the quantity the comparison is on -- see the docstring for why this is
        # not the millimetre figure that `murphy_axis_short` prints
        flags.append(f'{_BONE_FLAG_PREFIX[bone]}_condylar_line_short:{length_voxels:.2f}')
    metrics['flags'] = ';'.join(flags)
    return metrics


#: Reason codes for ``<bone>_condylar_landmark_failed``, matched against the exception message in
#: order. Deliberately a separate table from
#: ``morphometry.femur._LANDMARK_FAILURE_REASONS``: sharing it would mean ``knee.py`` importing
#: ``femur.py``, a region-to-region dependency the package layering does not have (see
#: ``tests/test_import_graph.py`` and the note on ``geometry.rotate_indices``), and the two tables
#: have no marker in common -- the Murphy / Lee table describes proximal-femur searches, this one
#: the condylar section. ``anisotropic_in_plane`` is spelled exactly as it is there, so one
#: ``qc_flags`` cell does not carry two names for the same refusal.
#:
#: **Reachability, measured rather than assumed, because two of these codes read as though they
#: covered more than they do.** An *empty* bone label does not reach either of the first two: the
#: layer walk returns a layer regardless, and ``utils.find_notch`` then fails on the empty section
#: with a bare ``IndexError`` whose message ("index 0 is out of bounds...") is too generic to match
#: on, so it arrives as ``<bone>_condylar_landmark_failed:IndexError`` via the type-name fallback.
#: That is the honest outcome and it is deliberately not given a named code by matching on an
#: index-error message. ``condylar_section_empty`` therefore covers only the ``RuntimeError`` the
#: *tangent* search raises, which needs a section that passed the notch search first.
#: ``obsolete_argument`` is unreachable from :func:`get_condylar_line_qc`, which never passes
#: ``thresh`` or ``step_size``; it is here so that a future caller which does gets a named flag
#: rather than a type name.
_CONDYLAR_FAILURE_REASONS = (
    ('holds no mask points', 'condylar_section_empty'),
    ('far side of the condylar divider', 'condylar_divider_one_sided'),
    ('in-plane voxel spacing', 'anisotropic_in_plane'),
    ('obsolete', 'obsolete_argument'),
)


def _condylar_failure_flag(exc: BaseException, bone: str = 'femur') -> str:
    """Turn a failed condylar reference-line search into one ``*_condylar_landmark_failed`` flag.

    :param exc: The exception the search raised.
    :param bone: The bone whose line failed, ``'femur'`` or ``'tibia'``; it only names the flag
        (see :data:`_BONE_FLAG_PREFIX`). An unrecognised value falls back to the bare
        ``condylar_`` prefix, because the one way to reach this with a bad ``bone`` is for that bad
        value to *be* the failure, and naming the flag after it would be circular.
    :return: ``<bone>_condylar_landmark_failed:<reason>``, where ``<reason>`` is the matching code
        from :data:`_CONDYLAR_FAILURE_REASONS` or, for anything unrecognised, the exception's type
        name (so a new failure mode is still visible, just unnamed).
    """
    prefix = _BONE_FLAG_PREFIX.get(bone)
    name = f'{prefix}_condylar_landmark_failed' if prefix else 'condylar_landmark_failed'
    message = str(exc)
    for marker, reason in _CONDYLAR_FAILURE_REASONS:
        if marker in message:
            return f'{name}:{reason}'
    return f'{name}:{type(exc).__name__}'


def get_condylar_line_qc(knee_image, bone: str, segmentation_label: int = 1,
                         side: str = 'left', spacing: Tuple[float, ...] | None = None) -> str:
    """
    QC flags for the posterior condylar reference line of one knee image half.

    The entry point the batch scripts use, parallel to
    :func:`morphometry.femur.get_murphy_neck_qc` and :func:`morphometry.femur.get_lee_neck_qc`: it
    resolves the condylar line exactly as its consumers do -- binarise the bone's label, call
    :func:`get_knee_reference_line` with the **default** ``segmentation_label`` on that binary
    array, which is what ``measurements.knee.calculate_knee_rotation_angle`` and
    ``measurements.tibia._tibial_torsion_points`` do -- and hands the endpoints to
    :func:`knee_condylar_line_qc`. So the flags describe the line the measurements actually used,
    and the layer walk, notch search and tangent sweep are not reimplemented here, which is what
    would let the gate and the measurements drift apart. The line is resolved a second time rather
    than threaded out of the measurement functions, which keeps their signatures and return types
    untouched; the cost is one extra layer walk and tangent search per bone and image half, which
    comes to roughly **0.9x one ``calculate_knee_rotation_angle``** of the same knee for all four
    of its lines -- about 100 ms per patient on the Augsburg knee MRI, i.e. a couple of minutes
    single-process over the whole 954-case cohort. The *ratio* is the figure to quote: two
    independent timings of PA000001 (means of 5 runs) gave 94 ms against 104 and 105 against 119,
    so the absolute numbers are machine-dependent in the way this package's other timings are, and
    are not a regression signal. That is far cheaper in relative terms than
    :func:`morphometry.femur.get_murphy_neck_qc`'s +11.4% per NaKo case, because it duplicates two
    section searches rather than a femoral-head fit -- so here the price is an argument *for*
    resolving the line again rather than threading it out of the measurements.

    **A right image half is reflected first, because this landmark's search is only approximately
    mirror-equivariant.** The three consumers all measure a right half as the left-right reflection
    of a left one (``mask[::-1]`` on the sagittal axis), and that is not cosmetic here: the layer
    choice, the notch search and the dorsal-pair selection make a half and its mirror give condylar
    lines differing by a median 0.33 and up to 2.06 degrees of bearing, so the *length* can differ
    too. Reflecting the half the same way the measurement does makes the flags describe the same
    line, bitwise, rather than a second line that merely resembles it. (The length of a *given* line
    is exactly mirror-invariant; it is the search that is not.) It is **not** a small effect on the
    length: over the 1903 measurable right halves of the Augsburg large cohort the reflected and
    direct readings differ by a median 2.27 voxels, up to 18.03, and agree exactly on only 276 --
    against a threshold of 14.30 -- and they disagree about *which* tibial legs are flagged. Do not
    infer length stability from task #5's mirror figure, which bounds the line's **bearing**
    (median 0.33, max 2.06 degrees) and says nothing about its length.

    **Which consumers this reproduces, and the one it does not.**
    ``measurements.knee.calculate_knee_rotation_angle`` and
    ``measurements.tibia._tibial_torsion_points`` both binarise the bone and call
    :func:`get_knee_reference_line` at its default label, so at ``segmentation_label=1`` --
    and, for the tibia, at whatever label the caller binarised with -- this gate sees their exact
    line. ``measurements.femur._mri_torsion`` instead passes its own ``segmentation_label``
    *through* to :func:`get_knee_reference_line` on an array it has already binarised, which
    double-filters to an empty mask for any label other than 1; at the default label all three
    agree with this gate, and for a non-default label that path is broken independently of this
    function. So the femoral torsion's condylar line is reproduced here exactly where it is
    well defined.

    :param knee_image: An ``Image`` / ``Segmentation`` of the knee segmentation mask (one image
        half), or a bare mask array. An ``Image`` is unwrapped with
        :func:`morphometry.image_io.mask_and_spacing` so its spacing reaches the in-plane isotropy
        check with no extra argument; a bare array carries no geometry, so unless ``spacing`` is
        given the check is skipped and no millimetre length is reported (see
        :func:`knee_condylar_line_qc`). The flag's value is in voxels either way.
    :param bone: ``'femur'`` or ``'tibia'`` -- which reference line to describe. The two are
        separate landmarks on separate sections and get one call each; the femoral line feeds the
        femoral torsion and the knee rotation angle, the tibial one the tibial torsion and the knee
        rotation angle.
    :param segmentation_label: The label of that bone in the knee segmentation mask (femur 1,
        tibia 2 in the Augsburg knee masks).
    :param side: Side of the IMAGE (not the patient!), either ``'left'`` or ``'right'``.
    :param spacing: The voxel spacing of the knee image, for a caller holding bare arrays (as
        ``morphometry/docker/compute_torsion.py`` does -- it splits the knee mask into two plain
        arrays and keeps the parent's spacing beside them). An explicit value **wins over** the
        image's, which is the convention task #10 established for the torsion entry points'
        ``knee_spacing`` / ``ankle_spacing``.
    :return: The ``;``-joined flag string, ``''`` when the line looks sound, or
        ``<bone>_condylar_landmark_failed:<reason>`` when the search itself failed -- the measurements
        that read this line are then NaN too (see :func:`_condylar_failure_flag` for the reason
        codes). **Never raises**, for the same reason the two femoral wrappers do not: a QC helper
        must not cost its caller a measurement, and the call sites run it outside any ``try``
        block.
    """
    try:
        if bone not in _BONE_FLAG_PREFIX:
            # validated here, before the search, so the named ValueError of
            # `knee_condylar_line_qc` is reachable through this wrapper too: otherwise
            # `get_knee_reference_line`'s own `assert bone in [...]` fires first and the flag reads
            # the uninformative `condylar_landmark_failed:AssertionError`.
            raise ValueError(f"bone must be either 'femur' or 'tibia', got {bone!r}.")
        if side not in ('left', 'right'):
            raise ValueError(f"side must be either 'left' or 'right', got {side!r}.")
        array, image_spacing = mask_and_spacing(knee_image)
        resolved_spacing = spacing if spacing is not None else image_spacing
        mask = np.where(np.asarray(array) == segmentation_label, 1, 0)
        if side == 'right':
            # the reflection the three consumers apply before searching; see the docstring
            mask = mask[::-1]
        _, start, end = get_knee_reference_line(mask, bone)
        return knee_condylar_line_qc(start, end, spacing=resolved_spacing, bone=bone)['flags']
    except Exception as exc:
        return _condylar_failure_flag(exc, bone=bone)
