"""Index-frame invariance of the posterior condylar line (``knee.get_knee_reference_line``), on real data.

Backlog task #5. Padding or cropping a mask changes voxel indices while leaving every physical
position untouched, so the reported condylar endpoints must shift by exactly that shift and no
derived angle may move. Up to task #4 they did not: the search read its landmarks off a
``skimage.transform.rotate`` resampling of the section, mapped them back through offsets derived
from the *rotated array's shape* and truncated the result to ``uint16``.

**The probes are run over several pads because one pad can pass by accident.** Measured over the
200 Augsburg reader-study sections (50 patients x 2 image halves x femur/tibia) against the
pre-change code, each of the five pads below left the endpoints correct on 45 to 85 of 200 (so even
a pad of one voxel on each in-plane axis failed 115 of them), but only **20 of 200** survived all
five (worst-pad error median 3.0, p90 27.4, max 57.0 voxels), and a crop to the mask's bounding box
left 32 of 200. The construction is now carried out in coordinates and all 200 are exact under
every one of them.

**What the upstream steps already did, and so what this is really pinning.** The layer choice, the
notch search, the divider and the rotation angle were all exactly equivariant before the change
(verified over the same 200 sections), so the entire defect lay in how the endpoints were read out
of the rotated frame. These probes pin the read-out.

The four (patient, image side) pairs below are chosen because they carried that content, not for
being typical -- each parametrisation records the pre-change error for that exact section. Note that
a large pre-change pad error is **not** the size of the change to the measurement: the endpoints
moved along a line that barely moved (as an undirected line the tangent's orientation changed by a
median 0.567, p90 1.75, max 3.995 degrees over the 200 sections), because the old read-out tended to
slide the reported points along the cortex rather than tilt the line.

**What these probes cannot catch.** The invariance now holds by construction and is provable rather
than measured: the rotation is taken about the divider, which is an integer index, so
``points - divider`` is exact integer arithmetic and a translated array reproduces the rotated
coordinates bitwise, while every selection (posterior-most point, side of the divider, extreme
bearing) is translation-invariant. These tests are therefore a guard against a resampling step being
reintroduced, not an independent derivation -- the same caveat
``test_knee_rotation_equivariance.py`` and ``test_femoral_head_equivariance.py`` carry. They do fail
against the pre-change code, which is the only reason they are worth having.

**What is still open.** A translation is not a rotation: the line is read in the index frame, where
"posterior" is the largest coronal index, so an obliquely acquired knee still gives a different
line, and the layer is still chosen by convex area per index slice. Position *and rotation*
dependence is task #9, and the one strict xfail left in ``test_head_centre_translation.py``
(Lee torsion, hip-side, through ``contour_femoral_neck``'s slice rotations) records what remains of
it. The alpha angle was the second of those until its oblique reslice was rewritten to sample the
section on a grid anchored to the head centre; see ``test_alpha_angle_invariance.py``.
"""
import numpy as np
import pytest

from morphometry.image_io import split_image_left_right
from morphometry.knee import get_knee_reference_line
from morphometry.measurements.knee import calculate_knee_rotation_angle

FEMUR_LABEL, TIBIA_LABEL = 1, 2
LABELS = {'femur': FEMUR_LABEL, 'tibia': TIBIA_LABEL}

#: (patient, image side, bone, pre-change pad error, pre-change bounding-box crop error), in voxels.
#: The pad is PAD below; the crop is to the mask's own bounding box in the two in-plane axes.
CASES = [
    ('PA001078', 'left', 'femur', 51.0, 1.0),
    ('PA000390', 'right', 'tibia', 38.0, 38.0),
    ('PA000941', 'left', 'tibia', 37.0, 1.0),
    ('PA000914', 'left', 'tibia', 10.0, 10.0),
]

#: Pads applied at the low end of each axis. Several, because a single pad can be survived by
#: accident (see the module docstring), and one of them is non-zero on the slice axis too so that
#: the layer index is exercised as well as the two in-plane axes.
PADS = [(1, 1, 0), (4, 3, 2), (7, 0, 0), (0, 5, 0), (13, 11, 1)]

#: A pad applied at the HIGH end of each axis instead. It leaves every anatomy index untouched, so
#: the correct answer is that nothing moves at all -- which isolates dependence on the array's
#: *shape*, the actual defect class (the old read-out derived its back-transform from the rotated
#: array's shape). Measured with this exact pad, the old code is correct on only 39 of the 200
#: reader-study sections (median error 1.0, p90 6.0, max 57.0 voxels); this one on all 200.
HIGH_END_PAD = (6, 4, 3)

pytestmark = pytest.mark.needs_augsburg_reader_study


def _half(knee, side: str) -> np.ndarray:
    """The requested image half of a knee segmentation, as a bare array.

    :param knee: The whole knee Segmentation (both legs).
    :param side: 'left' or 'right' image side.
    :return: That half's label array.
    """
    return split_image_left_right(knee)[0 if side == 'left' else 1].array


def _line(mask: np.ndarray, bone: str) -> np.ndarray:
    """The reported condylar line of a binary bone mask, as a 2x3 float array.

    :param mask: A 3D binary mask of the distal femur or proximal tibia.
    :param bone: 'femur' or 'tibia'.
    :return: ``[[start], [end]]`` as float index triples.
    """
    _, start, end = get_knee_reference_line(mask, bone)
    return np.asarray([start, end], dtype=float)


def _bone(knee, side: str, bone: str) -> np.ndarray:
    """The binary mask of one bone of one image half.

    :param knee: The whole knee Segmentation.
    :param side: 'left' or 'right' image side.
    :param bone: 'femur' or 'tibia'.
    :return: The binary mask as an int array.
    """
    return np.where(_half(knee, side) == LABELS[bone], 1, 0)


@pytest.mark.parametrize('patient,side,bone,pad_error,crop_error', CASES,
                         ids=[f'{p}-{s}-{b}' for p, s, b, _, _ in CASES])
def test_endpoints_follow_a_pad(augsburg_reader_study, patient, side, bone, pad_error, crop_error):
    """Padding the mask at the low end of every axis shifts both endpoints by exactly the pad.

    Asserted exactly, not to a tolerance: the rotated coordinates are reproduced bitwise under a
    translation, so any difference at all is a defect rather than floating-point reassociation. All
    five pads are checked on each section, for the reason the module docstring gives.

    :param pad_error: The error this section showed under the (4, 3, 2) pad before task #5, in
        voxels; carried only so the case cannot be mistaken for an arbitrary pick.
    :param crop_error: The same for the crop probe below.
    """
    mask = _bone(augsburg_reader_study(patient)['knee'], side, bone)

    before = _line(mask, bone)
    for pad in PADS:
        after = _line(np.pad(mask, [(k, 0) for k in pad]), bone)
        assert (after - before == np.asarray(pad, dtype=float)).all(), (
            f'{patient} image-{side} {bone}: endpoints moved by {(after - before).tolist()} under '
            f'a pad of {pad} (error was {pad_error:.1f} voxels under (4, 3, 2) before task #5)')


@pytest.mark.parametrize('patient,side,bone,pad_error,crop_error', CASES,
                         ids=[f'{p}-{s}-{b}' for p, s, b, _, _ in CASES])
def test_endpoints_follow_a_crop(augsburg_reader_study, patient, side, bone, pad_error, crop_error):
    """Cropping the empty margin away shifts both endpoints by exactly the crop.

    This is the operation the API actually performs (``api/ingest/dicom.py::_detect_slabs`` and
    ``_split_volume`` crop through ``image_io.crop_image``), which is why it is probed separately
    from the pad: the two are the same translation, but a crop also *shrinks* the array, and the old
    read-out derived its back-transform from the array's shape.

    :param pad_error: The pre-change error under the pad probe, in voxels.
    :param crop_error: The pre-change error under this probe, in voxels.
    """
    mask = _bone(augsburg_reader_study(patient)['knee'], side, bone)
    low = np.array([int(axis.min()) for axis in np.nonzero(mask)])
    high = np.array([int(axis.max()) + 1 for axis in np.nonzero(mask)])

    before = _line(mask, bone)
    cropped = mask[low[0]:high[0], low[1]:high[1], low[2]:high[2]]
    after = _line(cropped, bone)

    assert (after - before == -low.astype(float)).all(), (
        f'{patient} image-{side} {bone}: endpoints moved by {(after - before).tolist()} under a '
        f'crop to the bounding box {low.tolist()} (error was {crop_error:.1f} voxels before task #5)')


@pytest.mark.parametrize('patient,side,bone,pad_error,crop_error', CASES,
                         ids=[f'{p}-{s}-{b}' for p, s, b, _, _ in CASES])
def test_endpoints_ignore_a_pad_at_the_high_end(augsburg_reader_study, patient, side, bone,
                                                pad_error, crop_error):
    """Growing the array past the anatomy must not move the endpoints at all.

    The complement of the pad probe, and the sharper of the two: padding at the high end changes no
    anatomy index, so the expected shift is zero and any movement is pure dependence on the array's
    shape. That is exactly what the replaced read-out had -- it mapped its landmarks back through
    offsets computed from the rotated array's shape -- and under this pad it is correct on only 39
    of the 200 sections here, off by a median 1.0 and up to 57.0 voxels.

    :param pad_error: The pre-change error under the low-end (4, 3, 2) pad, in voxels.
    :param crop_error: The pre-change error under the crop probe, in voxels.
    """
    mask = _bone(augsburg_reader_study(patient)['knee'], side, bone)

    before = _line(mask, bone)
    after = _line(np.pad(mask, [(0, k) for k in HIGH_END_PAD]), bone)

    assert (after == before).all(), (
        f'{patient} image-{side} {bone}: endpoints moved by {(after - before).tolist()} when the '
        f'array was grown by {HIGH_END_PAD} past the anatomy')


@pytest.mark.parametrize('patient,side,bone,pad_error,crop_error', CASES,
                         ids=[f'{p}-{s}-{b}' for p, s, b, _, _ in CASES])
def test_endpoints_are_mask_points(augsburg_reader_study, patient, side, bone, pad_error, crop_error):
    """Both reported endpoints are voxels of the bone.

    **That, and not "on the posterior cortex", is the property task #5 guarantees.** The endpoints
    are mask points returned in their original coordinates, so being a mask voxel holds by
    construction. Before task #5 it did not: the back-rotation plus the ``uint16`` cast could land a
    point *outside* the mask entirely, and in one case (PA000717 image-right's tibial start) in a
    column holding no bone at all. By the repo's own cue (nearest point of the per-column posterior
    cortex, ``tests/unit/test_knee_reference_line.py::_cortex_distances``) only 195 of this cohort's
    400 endpoints lay on the cortex, the rest a voxel or two off it (median 1.00, max 2.00).

    All 400 are exactly on the cortex now, but that is **not** a general property and is deliberately
    not asserted. "Posterior-most" is defined in the *rotated* condylar frame, so where the frame
    angle is large that extremum need not be the posterior-most voxel of its own *unrotated* column.
    On ``dfg_knees`` it is not, on 2 of 95 clean, LPI, notch-resolving femurs: K5_90_unbelastet at a
    frame angle of -75.96 degrees (start 2 voxels anterior of its column's cortex) and
    K9_90_unbelastet at -75.07 (1 voxel), against K1_0_unbelastet at -34.08, which is exactly 0.
    Augsburg never shows it because its frame angles are small, which is why asserting it here would
    have been a trap: the test would pass on this cohort and fail on the first one with real
    condylar tilt. What is asserted instead is the cue that discriminates the defect task #4 fixed
    -- that the point is far from the ANTERIOR cortex, where the old chord-alignment fallback put
    it. ``tests/unit/test_knee_reference_line.py`` and ``test_condylar_fallback.py`` do pin the
    distance to the posterior cortex at 0, on phantoms and on three Augsburg legs whose frames are
    turned by a few degrees; those are regression checks on those cases, not statements of a
    property.
    """
    mask = _bone(augsburg_reader_study(patient)['knee'], side, bone)
    layer, start, end = get_knee_reference_line(mask, bone)
    section = mask[:, :, int(layer)]
    columns = np.nonzero(section.any(axis=1))[0]
    posterior = np.array([[x, np.nonzero(section[x])[0].max()] for x in columns], dtype=float)
    anterior = np.array([[x, np.nonzero(section[x])[0].min()] for x in columns], dtype=float)

    for name, point in (('start', start), ('end', end)):
        x, y = int(point[0]), int(point[1])
        assert float(point[0]) == x and float(point[1]) == y, f'{name} is not a voxel index'
        assert section[x, y], f'{name} {point.tolist()} is not a mask voxel'
        p = np.asarray(point, dtype=float)[:2]
        to_posterior = float(np.linalg.norm(posterior - p, axis=1).min())
        to_anterior = float(np.linalg.norm(anterior - p, axis=1).min())
        # Relative, so it survives a turned frame: the margin is the smallest over the whole
        # cohort (6.3 voxels, on a narrow femoral section), while the defect this discriminates
        # put the point ON the anterior cortex -- 40+ voxels the wrong way round.
        assert to_anterior >= to_posterior + 5, (
            f'{name} {point.tolist()} is {to_posterior:.1f} voxels from the posterior cortex and '
            f'{to_anterior:.1f} from the anterior one; it is on the wrong aspect of the bone')


@pytest.mark.parametrize('patient,side', sorted({(p, s) for p, s, _, _, _ in CASES}),
                         ids=lambda v: v if isinstance(v, str) else str(v))
def test_knee_rotation_angle_is_unchanged_by_a_pad(augsburg_reader_study, patient, side):
    """End to end: the angle built from two condylar lines does not move when the indices do.

    The endpoint invariance above is what matters for the overlays the UI draws; this is what
    matters for the numbers. Both condylar searches run on the same padded array, and on a right
    half both run on its reflection, whose extra empty columns land past the anatomy -- so the angle
    is unchanged on either side.
    """
    mask = _half(augsburg_reader_study(patient)['knee'], side)

    before = calculate_knee_rotation_angle(mask, FEMUR_LABEL, TIBIA_LABEL, side=side)
    for pad in PADS:
        after = calculate_knee_rotation_angle(np.pad(mask, [(k, 0) for k in pad]),
                                              FEMUR_LABEL, TIBIA_LABEL, side=side)
        assert after == before, (f'{patient} image-{side} knee rotation angle moved by '
                                 f'{abs(after - before):.4f} deg under a pad of {pad}')
