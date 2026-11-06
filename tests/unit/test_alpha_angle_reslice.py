"""The alpha angle's oblique reslice: its basis, its window, and the isotropy precondition.

``measurements.hip.calculate_alpha_angle`` reads its contour in the plane that contains the
femoral neck axis. It used to obtain that plane by rotating the whole volume with
``scipy.ndimage.rotate(mask, axes=(0, 2), angle=rotation_angle, reshape=True, order=0)`` and
slicing the result, transporting the head and neck landmarks through the same rotation by painting
a 3x3x3 cube, rotating it, re-centroiding and rounding. ``reshape=True`` rotates about the *input
array's centre* and sizes the output to fit, so both the sampling grid and the transported
landmarks followed the array bounds rather than the anatomy: over the 120 NaKo sample halves a pad
of (3, 0, 2) voxels -- which leaves every voxel's physical position untouched -- reproduced the
resampled slice on only 2 of 120 and moved the angle on 120 of 120, by a median 0.75 and up to
7.32 degrees.

``_reslice_femoral_neck_section`` samples that plane directly instead, on a grid anchored to the
(float) head centre and sized from the mask's own extent. These tests pin the three things that
make it correct rather than merely different:

1. **the basis**, against the ``rotate`` call it replaces. The sense of the rotation is not
   something to guess: the section is reproduced *bitwise* from a ``rotate(..., reshape=True,
   order=0)`` of the same phantom, at a general angle, and the two degenerate angles (0 and 90
   degrees) are checked against the plain array slices they must equal;
2. **the window**, which must come from the anatomy alone -- a one-voxel zero margin all round (so
   ``utils.get_contour``'s ``binary_erosion`` never reads the border as contour), the head centre
   on an integer cell, and exact invariance under a pad, a crop and a pad at the high end;
3. **the boundary mode**, implicitly: scipy's ``mode='constant'`` returns ``cval`` for any
   coordinate outside ``[0, n - 1]``, which drops the outer half voxel of every axis and so is
   itself a dependence on where the array ends. The sampler uses ``'grid-constant'`` instead, and
   the window and high-end-pad tests are what would catch a regression to the other.

The isotropy guard is pinned here too, because it needs no data: the construction is an
index-space rotation, which is a rotation of the anatomy only on isotropic voxels, so
``geometry.validate_isotropy`` refuses a known anisotropic spacing with a ``ValueError`` and stays
silent when the spacing is unknown. See ``tests/characterization/test_alpha_angle_invariance.py``
for the same properties inside the measurement, on real NaKo hips.
"""
import numpy as np
import pytest
from scipy.ndimage import rotate

from morphometry import constants as C
from morphometry import geometry as G
from morphometry.measurements.hip import _reslice_femoral_neck_section, calculate_alpha_angle

#: A phantom shape with an odd extent on both rotated axes, so that ``(shape - 1) / 2`` is an
#: integer voxel and a head centre placed there maps onto an integer cell of the ``rotate``
#: output -- which is what lets the two be compared cell by cell. See
#: :func:`test_section_reproduces_the_rotate_reference`.
PHANTOM_SHAPE = (41, 13, 37)
#: A rotation angle for which ``rotate(..., reshape=True)`` also comes out odd on both axes
#: (output shape (55, 13, 53), centre (27, 6, 26)). Deliberately a general angle: the sense of the
#: rotation cannot be checked at a multiple of 90 degrees.
PHANTOM_ANGLE = 33.0


@pytest.fixture
def phantom():
    """An asymmetric binary volume, its centre, and a neck centre offset from it.

    Asymmetric on purpose, in x and in z: a shape symmetric about either axis would be reproduced
    by the opposite rotation sense as well, and the sense is exactly what has to be pinned.

    :return: ``(volume, head_centre, neck_centre)`` with the head centre at the array centre.
    """
    volume = np.zeros(PHANTOM_SHAPE, dtype=np.int64)
    volume[8:30, 3:10, 6:22] = 1       # the bulk
    volume[30:36, 4:8, 22:30] = 1      # a lobe at high x, high z
    volume[12:16, 2:5, 24:33] = 1      # another at low x, higher z
    head = (np.asarray(PHANTOM_SHAPE, dtype=float) - 1) / 2
    neck = head + np.array([-7.3, 1.4, -4.2])
    return volume, head, neck


# --------------------------------------------------------------- the basis, against `rotate`

def test_section_reproduces_the_rotate_reference(phantom):
    """The sampled section is bitwise the slice the replaced ``rotate`` call produced.

    The comparison is exact rather than approximate, and that is the point: with ``order=0`` both
    resamplings pick one source voxel per sample, so if the basis ``u = (cos t, 0, -sin t)``,
    ``v = (0, 1, 0)`` is the one ``rotate(axes=(0, 2))`` applies, every cell must agree. With the
    opposite sense it does not (the plan's convention probe reproduced 23 of 53 slices that way
    against 53 of 53 with this one).

    The head centre is placed at the input array's centre, which ``rotate(reshape=True)`` maps
    onto the output array's centre; both are integer voxels for this shape and angle, so the
    section's cell ``(i, j)`` is the output's ``(out_c0 + i - Ru, out_c1 + j - Rv, out_c2)``.
    Cells of the window that fall outside the rotated array must be zero -- both grids pad with
    zeros, so that is part of the equality, not an exemption from it.
    """
    volume, head, neck = phantom
    section, head_cell, _ = _reslice_femoral_neck_section(volume, head, neck, PHANTOM_ANGLE)

    reference_volume = rotate(volume, axes=(0, 2), angle=PHANTOM_ANGLE, reshape=True, order=0)
    out_centre = (np.asarray(reference_volume.shape, dtype=float) - 1) / 2
    assert out_centre[0] == int(out_centre[0]) and out_centre[2] == int(out_centre[2]), (
        'the reference rotation must land its centre on a voxel for a cell-by-cell comparison')

    expected = np.zeros_like(section)
    i0, j0 = int(out_centre[0]) - head_cell[0], int(out_centre[1]) - head_cell[1]
    for i in range(section.shape[0]):
        for j in range(section.shape[1]):
            ii, jj = i0 + i, j0 + j
            if 0 <= ii < reference_volume.shape[0] and 0 <= jj < reference_volume.shape[1]:
                expected[i, j] = reference_volume[ii, jj, int(out_centre[2])]

    assert section.sum() > 0, 'the section is empty, so the comparison would be vacuous'
    assert np.array_equal(section, expected), (
        f'{int(np.sum(section != expected))} of {section.size} cells differ from the rotate '
        f'reference at {PHANTOM_ANGLE} degrees')


def test_section_at_zero_degrees_is_the_axial_slice_through_the_head(phantom):
    """At 0 degrees the basis is the identity, so the section is the plain ``(x, y)`` slice."""
    volume, head, neck = phantom
    section, (r_u, r_v), _ = _reslice_femoral_neck_section(volume, head, neck, 0.0)
    expected = np.zeros_like(section)
    for i in range(section.shape[0]):
        for j in range(section.shape[1]):
            x, y, z = int(head[0]) + i - r_u, int(head[1]) + j - r_v, int(head[2])
            if 0 <= x < volume.shape[0] and 0 <= y < volume.shape[1]:
                expected[i, j] = volume[x, y, z]
    assert np.array_equal(section, expected)


def test_section_at_ninety_degrees_is_the_sagittal_plane_through_the_head(phantom):
    """At 90 degrees ``u`` is ``-ez``, so the section is the ``(z, y)`` plane, reversed in z.

    The complement of the 0 degree case: it fixes which way round the collapsed axis goes, which
    a symmetric probe would not.
    """
    volume, head, neck = phantom
    section, (r_u, r_v), _ = _reslice_femoral_neck_section(volume, head, neck, 90.0)
    expected = np.zeros_like(section)
    for i in range(section.shape[0]):
        for j in range(section.shape[1]):
            y, z = int(head[1]) + j - r_v, int(head[2]) - (i - r_u)
            if 0 <= y < volume.shape[1] and 0 <= z < volume.shape[2]:
                expected[i, j] = volume[int(head[0]), y, z]
    assert np.array_equal(section, expected)


# ------------------------------------------------------------------------------- the window

def test_the_window_has_a_one_voxel_zero_margin(phantom):
    """No mask voxel reaches the window border, on any of the four sides.

    ``utils.get_contour`` erodes with the default ``border_value=0``, so a mask voxel on the
    border would be reported as contour and the head's circle fit would see a straight edge. One
    voxel of margin is enough because ``order=0`` rounds each coordinate to the nearest voxel: a
    sample that returns a mask voxel is within half a voxel of it on every axis, hence within
    ``0.5 * (|cos t| + |sin t|) <= 0.708`` of it along ``u`` and within 0.5 along ``v``.
    """
    volume, head, neck = phantom
    for angle in (0.0, 7.0, 33.0, 45.0, 62.0, 90.0):
        section, _, _ = _reslice_femoral_neck_section(volume, head, neck, angle)
        assert section[0].sum() == 0 and section[-1].sum() == 0, f'x border not empty at {angle}'
        assert section[:, 0].sum() == 0 and section[:, -1].sum() == 0, f'y border not empty at {angle}'


def test_the_window_covers_the_whole_anatomy(phantom):
    """Every mask voxel's projection falls inside the window, so nothing is cropped away.

    The window is sized from the mask's own extent; this is the property that makes it safe to
    not look at the array bounds at all.
    """
    volume, head, neck = phantom
    angle = PHANTOM_ANGLE
    theta = np.deg2rad(angle)
    u = np.array([np.cos(theta), 0.0, -np.sin(theta)])
    section, (r_u, r_v), _ = _reslice_femoral_neck_section(volume, head, neck, angle)
    voxels = np.argwhere(volume > 0).astype(float)
    du = voxels @ u - head @ u
    dv = voxels[:, 1] - head[1]
    assert r_u + du.min() >= 1.0 and r_u + du.max() <= section.shape[0] - 2
    assert r_v + dv.min() >= 1.0 and r_v + dv.max() <= section.shape[1] - 2


def test_the_head_centre_lands_on_an_integer_cell(phantom):
    """The head cell is integral, which is what lets the caller compare contour points to it.

    Every downstream comparison in ``calculate_alpha_angle`` is against this cell (the
    anterior/posterior split of the contour, the lateral restriction, the circle-fit band), so it
    has to be an exact grid position rather than a rounded float.
    """
    volume, head, neck = phantom
    _, head_cell, _ = _reslice_femoral_neck_section(volume, head, neck, PHANTOM_ANGLE)
    assert head_cell.dtype.kind in 'iu'
    assert head_cell.shape == (2,)


def test_the_neck_cell_is_continuous_and_carries_the_rotated_offset(phantom):
    """The neck cell is the head cell plus the neck offset in the section basis, unrounded.

    ``u`` is a unit vector, so the dot product *is* the coordinate along it, and ``v`` is index
    axis 1 exactly. Nothing indexes an array with this cell -- it is only used as the direction
    ``neck_cell - circle_centre`` -- so it is deliberately not rounded, unlike the cube-transported
    ``int16`` landmark it replaces.
    """
    volume, head, neck = phantom
    theta = np.deg2rad(PHANTOM_ANGLE)
    u = np.array([np.cos(theta), 0.0, -np.sin(theta)])
    _, head_cell, neck_cell = _reslice_femoral_neck_section(volume, head, neck, PHANTOM_ANGLE)
    expected = np.array([(neck - head) @ u, neck[1] - head[1]])
    np.testing.assert_allclose(neck_cell - head_cell, expected, atol=1e-12)
    assert not np.allclose(neck_cell, np.round(neck_cell)), 'the probe must exercise a non-integer cell'


@pytest.mark.parametrize('pad', [(1, 1, 1), (3, 0, 2), (7, 5, 4)])
def test_the_section_is_invariant_under_a_pad(phantom, pad):
    """Padding at the low end and moving the landmarks with it reproduces the section exactly.

    This is the defect's own probe: the anatomy has not moved, only its indices have, so the
    section and both cells must be identical. Asserted as exact array equality -- the sampled
    input coordinates are the same numbers plus the same integer pad, so ``order=0`` picks the
    same source voxels.
    """
    volume, head, neck = phantom
    before, head_before, neck_before = _reslice_femoral_neck_section(volume, head, neck, PHANTOM_ANGLE)
    shift = np.asarray(pad, dtype=float)
    after, head_after, neck_after = _reslice_femoral_neck_section(
        np.pad(volume, [(k, 0) for k in pad]), head + shift, neck + shift, PHANTOM_ANGLE)
    assert np.array_equal(after, before)
    assert np.array_equal(head_after, head_before)
    np.testing.assert_allclose(neck_after, neck_before, atol=1e-12)


def test_the_section_is_invariant_under_a_crop_to_the_bounding_box(phantom):
    """Cropping the empty margin away reproduces the section exactly.

    The operation the API performs (``api/ingest/dicom.py`` crops at detected gaps through
    ``image_io.crop_image``), and the sharper of the translation probes because it also *shrinks*
    the array -- which is what the replaced read-out derived its back-transform from.
    """
    volume, head, neck = phantom
    low = np.array([int(a.min()) for a in np.nonzero(volume)])
    high = np.array([int(a.max()) + 1 for a in np.nonzero(volume)])
    before, head_before, neck_before = _reslice_femoral_neck_section(volume, head, neck, PHANTOM_ANGLE)
    cropped = volume[low[0]:high[0], low[1]:high[1], low[2]:high[2]]
    after, head_after, neck_after = _reslice_femoral_neck_section(
        cropped, head - low, neck - low, PHANTOM_ANGLE)
    assert np.array_equal(after, before)
    assert np.array_equal(head_after, head_before)
    np.testing.assert_allclose(neck_after, neck_before, atol=1e-12)


def test_the_section_ignores_a_pad_at_the_high_end(phantom):
    """Growing the array past the anatomy must change nothing at all.

    No anatomy index moves, so the expected difference is zero and any movement is pure dependence
    on the array's *shape* -- which is exactly what ``rotate(reshape=True)`` had.
    """
    volume, head, neck = phantom
    before = _reslice_femoral_neck_section(volume, head, neck, PHANTOM_ANGLE)
    after = _reslice_femoral_neck_section(np.pad(volume, [(0, 6), (0, 4), (0, 3)]), head, neck,
                                          PHANTOM_ANGLE)
    assert np.array_equal(after[0], before[0])
    assert np.array_equal(after[1], before[1])
    np.testing.assert_allclose(after[2], before[2], atol=1e-12)


def test_a_section_through_the_outermost_half_voxel_still_reads_the_bone():
    """A plane that passes within half a voxel of the last slice reads it, padded or not.

    The boundary mode in its own right, on a phantom built for it: the bone reaches the array's
    last slice and the section's plane runs through ``z = n - 1 + 0.3``, inside the outermost
    voxel's half-voxel domain. scipy's ``mode='constant'`` returns ``cval`` for *any* coordinate
    outside ``[0, n - 1]``, so under it the section reads background until empty slices are
    appended -- a dependence on where the array ends, which is the defect class this whole module
    is about. ``'grid-constant'`` reads the voxel either way.

    This is not hypothetical: on the Augsburg reader-study hips (12 slices) exactly this cost
    PA000668 image-right 14.98 degrees of alpha angle under a 3-slice high-end pad, the single
    non-invariant half of 100 before the mode was changed.
    """
    volume = np.zeros((21, 7, 9), dtype=np.int64)
    volume[6:15, 2:5, 4:9] = 1                      # reaches the last slice, z = 8
    head = np.array([10.0, 3.0, 8.3])               # 0.3 voxels past it, inside its half-voxel cell
    neck = head + np.array([-5.0, 0.0, 0.0])        # a rotation angle of 0: u is +ex, the plane is z

    section, _, _ = _reslice_femoral_neck_section(volume, head, neck, 0.0)
    assert section.sum() > 0, 'the section read background instead of the outermost slice'

    padded, _, _ = _reslice_femoral_neck_section(np.pad(volume, [(0, 0), (0, 0), (0, 3)]),
                                                 head, neck, 0.0)
    assert np.array_equal(padded, section)


def test_a_failed_contour_search_is_a_value_error_naming_the_side():
    """When no contour point satisfies the anterior / posterior predicate, the result is a NaN row.

    **This path is live, not hypothetical.** It is never reached on the 120 NaKo halves (0 of 120
    on the base and all three index-frame probes, on the shipped code and on this one), but on the
    100 Augsburg reader-study halves the shipped code's search already found nothing on **35**
    (anterior 26, posterior 20) and this one on **31** (25, 19) -- and the shipped code pre-set the
    two points to ``np.array([0, 0])``, so each of those 35 silently reported the angle from the
    fitted circle's centre to the **window's corner**. That sentinel also sat somewhere different
    before and after this change (the reshaped rotation's array corner, then the anatomy-anchored
    window's), which would have made a silent wrong answer silently different.

    So the failure is now a ``ValueError``, which the batch scripts' and ``compute_torsion.py``'s
    per-measurement handlers catch and record as ``NaN``. ``ValueError`` specifically: an
    ``UnboundLocalError`` (i.e. a ``NameError``) would escape those handlers and abort a cohort
    run. The search predicate itself is untouched -- this changes what happens when it finds
    nothing, never what it finds.

    The phantom is a bare head sphere on a neck cylinder, built so the circle fit has nothing
    outside it to find; it is a negative control for the error path, not a plausible femur.
    """
    n = 90
    volume = np.zeros((n, n, n), dtype=np.int16)
    zz, yy, xx = np.meshgrid(np.arange(n), np.arange(n), np.arange(n), indexing='ij')
    volume[(xx - 60.) ** 2 + (yy - 45.) ** 2 + (zz - 25.) ** 2 <= 15. ** 2] = 1
    for t in np.linspace(0, 1, 400):     # a smooth neck, no head-neck prominence to find
        volume[(xx - (60 - 32 * t)) ** 2 + (yy - 45.) ** 2 + (zz - (25 + 26 * t)) ** 2 <= 9. ** 2] = 1

    with pytest.raises(ValueError, match=r'no (anterior|posterior) contour point'):
        calculate_alpha_angle(volume, 'left', 1, isotropic=True)


def test_an_empty_mask_is_a_value_error(phantom):
    """An empty femur gives the window no extent, and must fail as a ``ValueError``.

    The batch scripts and ``compute_torsion.py`` catch ``(RuntimeError, AssertionError,
    ValueError)`` per measurement, so anything else would abort a cohort run instead of writing a
    NaN row.
    """
    _, head, neck = phantom
    with pytest.raises(ValueError, match='empty'):
        _reslice_femoral_neck_section(np.zeros(PHANTOM_SHAPE, dtype=np.int64), head, neck, 10.0)


# ------------------------------------------------------------------- the isotropy precondition

#: Spacings of the cohorts the alpha angle is or could be run on. Only the isotropic ones are
#: admissible: the construction rotates in the ``(x, z)`` index plane and reads its contour in the
#: ``(rotated x, y)`` plane, so all three edges have to match -- a stronger rule than the in-plane
#: one ``geometry.validate_in_plane_isotropy`` applies to the axial angles.
ISOTROPIC_SPACINGS = [(1.0, 1.0, 1.0), (0.3125, 0.3125, 0.3125), (2.5, 2.5, 2.5)]
ANISOTROPIC_SPACINGS = [
    (1.09375, 1.09375, 10.0),    # Augsburg hip MRI -- in-plane square, slice axis 10 mm
    (0.76171875, 0.76171875, 3.0),  # NMDID whole-leg CT
    (0.75, 0.3125, 0.3125),      # dfg_knees after the LPI reorientation
]


@pytest.mark.parametrize('spacing', ISOTROPIC_SPACINGS)
def test_isotropy_validator_accepts_isotropic_spacing(spacing):
    assert G.validate_isotropy(spacing, 'probe') is None


@pytest.mark.parametrize('spacing', ANISOTROPIC_SPACINGS)
def test_isotropy_validator_refuses_anisotropic_spacing(spacing):
    with pytest.raises(ValueError, match='anisotropic'):
        G.validate_isotropy(spacing, 'probe')


@pytest.mark.parametrize('spacing', [(1.09375, 1.09375, 10.0), (0.76171875, 0.76171875, 3.0)])
def test_isotropy_is_stricter_than_in_plane_isotropy(spacing):
    """The Augsburg hip MRI and the NMDID CT pass the in-plane check and fail this one.

    That gap is the reason for a separate helper rather than a reuse: 1.09375 x 1.09375 x 10 mm is
    fine for an angle composed in the axial plane and useless for a rotation that mixes x and z.
    """
    assert G.validate_in_plane_isotropy(spacing, 'probe') is None
    with pytest.raises(ValueError, match='anisotropic'):
        G.validate_isotropy(spacing, 'probe')


def test_isotropy_validator_accepts_none_and_leaves_the_precondition_to_the_caller():
    """An unknown spacing is silent, deliberately -- no warning either.

    Task #10's choice, for the same reason: the alpha angle's production callers hand it a bare
    array over 1 mm isotropic NaKo, so a warning would fire on every one of 30,407 cases and be
    tuned out. The fix for an array-only caller is to pass its spacing.
    """
    assert G.validate_isotropy(None, 'probe') is None


def test_isotropy_validator_ignores_the_sign_of_the_spacing():
    """A negative zoom is a direction, not a smaller voxel (nibabel affines carry them)."""
    assert G.validate_isotropy((-1.0, 1.0, -1.0), 'probe') is None


def test_isotropy_validator_accepts_float_noise_within_the_tolerance():
    """Headers store their zooms as float32; equal-by-intent spacings must not be refused."""
    edge = 1.0 + 0.9 * C.IN_PLANE_SPACING_RTOL
    assert G.validate_isotropy((1.0, edge, 1.0), 'probe') is None


def test_isotropy_validator_refuses_just_outside_the_tolerance():
    edge = 1.0 + 1.1 * C.IN_PLANE_SPACING_RTOL
    with pytest.raises(ValueError, match='anisotropic'):
        G.validate_isotropy((1.0, 1.0, edge), 'probe')


@pytest.mark.parametrize('spacing,reason', [
    ((1.0, 1.0), 'two entries'),
    ((1.0,), 'one entry'),
    ((1.0, 1.0, 0.0), 'a zero edge'),
    ((1.0, float('nan'), 1.0), 'a NaN edge'),
    ((1.0, 1.0, float('inf')), 'an infinite edge'),
    (('a', 'b', 'c'), 'non-numeric entries'),
])
def test_isotropy_validator_refuses_malformed_spacing_as_a_value_error(spacing, reason):
    """Every rejection is a ``ValueError`` and nothing else.

    A malformed header must not escape as an ``IndexError`` or a ``ZeroDivisionError``: the
    per-measurement handlers only catch ``(RuntimeError, AssertionError, ValueError)``, so
    anything else aborts the cohort run that this check exists to keep going.

    :param reason: What is wrong with the spacing, for the test id.
    """
    with pytest.raises(ValueError):
        G.validate_isotropy(spacing, 'probe')


def _labelled_hip_phantom(label: int, n: int = 90) -> np.ndarray:
    """A head sphere on a neck cylinder, labelled ``label``, on which the alpha angle resolves.

    The counterpart of :func:`test_a_failed_contour_search_is_a_value_error_naming_the_side`'s
    phantom: the neck is wide enough (radius 10 against the head's 15) that the head-neck
    prominence leaves the fitted circle, so both the anterior and the posterior search find a
    contour point and the measurement returns a pair instead of raising. It is a negative control
    for the *label*, not a plausible femur -- the geometry is symmetric about the section plane's
    ``y``, so the two angles come out equal, which this test does not depend on.

    :param label: the segmentation label to paint the bone with.
    :param n: the cubic volume's extent along each axis.
    :return: a ``(n, n, n)`` ``int16`` mask holding the bone at ``label`` and 0 elsewhere.
    """
    volume = np.zeros((n, n, n), dtype=np.int16)
    zz, yy, xx = np.meshgrid(np.arange(n), np.arange(n), np.arange(n), indexing='ij')
    volume[(xx - 60.) ** 2 + (yy - 45.) ** 2 + (zz - 25.) ** 2 <= 15. ** 2] = label
    for t in np.linspace(0, 1, 400):
        volume[(xx - (60 - 32 * t)) ** 2 + (yy - 45.) ** 2 + (zz - (25 + 26 * t)) ** 2 <= 10. ** 2] = label
    return volume


@pytest.mark.parametrize('label', [2, 5, 7])
def test_the_contour_honours_the_callers_segmentation_label(label):
    """The femoral contour must be read at ``segmentation_label``, not at a hardcoded 1.

    ``calculate_alpha_angle`` built its contour from ``np.where(mask == 1, 1, 0)`` while the head
    centre above it honours ``segmentation_label``, so a caller passing any other label got a
    contour of label 1 measured against a sphere fitted to its own label -- and on a mask holding
    nothing at label 1 that contour is *empty*, so the measurement raised instead of measuring.
    Every caller in the repo passes 1, which is why nothing caught it and why the repair is
    zero-movement; this phantom is the only thing that exercises the other branch.

    Relabelling the bone cannot change the geometry, so the result must be bit-identical to the
    label-1 reading of the same shape.

    :param label: a segmentation label other than 1 to paint the identical phantom with.
    """
    reference = calculate_alpha_angle(_labelled_hip_phantom(1), 'left', 1, isotropic=True)
    assert np.all(np.isfinite(reference)), 'the phantom must resolve, or the test cannot discriminate'

    relabelled = calculate_alpha_angle(_labelled_hip_phantom(label), 'left', label, isotropic=True)
    assert relabelled == reference, (
        f'label {label} read {relabelled}, label 1 read {reference} on the same geometry')
