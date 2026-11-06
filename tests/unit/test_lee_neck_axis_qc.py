"""Unit tests for the Lee proximal-neck-axis QC gate, on phantoms (no image data needed).

``morphometry.femur.get_femoral_neck_center_lee`` fits the proximal neck axis to the femoral
contour inside a ring about the femoral head centre, and ``femur.contour_femoral_neck`` places that
ring entirely from one scalar: ``a = |notch - head centre| / r``, the ring then spanning
``(a - 0.1) * r`` to ``(a + 0.1) * r``. The ring is supposed to sit peripheral to the head, on the
neck. ``a < 1`` means it does not: the ring is drawn closer to the head axis than the head's own
radius, far too small to follow that peripheral contour, and ``femur.lee_neck_axis_qc`` flags it as
``lee_ring_within_head_radius`` without changing the angle. (It is **not** the claim that the ring
lies inside the head -- `a` is an in-plane distance and the layer sits well off the head's equator,
so on three of the four flagged Augsburg halves the ring misses the head's cross-section there
entirely. See `femur.lee_neck_axis_qc`'s docstring for the per-half numbers; the gate is unaffected,
only the mechanism is weaker than it looks.)

Four things are pinned here:

1. **the flag itself**, including the boundary (``a == 1`` is sound) and the stricter
   ``ring_wholly_within_head_radius`` reading that is reported but deliberately not a second flag;
2. **that ``a`` comes out of the measurement's own code path**, not out of a replica of it: the
   same phantom section is measured through ``get_femoral_neck_center_lee`` at several head radii
   and ``a`` scales exactly as ``1 / r``, because the notch distance it divides is a property of
   the section;
3. **that asking for ``a`` changes nothing**: the reported axis endpoint is identical with and
   without ``return_notch_distance``, on the function and through
   ``get_proximal_reference_line``;
4. **that the wrapper never raises**, whatever the landmark search does, because two call sites
   (the Augsburg scripts and ``morphometry/docker/compute_torsion.py``) run it outside any try
   block.

The real halves the gate was built for are in ``tests/characterization/test_lee_neck_qc.py``.
"""
import warnings

import nibabel as nib
import numpy as np
import pytest

from morphometry import constants as C
from morphometry.femur import (get_femoral_neck_center_lee, get_lee_neck_qc,
                               get_proximal_reference_line, lee_neck_axis_qc)
from morphometry.image_io import Segmentation, mirror_image_left_right
from tests.unit.test_murphy_neck_base import SPACING as FEMUR_SPACING
from tests.unit.test_murphy_neck_base import proximal_femur

#: The head centre of :func:`section`, as the integer voxel index the Lee path hands down.
PHANTOM_CENTRE = np.array([46, 21], dtype=np.int16)
#: The head-centre-to-notch distance :func:`section` produces, in voxels. ``a`` is this over the
#: head radius, so the two are interchangeable and the radii below place ``a`` either side of 1.
PHANTOM_NOTCH_DISTANCE = 17.46424919657298


def section() -> np.ndarray:
    """A 2-D proximal-femur-like section that is one outline and resolves a notch.

    A trochanteric lobe, a narrow neck bridge and a head, all touching, so
    ``skimage.measure.find_contours`` reports a single outline (the layer walk's first condition)
    while the bridge still leaves a dorsal gap of more than the notch search's ``thresh=5``. It is
    the section of ``tests/unit/test_femoral_neck_notch.py`` with its two one-voxel gaps closed;
    that one is deliberately three disjoint blobs, which the layer walk rejects.

    :return: The binary section as ``uint8``, indexed ``(sagittal, coronal)``.
    """
    mask = np.zeros((70, 46), dtype=np.uint8)
    mask[6:26, 6:31] = 1       # trochanteric lobe
    mask[26:33, 13:23] = 1     # the neck bridge
    mask[33:60, 9:34] = 1      # the head
    return mask


def prism(slices: int = 6) -> np.ndarray:
    """:func:`section` extruded along the axial axis, as a 3-D femur mask.

    :param slices: Number of identical axial slices.
    :return: The binary phantom as ``uint8``, indexed ``(sagittal, coronal, axial)``.
    """
    return np.stack([section()] * slices, axis=-1).astype(np.uint8)


def _image(mask: np.ndarray, spacing=(1.0, 1.0, 10.0)) -> Segmentation:
    """Wrap a phantom in a Segmentation with the given spacing (LPI index axes).

    :param mask: The phantom mask.
    :param spacing: Voxel spacing in mm.
    :return: The Segmentation.
    """
    return Segmentation.from_nibabel(
        nib.Nifti1Image(mask, np.diag([-spacing[0], -spacing[1], spacing[2], 1.0])))


def _neck_axis(mask: np.ndarray, r: float):
    """``get_femoral_neck_center_lee`` on a phantom, with its ``curve_fit`` warnings silenced.

    A phantom's contour is straight over long stretches, so ``scipy.optimize.curve_fit`` cannot
    estimate a covariance for the one-parameter slope fit and warns. The slope itself is fine and
    is all the function reads.
    :param mask: A 3-D binary femur mask.
    :param r: The femoral head radius in in-plane voxels.
    :return: ``(endpoint, a)``.
    """
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return get_femoral_neck_center_lee(mask, PHANTOM_CENTRE, r, return_notch_distance=True)


# --- the gate itself ------------------------------------------------------------------------

def test_the_threshold_is_the_head_radius():
    """The constant must stay the geometrically meaningful point, not drift into a tuned one."""
    assert C.LEE_MIN_NOTCH_DISTANCE_FACTOR == 1.0


@pytest.mark.parametrize('a', [0.3089, 0.7684, 0.8434, 0.95, 0.999])
def test_a_ring_inside_the_head_radius_is_flagged(a):
    """Below one head radius the ring is too small to lie on the peripheral neck contour."""
    assert lee_neck_axis_qc(a)['flags'] == f'lee_ring_within_head_radius:{a:.3f}'


@pytest.mark.parametrize('a', [1.0, 1.0192, 1.2171, 1.6222, 2.2406])
def test_a_ring_outside_the_head_radius_is_not_flagged(a):
    """At and above one head radius the ring is peripheral to the head, as the method intends.

    ``a == 1.0`` is the boundary and is deliberately *sound*: the test is ``a < threshold``, so
    a ring whose centre sits exactly on the head surface passes. The other values are the Augsburg
    large cohort's p1, p5, median and maximum, which must never be flagged.
    """
    assert lee_neck_axis_qc(a)['flags'] == ''


@pytest.mark.parametrize('a, wholly', [(0.3089, True), (0.7684, True), (0.8434, True),
                                       (0.8999, True), (0.9001, False), (1.5, False)])
def test_the_stricter_reading_is_reported_but_not_flagged(a, wholly):
    """The whole ring is within the head radius only below ``a = 0.9``; a metric, not a flag.

    Both readings fall inside the threshold window the constant was placed in and flag the same
    four Augsburg halves (all below 0.844), so the distinction is recorded in the dict rather than
    splitting the gate in two. The flag follows the weaker, earlier condition.
    """
    metrics = lee_neck_axis_qc(a)
    assert metrics['ring_wholly_within_head_radius'] is wholly
    assert bool(metrics['flags']) == (a < 1.0)


def test_the_metrics_dict_has_the_documented_keys_and_ring_bounds():
    """The ring factors bracket ``a`` by the published 0.1, and the voxel entries scale by ``r``."""
    metrics = lee_neck_axis_qc(0.75, head_radius=20.0)
    assert set(metrics) == {'notch_distance_factor', 'ring_inner_factor', 'ring_outer_factor',
                            'ring_wholly_within_head_radius', 'head_radius_voxels', 'notch_distance_voxels',
                            'head_clearance_voxels', 'flags'}
    assert metrics['notch_distance_factor'] == pytest.approx(0.75)
    assert metrics['ring_inner_factor'] == pytest.approx(0.65)
    assert metrics['ring_outer_factor'] == pytest.approx(0.85)
    assert metrics['head_radius_voxels'] == pytest.approx(20.0)
    assert metrics['notch_distance_voxels'] == pytest.approx(15.0)
    # negative = inside the head radius, the sign convention of murphy_neck_base_qc's clearance
    assert metrics['head_clearance_voxels'] == pytest.approx(-5.0)


def test_without_a_head_radius_the_voxel_entries_are_none_and_the_flag_still_fires():
    """``a`` is already in units of ``r``, so the gate needs no radius; the radius is reporting."""
    metrics = lee_neck_axis_qc(0.5)
    assert metrics['head_radius_voxels'] is None
    assert metrics['notch_distance_voxels'] is None
    assert metrics['head_clearance_voxels'] is None
    assert metrics['flags'] == 'lee_ring_within_head_radius:0.500'


@pytest.mark.parametrize('a', [float('nan'), float('inf'), 0.0, -1.0])
def test_a_meaningless_ring_factor_is_refused_rather_than_called_sound(a):
    """A non-finite or non-positive ``a`` compares false against any threshold.

    It must not pass silently as an unflagged axis, which is what a bare ``a < threshold`` would
    do with a NaN. ``ValueError`` is in the tuple the batch scripts catch, and the never-raising
    wrapper turns it into a flag either way.
    """
    with pytest.raises(ValueError, match='finite and positive'):
        lee_neck_axis_qc(a)


# --- `a` comes from the measurement's own path --------------------------------------------

@pytest.mark.parametrize('r', [10.0, 13.0, 16.0, 18.0, 20.0])
def test_the_ring_factor_is_the_notch_distance_over_the_head_radius(r):
    """Measured through the shipped Lee code, ``a * r`` is the section's own notch distance.

    The same section at five head radii: ``a`` scales exactly as ``1 / r``, because the ring factor
    divides a distance that belongs to the anatomy by a radius that belongs to the head fit. This
    is what makes an inflated head fit -- not a moved notch -- the thing the gate catches.
    """
    _, a = _neck_axis(prism(3), r)
    assert a * r == pytest.approx(PHANTOM_NOTCH_DISTANCE, abs=1e-9)


@pytest.mark.parametrize('r, flagged', [(16.0, False), (18.0, True)])
def test_the_gate_follows_the_head_radius_on_one_section(r, flagged):
    """One section, two head radii either side of the threshold, measured end to end.

    At ``r = 16`` the notch sits 1.09 head radii out and the ring is on the neck; at ``r = 18`` it
    sits 0.97 out and the ring has moved inside that radius. Same anatomy, same notch, so the only
    thing that changed is the head fit -- the failure mode on PA000346.
    """
    _, a = _neck_axis(prism(3), r)
    assert (a < C.LEE_MIN_NOTCH_DISTANCE_FACTOR) is flagged
    assert bool(lee_neck_axis_qc(a, head_radius=r)['flags']) is flagged


def test_asking_for_the_ring_factor_does_not_move_the_axis():
    """``return_notch_distance`` widens the return tuple and changes nothing else.

    The whole gate is worthless if it perturbs the measurement it describes, so the default path
    and the flagged path must agree bit for bit on the reported endpoint.
    """
    mask = prism(3)
    for r in (13.0, 18.0):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            plain = get_femoral_neck_center_lee(mask, PHANTOM_CENTRE, r)
        widened, _ = _neck_axis(mask, r)
        assert np.array_equal(plain, widened)
        assert plain.dtype == widened.dtype


def test_the_reference_line_is_unchanged_when_it_also_returns_the_ring_factor():
    """Same property one level up, on the function the QC wrapper and the measurement share.

    And the ``a`` it hands out is the one its own neck search produced: re-running
    ``get_femoral_neck_center_lee`` on the head centre and radius the line reports reproduces it
    exactly, which is what makes the flag a statement about the measured axis rather than about a
    second, parallel search.
    """
    image = _image(prism())
    x_ratio = abs(image.spacing[2]) / 2 * abs(image.spacing[0])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        start, end, r_fh = get_proximal_reference_line(image, side='left', method='lee',
                                                       x_ratio=x_ratio)
        start_q, end_q, r_q, a = get_proximal_reference_line(image, side='left', method='lee',
                                                             x_ratio=x_ratio,
                                                             return_notch_distance=True)
        array = np.where(image.array == 1, 1, 0)
        _, direct = get_femoral_neck_center_lee(array, np.asarray(start).astype(np.int16), r_fh,
                                                return_notch_distance=True)
    assert np.array_equal(start, start_q) and np.array_equal(end, end_q)
    assert r_q == r_fh
    assert a == direct


@pytest.mark.parametrize('method', ['murphy', 'tomczak'])
def test_the_ring_factor_is_refused_for_a_method_that_has_no_ring(method):
    """Only the Lee path builds a neck ring, so asking the others for one is a programming error."""
    with pytest.raises(AssertionError, match='method="lee"'):
        get_proximal_reference_line(_image(prism()), side='left', method=method,
                                    return_notch_distance=True)


# --- the never-raising wrapper --------------------------------------------------------------

def test_the_wrapper_flags_a_phantom_whose_ring_is_inside_the_head_radius():
    """The whole chain -- head fit, layer walk, notch, ring factor, flag -- through the wrapper.

    The phantom is a prism, so the sphere fit returns a head radius (19.3 voxels) large against the
    section's 17.5-voxel notch distance and the ring lands inside that radius. That is the *shape* of
    the defect rather than a realistic femur -- what makes it the right unit test is that no step
    is stubbed; the real halves are pinned in ``tests/characterization/test_lee_neck_qc.py``.
    """
    image = _image(prism())
    x_ratio = abs(image.spacing[2]) / 2 * abs(image.spacing[0])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        assert get_lee_neck_qc(image, 'left', x_ratio=x_ratio) == 'lee_ring_within_head_radius:0.771'


def test_the_wrapper_leaves_an_anatomical_phantom_unflagged():
    """A phantom shaped like a femur rather than like a prism must come back clean.

    The counterpart of the test above, and the reason the gate is not simply "everything is
    flagged": the proximal-femur phantom of ``tests/unit/test_murphy_neck_base.py`` (head sphere,
    neck cylinder, shaft, greater trochanter) puts its notch 2.56 head radii out, well clear of the
    threshold. Imported rather than copied so the two gates are exercised on the same anatomy.
    """
    image = _image(proximal_femur((9,)), spacing=FEMUR_SPACING)
    x_ratio = abs(image.spacing[2]) / 2 * abs(image.spacing[0])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        _, _, r_fh, a = get_proximal_reference_line(image, side='left', method='lee',
                                                    x_ratio=x_ratio, return_notch_distance=True)
        assert get_lee_neck_qc(image, 'left', x_ratio=x_ratio) == ''
    assert a == pytest.approx(2.563171, abs=1e-5)
    assert a > C.LEE_MIN_NOTCH_DISTANCE_FACTOR
    assert lee_neck_axis_qc(a, head_radius=r_fh)['head_clearance_voxels'] > 0


def test_the_wrapper_is_mirror_invariant_on_a_phantom():
    """A left half and its mirror measured as a right half give the same flag string.

    ``get_proximal_reference_line`` measures a right image half as the reflection of a left one and
    ``a`` is a ratio of two distances, both of which a reflection preserves, so this is a property
    of the construction. It is checked on the real PA000001 hip as well.
    """
    image = _image(prism())
    x_ratio = abs(image.spacing[2]) / 2 * abs(image.spacing[0])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        left = get_lee_neck_qc(image, 'left', x_ratio=x_ratio)
        mirrored = get_lee_neck_qc(mirror_image_left_right(image), 'right', x_ratio=x_ratio)
    assert left == mirrored == 'lee_ring_within_head_radius:0.771'


@pytest.mark.parametrize('spacing', [(1.0, 1.4, 10.0), (1.4, 1.0, 10.0)])
def test_the_wrapper_refuses_in_plane_anisotropic_voxels(spacing):
    """Anisotropic in-plane voxels must come back as a failure, not as a clean axis.

    ``a`` is dimensionless but is a ratio of two *index-space* quantities, so an in-plane shear
    moves it: on this phantom a 1.4 : 1 grid takes it from 0.771 to 1.167 and flips the verdict,
    where an isotropic refinement to 0.5 mm moves it only to 0.714 and does not. The measurement
    refuses such data outright (``measurements/femur.py::calculate_femoral_torsion`` validates this
    same hip image's spacing), so reporting ``''`` here would put "the axis looks sound" beside a
    NaN angle -- and in the same ``qc_flags`` cell as a Murphy family already saying
    ``murphy_landmark_failed:anisotropic_in_plane``.
    """
    image = _image(prism(), spacing=spacing)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        flags = get_lee_neck_qc(image, 'left', x_ratio=abs(spacing[2]) / 2 * abs(spacing[0]))
    assert flags == 'lee_landmark_failed:anisotropic_in_plane'


@pytest.mark.parametrize('mask, reason', [
    (np.zeros((20, 20, 6), dtype=np.uint8), 'lee_landmark_failed:'),
    (np.ones((20, 20, 6), dtype=np.uint8), 'lee_landmark_failed:'),
])
def test_the_wrapper_reports_a_failed_landmark_instead_of_raising(mask, reason):
    """A QC helper must never cost its caller a measurement.

    Two call sites run it outside any try block, so an empty mask (no femur to fit a head to) and a
    solid block (no notch, no single-outline layer) must both come back as a flag string.
    """
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        flags = get_lee_neck_qc(_image(mask), 'left', x_ratio=5.0)
    assert flags.startswith(reason)
    assert ';' not in flags and ',' not in flags  # safe in the CSV columns


def test_a_lee_landmark_failure_is_named_not_just_typed():
    """The Lee path's own failure modes carry reason codes, like the Murphy ones.

    A solid block has no layer the walk accepts, which is the Lee analogue of
    ``murphy_landmark_failed:trochanter_implausible`` -- a named cause rather than a bare
    ``ValueError``, so a cohort's NaN rows can be counted by reason.
    """
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        flags = get_lee_neck_qc(_image(np.ones((20, 20, 6), dtype=np.uint8)), 'left', x_ratio=5.0)
    assert flags == 'lee_landmark_failed:neck_layer_not_found'
