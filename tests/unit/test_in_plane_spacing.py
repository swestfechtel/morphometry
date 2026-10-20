"""The equal-in-plane-spacing precondition of the index-space axial angles (backlog task #10).

The femoral torsion, the tibial torsion and the knee rotation angle all compose their angle in
the axial ``(x, y)`` **index** plane (``geometry.signed_axial_angle``). That is the anatomical
angle only if the two in-plane voxel edges have the same length: at spacing ``(sx, sy)`` a
physical direction ``(cos t, sin t)`` has index direction ``(cos t / sx, sin t / sy)``, so the
index angle is ``atan2(k sin t, cos t)`` with ``k = sx / sy`` and the angle *between* two lines
comes out sheared by an amount that depends on how the limb happens to lie in the slice.

These tests do three things:

1. demonstrate the defect on the anisotropy that actually exists in this project's data
   (``dfg_knees``: 0.75 x 0.3125 mm in plane after the LPI reorientation, ``k = 2.4``), where a
   true 15 degree angle reads anywhere between 6.3 and 35.1 degrees. **No sheared value has ever
   been produced in this repository**: the only non-test callers of the knee rotation angle run on
   Augsburg (square 1.09375 mm), the ``dfg_knees`` scripts are cartilage-only, and ``dfg_knees``
   has no ankle label, so the tibial torsion cannot reach it either. The defect is what *would*
   happen if these measurements were pointed at that cohort;
2. pin that ``geometry.validate_in_plane_isotropy`` accepts every spacing the measured cohorts
   present (Augsburg 1.09375, NaKo 1.0, CT 0.8 -- all stored with the two in-plane zooms
   bit-identical) and refuses the ``dfg_knees`` one with a message naming the property;
3. pin that the check is a pure no-op on conforming data: the knee rotation angle of a phantom
   is *bit-identical* with ``spacing=None`` (unchecked, the legacy call) and with a conforming
   spacing passed in.
"""
import numpy as np
import pytest
from skimage.draw import ellipse

from morphometry import constants as C
from morphometry import geometry as G
from morphometry.image_io import Segmentation
from morphometry.measurements.femur import (calculate_femoral_torsion, calculate_femoral_torsion_ct,
                                            get_femoral_torsion_landmarks_ct)
from morphometry.measurements.hip import calculate_anteversion
from morphometry.measurements.knee import calculate_knee_rotation_angle
from morphometry.measurements.tibia import calculate_tibial_torsion, get_tibial_torsion_landmarks

#: The in-plane spacing of the ``dfg_knees`` knee MRI after the LPI reorientation, measured over
#: all 96 scans: the 0.75 mm sagittal slice axis becomes an in-plane axis (axis 0 on 94 scans,
#: axis 1 on K10_90_belastet), giving ratio 2.4 on 95 of them. K11_90_belastet is 1 mm isotropic.
DFG_KNEES_SPACING = (0.75, 0.3125, 0.3125)
#: Spacings that every other cohort in reach presents (both in-plane zooms bit-identical).
CONFORMING_SPACINGS = [
    (1.09375, 1.09375, 10.0),   # Augsburg hip MRI
    (1.09375, 1.09375, 4.0),    # Augsburg knee / ankle MRI
    (1.0, 1.0, 1.0),            # NaKo hip MRI (isotropic)
    (0.8, 0.8, 3.0),            # NMDID whole-leg CT
]


# --------------------------------------------------------------------------- the defect itself

def _index_angle(true_angle_deg: float, reference_deg: float, k: float) -> float:
    """Angle the index-space composition returns for a known physical angle on anisotropic voxels.

    Both lines are taken as exact physical directions and divided by the voxel spacing, which is
    what a landmark search finds when its landmarks sit at voxel centres. The angles are measured
    in the sense :func:`morphometry.geometry.signed_axial_angle` calls positive on a left image
    half, i.e. from medial (+x) towards anterior (-y), so a conforming spacing returns
    ``true_angle_deg`` itself.
    :param true_angle_deg: The true (physical) angle from the reference line to the measured line.
    :param reference_deg: Orientation of the reference line in the physical axial plane, i.e. how
        the limb happens to lie in the slice.
    :param k: In-plane anisotropy ``sx / sy``.
    :return: The composed angle in degrees, as :func:`morphometry.geometry.signed_axial_angle`
        returns it (folded into (-90, 90]).
    """
    t_ref, t_meas = np.radians(reference_deg), np.radians(reference_deg + true_angle_deg)
    ref = np.array([np.cos(t_ref) / k, -np.sin(t_ref), 0.0])
    meas = np.array([np.cos(t_meas) / k, -np.sin(t_meas), 0.0])
    return G.signed_axial_angle(ref, meas, 'left')


def test_index_composition_is_exact_when_in_plane_spacing_is_equal():
    """With equal in-plane voxel edges the index-space angle IS the physical angle."""
    for reference in range(0, 180, 5):
        assert _index_angle(15.0, reference, k=1.0) == pytest.approx(15.0, abs=1e-9)


def test_index_composition_is_sheared_on_dfg_knees_anisotropy():
    """At the ``dfg_knees`` k = 2.4 a true 15 degree angle reads 6.3 to 35.1 degrees.

    The error depends on the limb's orientation in the slice, not only on the anisotropy, so it
    cannot be corrected after the fact by a per-dataset factor.
    """
    k = DFG_KNEES_SPACING[0] / DFG_KNEES_SPACING[1]
    assert k == pytest.approx(2.4)
    composed = np.array([_index_angle(15.0, r, k) for r in np.arange(0.0, 180.0, 0.25)])
    assert composed.min() == pytest.approx(6.28, abs=0.02)
    assert composed.max() == pytest.approx(35.07, abs=0.02)
    assert np.abs(composed - 15.0).max() > 20.0


@pytest.mark.parametrize("k,true_angle,expected_max_error", [
    (1.001, 45.0, 0.041), (1.01, 45.0, 0.405), (1.05, 30.0, 1.428), (1.1, 30.0, 2.845),
    (1.2, 15.0, 2.955), (1.5, 15.0, 7.342), (2.0, 15.0, 14.503), (2.4, 30.0, 35.488),
])
def test_shear_error_grows_with_anisotropy(k, true_angle, expected_max_error):
    """Worst-case error over all limb orientations, as documented on ``IN_PLANE_SPACING_RTOL``."""
    composed = np.array([_index_angle(true_angle, r, k) for r in np.arange(0.0, 180.0, 0.05)])
    assert np.abs(composed - true_angle).max() == pytest.approx(expected_max_error, abs=0.01)


def test_tolerance_is_tight_enough_to_make_the_shear_irrelevant():
    """At the accepted tolerance the shear is below 0.06 degrees for ANY true angle.

    The error grows with the true angle, so probing 45 degrees would understate it (real values
    exceed 45: PA000405's Murphy torsion is 55.3 degrees). The supremum is approached as the true
    angle tends to 90, where the folded angle can also cross the (-90, 90] seam -- hence the stop
    at 89.9.
    """
    k = 1.0 + C.IN_PLANE_SPACING_RTOL
    worst = max(abs(_index_angle(true, r, k) - true)
                for true in np.arange(5.0, 89.95, 2.5)
                for r in np.arange(0.0, 180.0, 1.0))
    assert worst == pytest.approx(0.0573, abs=0.002)
    assert worst < 0.06


# ---------------------------------------------------------------------------- the validator

@pytest.mark.parametrize("spacing", CONFORMING_SPACINGS)
def test_validator_accepts_every_measured_cohort_spacing(spacing):
    assert G.validate_in_plane_isotropy(spacing, 'test') is None


def test_validator_accepts_none_and_leaves_the_precondition_to_the_caller():
    assert G.validate_in_plane_isotropy(None, 'test') is None


def test_validator_ignores_the_slice_axis_and_the_sign_of_the_spacing():
    """Only the first two entries matter, and a negative (direction-carrying) zoom is fine."""
    assert G.validate_in_plane_isotropy((-1.0, -1.0, -99.0), 'test') is None
    assert G.validate_in_plane_isotropy((1.0, 1.0, 1.0, 1.0), 'test') is None  # 4D header zooms


def test_validator_accepts_float_noise_within_the_tolerance():
    base = 0.3125
    G.validate_in_plane_isotropy((base, base * (1 + 0.5 * C.IN_PLANE_SPACING_RTOL), 1.0), 'test')


@pytest.mark.parametrize("spacing,reason", [
    ((0.0, 1.0, 1.0), 'a zero in-plane edge: the ratio is undefined (used to be ZeroDivisionError)'),
    ((0.0, 0.0, 1.0), 'both in-plane edges zero: equal, but a degenerate index-to-physical map'),
    ((-1.0, 0.0, 1.0), 'a zero edge behind an abs()'),
    ((float('nan'), 1.0, 1.0), 'nan: every comparison is False, so it used to be ACCEPTED'),
    ((1.0, float('nan'), 1.0), 'nan in the second entry'),
    ((float('inf'), 1.0, 1.0), 'an infinite edge'),
    ((1.0,), 'only one entry: used to be a bare IndexError'),
    ((), 'no entries at all'),
    (5.0, 'not a sequence at all: used to be a bare TypeError'),
])
def test_validator_refuses_malformed_spacing_as_a_value_error(spacing, reason):
    """Every rejection must be a ``ValueError``, so the per-measurement handlers catch it.

    ``compute_torsion.py`` and ``scripts/process_augsburg_*`` catch only
    ``(RuntimeError, AssertionError, ValueError)``, so a ``ZeroDivisionError``, ``IndexError`` or
    ``TypeError`` escaping this check would abort a cohort run -- exactly the failure mode the
    ``ValueError`` choice was made to avoid. A silently accepted ``nan`` is just as bad: it would
    let the sheared angle through.
    """
    with pytest.raises(ValueError):
        G.validate_in_plane_isotropy(spacing, 'test')


def test_validator_refuses_just_outside_the_tolerance():
    base = 1.0
    with pytest.raises(ValueError):
        G.validate_in_plane_isotropy((base, base * (1 + 2 * C.IN_PLANE_SPACING_RTOL), 1.0), 'test')


def test_validator_refuses_dfg_knees_naming_the_property_and_the_source():
    with pytest.raises(ValueError) as excinfo:
        G.validate_in_plane_isotropy(DFG_KNEES_SPACING, 'knee rotation angle segmentation')
    message = str(excinfo.value)
    assert 'knee rotation angle segmentation' in message
    assert 'in-plane voxel spacing is anisotropic' in message
    assert '0.75' in message and '0.3125' in message and '2.4' in message


# ------------------------------------------------------------------- end to end on a phantom

def _two_lobe_section(shape, centre, half_gap, angle_deg, radii=(11, 6)):
    """Rasterise two ellipse "condyles" straddling a gap, the pair rotated by ``angle_deg``.

    :param shape: ``(nx, ny)`` of the section.
    :param centre: Centre of the pair in ``(x, y)`` index coordinates.
    :param half_gap: Half the distance between the two lobe centres, in voxels.
    :param angle_deg: Rotation of the lobe-to-lobe line, positive from ``+x`` towards ``+y``.
    :param radii: ``(rx, ry)`` of each lobe.
    :return: The ``uint8`` section mask.
    """
    section = np.zeros(shape, dtype=np.uint8)
    rad = np.radians(angle_deg)
    direction = np.array([np.cos(rad), np.sin(rad)])
    for sign in (+1, -1):
        cx, cy = np.asarray(centre, dtype=float) + sign * half_gap * direction
        rr, cc = ellipse(cx, cy, radii[0], radii[1], shape=shape)
        section[rr, cc] = 1
    return section


#: The knee-rotation angle the phantom below produces. Its designed mismatch is 12 degrees and
#: the composition's sign convention makes that negative on a left image half; the value is not
#: exactly -12 because the two lobe pairs are rasterised onto the voxel grid and the condylar
#: search takes the posterior-most point on either side of the divider, not the lobe centres.
#: Pinned so that a degenerate search (which would return 0.0, or the same number for both bones)
#: cannot pass the bit-identity test below.
PHANTOM_KNEE_ROTATION = -12.200468727380784


@pytest.fixture
def knee_phantom():
    """A knee phantom with a femoral and a tibial condyle pair at a known rotation mismatch.

    Femur (label 1) occupies the superior slices and tibia (label 2) the inferior ones; the
    tibial condylar line is turned by 12 degrees relative to the femoral one. Neither section
    resolves an intercondylar gap at the threshold ``find_notch`` uses, so what is exercised here
    is ``knee.get_condylar_divider``'s centroid-divider fallback geometry -- which is fine for
    this file's purpose (the spacing check sits in front of the landmark search and is
    indifferent to which divider it used), and is pinned by
    ``tests/unit/test_knee_reference_line.py`` in its own right.
    :return: The ``(90, 90, 20)`` label volume.
    """
    shape = (90, 90)
    volume = np.zeros(shape + (20,), dtype=np.uint8)
    femur = _two_lobe_section(shape, (45, 45), 16, 0.0)
    tibia = _two_lobe_section(shape, (45, 50), 16, 12.0)
    for z in range(0, 10):
        volume[:, :, z] = femur
    for z in range(10, 20):
        volume[:, :, z] = np.where(tibia > 0, 2, 0)
    return volume


@pytest.mark.parametrize("side", ["left", "right"])
def test_knee_rotation_angle_is_bit_identical_with_a_conforming_spacing(knee_phantom, side):
    """Passing a conforming spacing only adds a check: the returned float is unchanged.

    The reference value is pinned rather than merely asserted finite, so a landmark search that
    degenerated into returning the same line for both bones (0.0) would be caught here instead of
    making the bit-identity claim vacuous. ``side='right'`` holds the mirror-image knee, hence the
    sign flip.
    """
    expected = PHANTOM_KNEE_ROTATION if side == 'left' else -PHANTOM_KNEE_ROTATION
    reference = calculate_knee_rotation_angle(knee_phantom, 1, 2, side=side)
    assert reference == pytest.approx(expected, abs=1e-9)
    for spacing in CONFORMING_SPACINGS:
        assert calculate_knee_rotation_angle(knee_phantom, 1, 2, side=side,
                                             spacing=spacing) == reference


def test_knee_rotation_angle_refuses_dfg_knees_spacing(knee_phantom):
    """The one in-plane anisotropic cohort in reach now fails loudly instead of shearing."""
    with pytest.raises(ValueError, match='in-plane voxel spacing is anisotropic'):
        calculate_knee_rotation_angle(knee_phantom, 1, 2, side='left', spacing=DFG_KNEES_SPACING)


def test_knee_rotation_angle_takes_the_spacing_from_an_image(knee_phantom):
    """Handing over the ``Segmentation`` instead of ``.array`` checks the spacing with no kwarg.

    This is the point of accepting an image: the check is otherwise opt-in on exactly the paths
    where it could fire (the bare-array knee entry points), and ``dfg_knees`` -- the one
    violating cohort -- is a knee dataset.
    """
    conforming = _labelled_segmentation(knee_phantom, (1.09375, 1.09375, 4.0))
    assert (calculate_knee_rotation_angle(conforming, 1, 2, side='left')
            == calculate_knee_rotation_angle(knee_phantom, 1, 2, side='left'))
    anisotropic = _labelled_segmentation(knee_phantom, DFG_KNEES_SPACING)
    with pytest.raises(ValueError, match='in-plane voxel spacing is anisotropic'):
        calculate_knee_rotation_angle(anisotropic, 1, 2, side='left')


def test_tibial_torsion_and_femoral_torsion_take_the_spacing_from_an_image():
    """Same for the tibial torsion's two masks and the MRI femoral torsion's knee mask."""
    empty = np.zeros((20, 20, 20), dtype=np.uint8)
    bad = _labelled_segmentation(empty, DFG_KNEES_SPACING)
    good = _labelled_segmentation(empty, (1.0, 1.0, 4.0))
    with pytest.raises(ValueError, match='knee mask: in-plane voxel spacing is anisotropic'):
        calculate_tibial_torsion(bad, good, 2, 1, 2, side='left')
    with pytest.raises(ValueError, match='ankle mask: in-plane voxel spacing is anisotropic'):
        calculate_tibial_torsion(good, bad, 2, 1, 2, side='left')
    with pytest.raises(ValueError, match='knee mask: in-plane voxel spacing is anisotropic'):
        get_tibial_torsion_landmarks(bad, good, 2, 1, 2, side='left')
    with pytest.raises(ValueError, match='knee mask: in-plane voxel spacing is anisotropic'):
        calculate_femoral_torsion(_nifti_segmentation((1.0, 1.0, 4.0)), bad, side='left')


def test_an_explicit_spacing_wins_over_an_images_own(knee_phantom):
    """Documented precedence, and the only way to override a wrong header."""
    anisotropic = _labelled_segmentation(knee_phantom, DFG_KNEES_SPACING)
    assert (calculate_knee_rotation_angle(anisotropic, 1, 2, side='left', spacing=(1.0, 1.0, 1.0))
            == calculate_knee_rotation_angle(knee_phantom, 1, 2, side='left'))


def _nifti_segmentation(spacing):
    """A tiny empty ``Segmentation`` whose header carries ``spacing``.

    :param spacing: ``(sx, sy, sz)`` voxel edge lengths in mm.
    :return: The ``Segmentation`` (LPI affine, all-zero mask).
    """
    import nibabel as nib
    affine = np.diag([-spacing[0], -spacing[1], -spacing[2], 1.0])
    return Segmentation.from_nibabel(nib.Nifti1Image(np.zeros((20, 20, 20), dtype=np.uint8), affine))


def _labelled_segmentation(array, spacing):
    """Wrap a label array in a ``Segmentation`` whose header carries ``spacing``.

    :param array: The label volume.
    :param spacing: ``(sx, sy, sz)`` voxel edge lengths in mm.
    :return: The ``Segmentation`` (LPI affine).
    """
    import nibabel as nib
    affine = np.diag([-spacing[0], -spacing[1], -spacing[2], 1.0])
    return Segmentation.from_nibabel(nib.Nifti1Image(np.asarray(array), affine))


def test_anteversion_checks_its_hip_image():
    """The neck angle is a single-line index angle, so it takes the shear with no cancellation."""
    with pytest.raises(ValueError, match='anteversion hip image: in-plane voxel spacing is '
                                         'anisotropic'):
        calculate_anteversion(_nifti_segmentation(DFG_KNEES_SPACING))
    # conforming: past the check, then fails on the empty phantom
    with pytest.raises(ValueError) as excinfo:
        calculate_anteversion(_nifti_segmentation((1.0, 1.0, 1.0)))
    assert 'in-plane voxel spacing' not in str(excinfo.value)


@pytest.mark.parametrize("side", ["left", "right"])
def test_mri_femoral_torsion_checks_the_hip_image(side):
    """The hip half is an ``Image``, so its spacing is checked without the caller doing anything."""
    knee = np.zeros((20, 20, 20), dtype=np.uint8)
    with pytest.raises(ValueError, match='hip image: in-plane voxel spacing is anisotropic'):
        calculate_femoral_torsion(_nifti_segmentation(DFG_KNEES_SPACING), knee, side=side)


def test_mri_femoral_torsion_checks_the_knee_mask_when_its_spacing_is_given():
    """The knee mask is a bare array; the check reaches it only through ``knee_spacing``."""
    knee = np.zeros((20, 20, 20), dtype=np.uint8)
    hip = _nifti_segmentation((1.0, 1.0, 4.0))
    with pytest.raises(ValueError, match='knee mask: in-plane voxel spacing is anisotropic'):
        calculate_femoral_torsion(hip, knee, side='left', knee_spacing=DFG_KNEES_SPACING)
    # a conforming pair gets past the check and fails later, on the empty phantom: the landmark
    # search raises its own ValueError ('zero-size array to reduction operation minimum'), which
    # is what shows the check itself was a no-op
    with pytest.raises(ValueError) as excinfo:
        calculate_femoral_torsion(hip, knee, side='left', knee_spacing=(1.0, 1.0, 4.0))
    assert 'in-plane voxel spacing' not in str(excinfo.value)


@pytest.mark.parametrize("func", [calculate_femoral_torsion_ct, get_femoral_torsion_landmarks_ct])
def test_ct_femoral_torsion_checks_both_images(func):
    """Both CT inputs are ``Segmentation`` objects, so neither needs a spacing argument."""
    good, bad = _nifti_segmentation((0.8, 0.8, 3.0)), _nifti_segmentation((0.8, 1.6, 3.0))
    with pytest.raises(ValueError, match='femur image: in-plane voxel spacing is anisotropic'):
        func(bad, good, side='left')
    with pytest.raises(ValueError, match='knee image: in-plane voxel spacing is anisotropic'):
        func(good, bad, side='left')


@pytest.mark.parametrize("func", [calculate_tibial_torsion, get_tibial_torsion_landmarks])
@pytest.mark.parametrize("which", ["knee", "ankle"])
def test_tibial_torsion_checks_each_supplied_spacing(func, which):
    """Both masks are bare arrays, so each is checked only via its own ``*_spacing`` argument."""
    mask = np.zeros((20, 20, 20), dtype=np.uint8)
    kwargs = {'knee_spacing': (1.0, 1.0, 4.0), 'ankle_spacing': (1.0, 1.0, 4.0)}
    kwargs[f'{which}_spacing'] = DFG_KNEES_SPACING
    with pytest.raises(ValueError, match=f'{which} mask: in-plane voxel spacing is anisotropic'):
        func(mask, mask, 2, 1, 2, side='left', **kwargs)
