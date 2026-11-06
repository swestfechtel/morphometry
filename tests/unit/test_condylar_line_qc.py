"""Unit tests for the posterior condylar reference-line QC gate, on phantoms (no image data).

``morphometry.knee.get_knee_reference_line`` returns two posterior-condylar endpoints whose
in-plane **direction** is the entire anatomical content of the femoral torsion, the tibial torsion
and the knee rotation angle. A short line is therefore badly conditioned in the one quantity that
is used -- one voxel of landmark error turns a 9.5-voxel line by 6.0 degrees -- while the angle it
produces looks perfectly ordinary, and ``knee.knee_condylar_line_qc`` flags it as
``<bone>_condylar_line_short`` without changing anything.

Six things are pinned here:

1. **the flag and its boundary.** The threshold is ``1 / tan(C.CONDYLAR_LINE_ANGLE_TOLERANCE_DEG)``
   in-plane voxels, derived from the tolerance rather than fitted, and the comparison is strict, so
   a line of exactly that length is sound. The derivation is pinned the other way round as well:
   at the threshold length, one voxel of error tilts the line by exactly the tolerance.
2. **that the threshold is in voxels while the flag prints millimetres**, which is the one place
   this gate can mislead: two cohorts with the same voxel count and different spacings get the same
   verdict and different printed values. Both figures are in the dict.
3. **that the flag name carries the bone.** One leg has two condylar lines and both land in the
   same ``qc_flags`` cell, so an unprefixed name would be ambiguous exactly where it matters. The
   threshold is nonetheless shared: a landmark error of one voxel is the same error on either bone.
4. **the in-plane isotropy precondition** (backlog task #10). A count of in-plane voxels is only a
   length where the two in-plane voxel edges agree, so a known anisotropic spacing is refused with
   ``ValueError`` -- the same class the batch scripts' per-measurement handlers catch -- and an
   unknown one (``spacing=None``) is trusted to the caller.
5. **that the wrapper describes the line the measurement uses**, which on a right image half means
   the *reflected* one: all three consumers reflect, the search is only approximately
   mirror-equivariant, and on the real leg that motivates the gate the two readings differ by 4.1
   voxels (see ``tests/characterization/test_condylar_line_qc.py``). Mirroring a half and asking
   for the other side double-reflects to a bitwise identical array, so the flag cannot differ at
   all -- the same structural argument ``femur.get_lee_neck_qc`` rests on.
6. **that the wrapper never raises**, whatever the search does, because the call sites (the two
   Augsburg scripts) run it outside any try block.

The real halves the gate was built for are in ``tests/characterization/test_condylar_line_qc.py``.
"""
import math

import numpy as np
import pytest
from skimage.draw import disk, polygon

from morphometry import constants as C
from morphometry.image_io import Segmentation
from morphometry.knee import (get_condylar_line_qc, get_knee_reference_line,
                              knee_condylar_line_qc)

#: The gate's minimum line length in in-plane voxels, as the implementation derives it.
THRESHOLD_VOXELS = 1.0 / math.tan(math.radians(C.CONDYLAR_LINE_ANGLE_TOLERANCE_DEG))
#: An isotropic in-plane spacing with a thick slice axis, like the Augsburg knee MRI.
SPACING = (1.09375, 1.09375, 4.0)
SHAPE = (110, 80)


def line(length_voxels: float, layer: int = 1, tilt_deg: float = 20.0):
    """Two endpoints a given in-plane distance apart, at an arbitrary in-plane bearing.

    The bearing is deliberately not axis-aligned, so a length computed from one coordinate
    difference rather than the in-plane norm would not reproduce it.
    :param length_voxels: The in-plane distance between the two points, in voxels.
    :param layer: The slice index both points share (the reference line is one section's).
    :param tilt_deg: Bearing of the line in the axial plane, in degrees.
    :return: ``(start, end)`` as float ``(sagittal, coronal, layer)`` index triples.
    """
    a = math.radians(tilt_deg)
    start = np.array([30.0, 40.0, float(layer)])
    end = start + np.array([length_voxels * math.cos(a), length_voxels * math.sin(a), 0.0])
    return start, end


def notched_plateau(shape=SHAPE) -> np.ndarray:
    """Two condyles joined by an anterior bridge: a section whose notch ``find_notch`` resolves.

    The same construction as ``tests/unit/test_knee_reference_line.py``'s phantom, which gives a
    long, well-conditioned condylar line -- the sound case this gate must leave alone.
    :param shape: Shape of the section.
    :return: The ``uint8`` 2-D mask.
    """
    mask = np.zeros(shape, dtype=np.uint8)
    for cx in (32, 78):
        rr, cc = disk((cx, 48), 17, shape=shape)
        mask[rr, cc] = 1
    rr, cc = polygon(np.array([18, 92, 92, 18]), np.array([18, 18, 44, 44]), shape)
    mask[rr, cc] = 1
    return mask


def stub_condyles(shape=SHAPE) -> np.ndarray:
    """A section whose two condyles are so close together that the reference line is short.

    The degenerate case: two small discs a few voxels apart, which is geometrically what a
    field-of-view-edge section of a femur looks like once most of the condyles have been cut off.
    :param shape: Shape of the section.
    :return: The ``uint8`` 2-D mask.
    """
    mask = np.zeros(shape, dtype=np.uint8)
    for cx in (52, 58):
        rr, cc = disk((cx, 48), 3, shape=shape)
        mask[rr, cc] = 1
    return mask


def volume(section: np.ndarray, label: int = 1, layers: int = 3) -> np.ndarray:
    """Embed a 2-D section in a small 3-D stack whose middle layer is the widest.

    :param section: The 2-D section.
    :param label: The label to write the section with.
    :param layers: Number of layers in the stack.
    :return: A 3-D ``uint8`` label mask; ``get_layer_with_biggest_convex_area`` selects layer 1.
    """
    out = np.zeros(section.shape + (layers,), dtype=np.uint8)
    out[:, :, 1] = section * label
    return out


def as_image(array: np.ndarray, spacing=SPACING) -> Segmentation:
    """Wrap a label array in a Segmentation with a diagonal affine of the given spacing.

    :param array: The label array.
    :param spacing: The voxel spacing to give it.
    :return: The Segmentation.
    """
    import nibabel as nib
    affine = np.diag([-spacing[0], -spacing[1], -spacing[2], 1.0])
    return Segmentation.from_nibabel(nib.Nifti1Image(array.astype(np.uint8), affine))


# --- the flag and its boundary ----------------------------------------------------------------

def test_a_long_line_is_sound_and_a_short_one_is_flagged():
    """Either side of the threshold, with the flag naming the bone and printing millimetres."""
    assert knee_condylar_line_qc(*line(40.0), spacing=SPACING, bone='femur')['flags'] == ''
    assert knee_condylar_line_qc(*line(9.0), spacing=SPACING,
                                 bone='femur')['flags'] == 'femoral_condylar_line_short:9.00'


def test_the_boundary_is_the_tolerance_and_the_comparison_is_strict():
    """A line of exactly ``1 / tan(tolerance)`` voxels is sound; one hair shorter is not.

    The gate reads ``length < threshold``, so the threshold length itself passes. Pinned because
    the alternative (``<=``) is a one-character change that would silently alter the verdict of any
    line landing exactly on it.
    """
    assert knee_condylar_line_qc(*line(THRESHOLD_VOXELS), spacing=SPACING, bone='tibia')['flags'] == ''
    assert knee_condylar_line_qc(*line(THRESHOLD_VOXELS * (1 - 1e-9)), spacing=SPACING,
                                 bone='tibia')['flags'] != ''


def test_the_threshold_is_the_tolerance_read_backwards():
    """At the threshold length, one in-plane voxel of landmark error tilts the line by the tolerance.

    This is the whole derivation of the constant, pinned in the direction that makes it falsifiable:
    the threshold is not a fitted length but the length at which ``angle_per_voxel_deg`` reaches
    ``C.CONDYLAR_LINE_ANGLE_TOLERANCE_DEG``.
    """
    at = knee_condylar_line_qc(*line(THRESHOLD_VOXELS), spacing=SPACING, bone='femur')
    assert at['angle_per_voxel_deg'] == pytest.approx(C.CONDYLAR_LINE_ANGLE_TOLERANCE_DEG, abs=1e-9)
    # and the quantity is the conditioning it claims to be: 9.5 voxels -> 6 degrees per voxel
    assert knee_condylar_line_qc(*line(9.49), spacing=SPACING, bone='femur')[
        'angle_per_voxel_deg'] == pytest.approx(6.0, abs=0.05)


def test_a_degenerate_line_is_flagged_rather_than_refused():
    """A zero-length line is still the line the measurement used, so it is marked, not rejected.

    The gate's contract is that it never costs a caller a measurement. A length of zero is the
    extreme of the thing being flagged, and ``angle_per_voxel_deg`` is then 90 degrees rather than
    a division by zero.
    """
    metrics = knee_condylar_line_qc(*line(0.0), spacing=SPACING, bone='femur')
    assert metrics['flags'] == 'femoral_condylar_line_short:0.00'
    assert metrics['angle_per_voxel_deg'] == pytest.approx(90.0)


def test_the_two_endpoints_are_interchangeable():
    """Nothing here reads the endpoint order, which ``order_condylar_points`` normalises away."""
    start, end = line(9.0)
    assert (knee_condylar_line_qc(start, end, spacing=SPACING, bone='femur')
            == knee_condylar_line_qc(end, start, spacing=SPACING, bone='femur'))


# --- units: voxels decide, millimetres are printed ---------------------------------------------

def test_the_flag_value_is_voxels_and_so_does_not_depend_on_the_spacing():
    """Same voxel count, four spacings, one flag string -- and the millimetre figures all differ.

    The flag reports the quantity the gate tests, so a flagged value is always below the 14.30
    threshold and an unflagged one always at or above it, on every cohort and with no spacing
    needed to read it. This deliberately departs from ``murphy_axis_short``, which prints
    millimetres: the Augsburg large cohort carries four in-plane spacings, so in millimetres the
    printed value is **not monotone in the verdict** -- 51 of its unflagged lines are shorter in
    millimetres than its longest flagged one. Pinned because the alternative passes every other
    test in this file.
    """
    variants = [knee_condylar_line_qc(*line(9.0), spacing=(s, s, 4.0), bone='femur')
                for s in (0.5, 1.09375, 1.640625, 2.0)]
    assert {v['flags'] for v in variants} == {'femoral_condylar_line_short:9.00'}
    assert all(v['line_length_voxels'] == pytest.approx(9.0) for v in variants)
    assert sorted(v['line_length_mm'] for v in variants) == pytest.approx([4.5, 9.84375, 14.765625, 18.0])


def test_without_a_spacing_only_the_millimetre_figure_is_lost():
    """``spacing=None``: no millimetre figure, and the flag is unchanged.

    The condylar searches historically take bare arrays, so a caller may genuinely hold no
    geometry. Because the flag prints voxels, such a call loses the in-plane isotropy precondition
    and the physical length but never the verdict's meaning -- there is no second unit system
    hiding under one flag name.
    """
    metrics = knee_condylar_line_qc(*line(9.0), bone='tibia')
    assert metrics['line_length_mm'] is None
    assert metrics['flags'] == 'tibial_condylar_line_short:9.00'
    assert metrics['flags'] == knee_condylar_line_qc(*line(9.0), spacing=SPACING,
                                                     bone='tibia')['flags']


# --- the bone prefix --------------------------------------------------------------------------

def test_the_flag_names_the_bone_but_the_threshold_does_not_depend_on_it():
    """Both bones share one threshold and differ only in the flag's name.

    The two length distributions differ substantially on real data (Augsburg median 45.10 voxels
    femoral against 29.65 tibial), so a per-bone threshold would be fitting to a distribution --
    which is exactly what deriving the threshold from the tolerance avoids.
    """
    femoral = knee_condylar_line_qc(*line(9.0), spacing=SPACING, bone='femur')
    tibial = knee_condylar_line_qc(*line(9.0), spacing=SPACING, bone='tibia')
    assert femoral['flags'].startswith('femoral_condylar_line_short:')
    assert tibial['flags'].startswith('tibial_condylar_line_short:')
    assert femoral['flags'].split(':')[1] == tibial['flags'].split(':')[1]
    for length in (THRESHOLD_VOXELS * 0.999, THRESHOLD_VOXELS, THRESHOLD_VOXELS * 1.001):
        assert (bool(knee_condylar_line_qc(*line(length), spacing=SPACING, bone='femur')['flags'])
                == bool(knee_condylar_line_qc(*line(length), spacing=SPACING, bone='tibia')['flags']))


def test_an_unknown_bone_is_refused():
    """The pure verdict has no default bone to fall back on: the name would be wrong, not missing."""
    with pytest.raises(ValueError, match='femur'):
        knee_condylar_line_qc(*line(40.0), spacing=SPACING, bone='patella')


# --- the in-plane isotropy precondition -------------------------------------------------------

def test_anisotropic_in_plane_spacing_is_refused():
    """A voxel count is not a length on 0.75 x 0.3125 mm data, so the gate refuses it (task #10).

    ``ValueError`` specifically: that is the class the batch scripts' and ``compute_torsion.py``'s
    per-measurement handlers catch, so one unsuitable dataset writes a NaN row instead of aborting
    a cohort run. It is also the same precondition the three consuming angles enforce, so a dataset
    this refuses is one on which none of them returns a value either.
    """
    with pytest.raises(ValueError, match='anisotropic'):
        knee_condylar_line_qc(*line(40.0), spacing=(0.75, 0.3125, 1.0), bone='femur')


def test_a_malformed_spacing_is_refused_as_a_value_error():
    """A degenerate header must not escape as an IndexError or be accepted as a NaN."""
    with pytest.raises(ValueError):
        knee_condylar_line_qc(*line(40.0), spacing=(0.0, 0.0, 1.0), bone='femur')


def test_non_finite_endpoints_are_refused():
    """A NaN endpoint would compare false against the threshold, i.e. read as 'sound'."""
    start, end = line(40.0)
    end[1] = np.nan
    with pytest.raises(ValueError, match='finite'):
        knee_condylar_line_qc(start, end, spacing=SPACING, bone='femur')


# --- the wrapper ------------------------------------------------------------------------------

def test_the_wrapper_describes_a_real_search_on_a_sound_section():
    """A long-lined phantom goes through the real search unflagged, with the length to match."""
    mask = volume(notched_plateau())
    _, start, end = get_knee_reference_line(mask, 'femur')
    direct = knee_condylar_line_qc(start, end, spacing=SPACING, bone='femur')
    assert direct['line_length_voxels'] > THRESHOLD_VOXELS
    assert get_condylar_line_qc(as_image(mask), 'femur', segmentation_label=1, side='left') == ''


def test_the_wrapper_flags_a_short_lined_section_and_reports_the_searched_length():
    """A stub-condyle phantom is flagged, with the value the real search produced.

    The point is that the wrapper is not a replica of the search: the length in the flag is the one
    ``get_knee_reference_line`` returns for that section.
    """
    mask = volume(stub_condyles())
    _, start, end = get_knee_reference_line(mask, 'femur')
    expected = knee_condylar_line_qc(start, end, spacing=SPACING, bone='femur')
    assert expected['flags'] != ''
    assert get_condylar_line_qc(as_image(mask), 'femur', segmentation_label=1,
                                side='left') == expected['flags']


def test_the_wrapper_reads_the_bone_label_it_is_given():
    """Femur and tibia are separate landmarks in one mask, and each call takes its own label."""
    mask = volume(notched_plateau(), label=1) + volume(stub_condyles(), label=2)
    image = as_image(mask)
    assert get_condylar_line_qc(image, 'femur', segmentation_label=1, side='left') == ''
    assert get_condylar_line_qc(image, 'tibia', segmentation_label=2,
                                side='left').startswith('tibial_condylar_line_short:')


def test_the_wrapper_reflects_a_right_half_as_the_measurements_do():
    """A half and its mirror, measured as the other side, give a bitwise identical flag.

    The three consumers measure a right image half as the left-right reflection of a left one, so a
    mirrored half passed in as ``side='right'`` **double** reflects to a bitwise identical array and
    the flag cannot differ at all -- it is the same computation on the same bits, not two
    computations that happen to agree. That is the structural argument ``femur.get_lee_neck_qc``
    rests on, and it is what makes the flags describe the line the measurement used rather than a
    second line that merely resembles it.
    """
    mask = volume(stub_condyles())
    left = get_condylar_line_qc(as_image(mask), 'femur', segmentation_label=1, side='left')
    mirrored = get_condylar_line_qc(as_image(mask[::-1]), 'femur', segmentation_label=1,
                                    side='right')
    assert left == mirrored != ''


def test_a_given_line_is_mirror_invariant():
    """Reflecting the index frame preserves the length, so the verdict on one line cannot move.

    This is weaker than the test above and is here to separate the two claims: the *length of a
    line* is exactly mirror-invariant, while the *search* that finds it is only approximately
    mirror-equivariant. Only the first is a property of this function.
    """
    width = 110
    start, end = line(9.0)
    flipped_start = np.array([width - 1 - start[0], start[1], start[2]])
    flipped_end = np.array([width - 1 - end[0], end[1], end[2]])
    assert (knee_condylar_line_qc(start, end, spacing=SPACING, bone='femur')
            == knee_condylar_line_qc(flipped_start, flipped_end, spacing=SPACING, bone='femur'))


@pytest.mark.parametrize('kwargs, expected', [
    ({'bone': 'femur'}, 'femoral_condylar_landmark_failed:'),
    ({'bone': 'tibia'}, 'tibial_condylar_landmark_failed:'),
])
def test_the_wrapper_never_raises_on_an_empty_label(kwargs, expected):
    """An absent bone label becomes a named failure flag, not an exception.

    The Augsburg scripts call this outside any ``try`` block, so an escaping exception would cost
    the caller the whole patient. An empty label is reported as
    ``<bone>_condylar_landmark_failed:IndexError``: the layer walk returns a layer regardless and
    ``utils.find_notch`` then fails on the empty section with a message too generic to match on, so
    the type-name fallback names it. Deliberately not given a prettier code by matching on an index
    error's message.
    """
    flags = get_condylar_line_qc(np.zeros((40, 40, 3), dtype=np.uint8), **kwargs)
    assert flags.startswith(expected)


def test_the_wrapper_never_raises_on_a_bad_bone_or_side():
    """Bad arguments come back as flags too, and the bone-less fallback name is used for a bad bone."""
    mask = volume(notched_plateau())
    assert get_condylar_line_qc(as_image(mask), 'patella', segmentation_label=1,
                                side='left') == 'condylar_landmark_failed:ValueError'
    assert get_condylar_line_qc(as_image(mask), 'femur', segmentation_label=1,
                                side='middle') == 'femoral_condylar_landmark_failed:ValueError'


def test_the_wrapper_never_raises_on_anisotropic_data():
    """An in-plane anisotropic image is reported as the refusal it is, under the shared code name.

    ``anisotropic_in_plane`` is spelled exactly as ``femur._LANDMARK_FAILURE_REASONS`` spells it,
    so one ``qc_flags`` cell cannot carry two names for the same refusal.
    """
    mask = volume(notched_plateau())
    assert get_condylar_line_qc(as_image(mask, spacing=(0.75, 0.3125, 1.0)), 'femur',
                                segmentation_label=1,
                                side='left') == 'femoral_condylar_landmark_failed:anisotropic_in_plane'


def test_the_wrapper_never_raises_on_nonsense():
    """Whatever is handed in, the caller gets a string back."""
    assert get_condylar_line_qc(None, 'femur').startswith('femoral_condylar_landmark_failed:')
    assert get_condylar_line_qc('not a mask', 'tibia').startswith('tibial_condylar_landmark_failed:')


def test_an_explicit_spacing_wins_over_the_image_and_reaches_the_guard():
    """``spacing=`` overrides the Image's, which is how a bare-array caller gets the precondition.

    ``morphometry/docker/compute_torsion.py`` splits the knee mask into two plain arrays and keeps
    the parent's spacing beside them, so without this the one call site that most needs the guard
    could not have it. The override direction (explicit wins) is task #10's convention for the
    torsion entry points' ``knee_spacing`` / ``ankle_spacing``.
    """
    mask = volume(notched_plateau())
    # an anisotropic override on an isotropic image is refused, so the argument is really read
    assert get_condylar_line_qc(as_image(mask), 'femur', segmentation_label=1, side='left',
                                spacing=(0.75, 0.3125, 1.0)) == \
        'femoral_condylar_landmark_failed:anisotropic_in_plane'
    # and a bare array plus a spacing is checked where the array alone would not be
    assert get_condylar_line_qc(mask, 'femur', segmentation_label=1, side='left',
                                spacing=(0.75, 0.3125, 1.0)) == \
        'femoral_condylar_landmark_failed:anisotropic_in_plane'
    assert get_condylar_line_qc(mask, 'femur', segmentation_label=1, side='left',
                                spacing=SPACING) == ''


def test_the_wrapper_accepts_a_bare_array_and_then_reports_voxels():
    """A bare array carries no geometry, so the isotropy check is skipped and mm is unavailable.

    The flag's value is then the voxel count, which is why the scripts pass the ``Segmentation``.
    """
    mask = volume(stub_condyles())
    assert get_condylar_line_qc(mask, 'femur', segmentation_label=1, side='left') == \
        knee_condylar_line_qc(*get_knee_reference_line(mask, 'femur')[1:], bone='femur')['flags']
