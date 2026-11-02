"""Data-free tests for the pure halves of the two whole-leg CT survey scripts.

Follows ``tests/unit/test_reader_bias_script.py``: both scripts are loaded by path
(``scripts/`` is not a package) and only the functions that transform
already-gathered values are exercised, so no cohort and no sshfs mount is needed.

The reductions worth pinning are the ones whose mistakes would be invisible in the
reports:

* ``survey_ct_spacing.post_lpi_zooms`` must permute the zooms into the axis order
  the **LPI** reorientation produces, not the RAS one. Deriving the transform
  relative to RAS is the natural mistake -- ``nibabel.io_orientation`` is relative
  to RAS -- and it reports every image as already-canonical, which silently turns
  the survey into a tautology. It is also the step that makes the survey able to
  catch a ``dfg_knees``-style scan whose slice axis becomes an in-plane axis.
* ``survey_ct_murphy_qc.summarise`` must not confuse the flag-string column with
  ``DataFrame.flags``, must count a landmark failure as a failure rather than an
  unflagged pass, and must quantile the voxel and millimetre axis lengths
  separately (the in-plane spacing varies per case, so the two orderings differ).
"""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parent.parent.parent


def _load(name):
    """Import one of the survey scripts by path."""
    spec = importlib.util.spec_from_file_location(name, REPO / 'scripts' / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope='module')
def spacing_script():
    """``scripts/survey_ct_spacing.py``."""
    return _load('survey_ct_spacing')


@pytest.fixture(scope='module')
def murphy_script():
    """``scripts/survey_ct_murphy_qc.py``."""
    return _load('survey_ct_murphy_qc')


# --------------------------------------------------------------------------- spacing


def test_post_lpi_zooms_is_identity_on_an_lpi_image(spacing_script):
    """An already-LPI affine permutes to itself, zooms unchanged."""
    affine = np.diag([-0.8, -0.8, -3.0, 1.0])
    zooms, achieved = spacing_script.post_lpi_zooms(affine, (0.8, 0.8, 3.0), (512, 512, 300))

    assert tuple(achieved) == ('L', 'P', 'I')
    assert zooms == pytest.approx([0.8, 0.8, 3.0])


def test_post_lpi_zooms_targets_lpi_not_ras(spacing_script):
    """The reported axis codes must be the LPI the pipelines request.

    An LPS image (which is what every NMDID case is) reorients to LPI, not to RAS.
    A survey built on ``io_orientation`` alone would report RAS here and mark every
    image as non-LPI.
    """
    affine = np.diag([-0.9, -0.9, 3.0, 1.0])              # LPS
    _, achieved = spacing_script.post_lpi_zooms(affine, (0.9, 0.9, 3.0), (512, 512, 400))

    assert tuple(achieved) == ('L', 'P', 'I')


def test_post_lpi_zooms_moves_a_sagittal_slice_axis_into_the_plane(spacing_script):
    """The permutation is the point: a sagittal acquisition's slice axis becomes in-plane.

    This is the ``dfg_knees`` situation (0.75 mm slices becoming an in-plane axis,
    giving 0.75 x 0.3125 mm and k = 2.4). Surveying the raw header zooms would read
    this image as in-plane isotropic and miss the violation entirely.
    """
    # array axis 0 runs left-right in world terms (the 0.75 mm sagittal slice axis),
    # axis 1 anterior-posterior, axis 2 superior-inferior
    affine = np.array([[-0.75, 0.0, 0.0, 0.0],
                       [0.0, -0.3125, 0.0, 0.0],
                       [0.0, 0.0, -0.3125, 0.0],
                       [0.0, 0.0, 0.0, 1.0]])
    zooms, achieved = spacing_script.post_lpi_zooms(affine, (0.75, 0.3125, 0.3125),
                                                    (172, 512, 512))

    assert tuple(achieved) == ('L', 'P', 'I')
    # the 0.75 mm axis is one of the two IN-PLANE axes, so the plane is anisotropic
    assert np.isclose(zooms[:2], 0.75).any()
    assert zooms[0] != pytest.approx(zooms[1])


def test_survey_file_reports_an_unreadable_file_as_a_row(spacing_script, tmp_path):
    """A bad header is a result, not a crash: the row carries ``error`` and fails the check."""
    broken = tmp_path / 'not-a-nifti.nii.gz'
    broken.write_bytes(b'definitely not a nifti')
    row = spacing_script.survey_file(str(broken))

    assert 'error' in row and row['error']
    assert row['isotropy_ok'] is False
    assert np.isnan(row['lpi_x'])


# ---------------------------------------------------------------------------- murphy


def _murphy_frame(rows):
    """Build a per-half frame with the columns :func:`summarise` reads."""
    columns = ['case', 'side', 'landmark', 'flags', 'failure_reason',
               'axis_length_mm', 'axis_length_voxels', 'head_clearance_mm']
    return pd.DataFrame(rows, columns=columns)


def test_summarise_separates_failures_from_unflagged_passes(murphy_script):
    """A half with no landmark is a failure, never a silently sound neck."""
    frame = _murphy_frame([
        ['case-1', 'left', True, '', None, 18.0, 20.0, 30.0],
        ['case-1', 'right', False, 'murphy_landmark_failed:trochanter_implausible',
         'trochanter_implausible', None, None, None],
    ])
    report = murphy_script.summarise(frame)

    assert '**1 of 2 halves yield a Murphy landmark; 1 do not.**' in report
    assert 'trochanter_implausible' in report
    # the failing half must not be counted among the measurable, unflagged ones
    assert '**0 of 1** measurable halves are flagged' in report


def test_summarise_reads_the_flag_column_not_the_dataframe_property(murphy_script):
    """``flags`` collides with ``DataFrame.flags``; the summary must use the column."""
    frame = _murphy_frame([
        ['case-1', 'left', True, 'murphy_axis_short:10.46', None, 10.46, 13.73, 40.0],
        ['case-1', 'right', True, '', None, 18.0, 22.0, 30.0],
    ])
    report = murphy_script.summarise(frame)

    assert '**1 of 2** measurable halves are flagged' in report
    assert 'murphy_axis_short' in report


def test_summarise_calls_out_the_hard_one_voxel_raise(murphy_script):
    """``axis_unresolved`` never fired on MRI, so its CT appearance must be named."""
    frame = _murphy_frame([
        ['case-1', 'left', False, 'murphy_landmark_failed:axis_unresolved',
         'axis_unresolved', None, None, None],
        ['case-1', 'right', True, '', None, 18.0, 22.0, 30.0],
    ])
    report = murphy_script.summarise(frame)

    assert 'axis_unresolved` fires on 1 halves' in report
    assert 'never fired on any of the 1887 MRI halves' in report


def test_summarise_says_so_when_the_one_voxel_raise_does_not_fire(murphy_script):
    """The absence of that failure mode is itself a reportable result."""
    frame = _murphy_frame([
        ['case-1', 'left', False, 'murphy_landmark_failed:shaft_not_found',
         'shaft_not_found', None, None, None],
        ['case-1', 'right', True, '', None, 18.0, 22.0, 30.0],
    ])
    report = murphy_script.summarise(frame)

    assert 'does **not** fire' in report


def test_summarise_quantiles_voxels_and_millimetres_independently(murphy_script):
    """Per-case spacing varies, so the two orderings differ and must not be paired.

    Here the shortest axis in voxels is the LONGEST in millimetres, which a row-wise
    lookup would report as the minimum of both columns.
    """
    frame = _murphy_frame([
        ['case-1', 'left', True, '', None, 30.0, 15.0, 30.0],    # coarse voxels
        ['case-2', 'left', True, '', None, 10.0, 40.0, 30.0],    # fine voxels
    ])
    report = murphy_script.summarise(frame)

    # min voxels is 15.00 while min mm is 10.0 -- they come from different rows
    assert '| min | 15.00 | 10.0 |' in report


def test_axis_threshold_matches_the_library_constant(murphy_script):
    """The reported threshold must be derived from the tolerance, never hard-coded."""
    from morphometry import constants as C
    expected = 1.0 / np.tan(np.deg2rad(C.MURPHY_AXIS_ANGLE_TOLERANCE_DEG))

    assert murphy_script.AXIS_THRESHOLD_VOXELS == pytest.approx(expected)
