"""Data-free tests for the pure half of ``scripts/analyse_ccd_x_ratio.py``.

Follows the pattern of ``tests/unit/test_reader_bias_script.py``: the script is
loaded by path (``scripts/`` is not a package) and only the functions that reduce
already-measured tables are exercised, so no cohort is needed.

The reductions worth pinning are the ones whose mistakes would be invisible in the
report: pairing the two legs of a patient for the reader-free left/right agreement,
counting the thin neck clouds that are the script's whole mechanism, and keeping a
factor's row when one of its criteria cannot be computed rather than dropping the
factor.
"""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parent.parent.parent
SCRIPT = REPO / 'scripts' / 'analyse_ccd_x_ratio.py'


@pytest.fixture(scope='module')
def script():
    """Import ``analyse_ccd_x_ratio.py`` by path, as the other script tests do."""
    spec = importlib.util.spec_from_file_location('analyse_ccd_x_ratio', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sweep(rows):
    """Build a long sweep frame with the columns :func:`summarise` reads."""
    return pd.DataFrame(rows, columns=['patient', 'side', 'x_ratio', 'ccd',
                                       'neck_voxels', 'neck_slice_span'])


def test_summarise_pairs_the_two_legs_of_a_patient(script):
    """``lr_*`` is a within-patient quantity: the legs must be paired, not pooled."""
    rows = []
    # two factors; at 5.0 the legs agree exactly, at 9.0 they disagree by 10 degrees
    for patient, left in (('PA000001', 130.), ('PA000002', 140.), ('PA000003', 150.)):
        rows.append([patient, 'left', 5.0, left, 500, 4.])
        rows.append([patient, 'right', 5.0, left, 500, 4.])
        rows.append([patient, 'left', 9.0, left, 400, 2.])
        rows.append([patient, 'right', 9.0, left + 10., 400, 2.])
    summary = script.summarise(_sweep(rows), {})

    at5 = summary[summary.x_ratio == 5.0].iloc[0]
    at9 = summary[summary.x_ratio == 9.0].iloc[0]
    assert at5.lr_mad == pytest.approx(0.0)
    assert at9.lr_mad == pytest.approx(10.0)
    # a constant offset leaves the correlation perfect: lr_pearson and lr_mad are
    # different criteria and the summary must carry both.
    assert at5.lr_pearson == pytest.approx(1.0)
    assert at9.lr_pearson == pytest.approx(1.0)


def test_summarise_counts_thin_neck_clouds_at_the_threshold(script):
    """``thin_halves`` is the mechanism the script exists to show, so it is inclusive."""
    rows = [['PA000001', 'left', 9.0, 140., 400, span] for span in (1., 2., 3., 2.)]
    rows += [['PA000002', 'right', 9.0, 141., 400, 5.]]
    summary = script.summarise(_sweep(rows), {}, thin_slices=2)

    # spans of 1, 2 and 2 are thin; 3 and 5 are not
    assert summary.iloc[0].thin_halves == 3
    assert summary.iloc[0].neck_slice_span_median == pytest.approx(2.0)   # 1, 2, 2, 3, 5


def test_summarise_keeps_a_factor_whose_reader_comparison_is_missing(script):
    """A factor with no reader column keeps its row with NaN bias, not no row at all.

    The sweep's reader-free criteria are the ones that disagree with the reader ones,
    so losing a factor because the readers are absent would hide exactly that.
    """
    rows = [['PA000001', 'left', 5.0, 130., 500, 4.],
            ['PA000001', 'right', 5.0, 131., 500, 4.]]
    summary = script.summarise(_sweep(rows), {})

    assert len(summary) == 1
    row = summary.iloc[0]
    assert np.isnan(row.bias_left) and np.isnan(row.mad_left)
    assert row.lr_mad == pytest.approx(1.0)        # still computed


def test_summarise_compares_against_the_reader_mean_per_side(script):
    """The bias is per patient side against that side's reader mean."""
    rows = [['PA000001', 'left', 5.0, 140., 500, 4.],
            ['PA000001', 'right', 5.0, 120., 500, 4.]]
    means = {('ccd', 'left'): pd.Series({'PA000001': 130.}),
             ('ccd', 'right'): pd.Series({'PA000001': 130.})}
    summary = script.summarise(_sweep(rows), means).iloc[0]

    assert summary.bias_left == pytest.approx(+10.0)
    assert summary.bias_right == pytest.approx(-10.0)
    # MAD is unsigned, so both sides read the same magnitude
    assert summary.mad_left == pytest.approx(10.0)
    assert summary.mad_right == pytest.approx(10.0)


def test_summarise_is_sorted_by_factor_and_covers_every_factor(script):
    """The report reads as a sweep, so the rows must be in factor order."""
    rows = []
    for factor in (9.0, 1.0, 5.0):
        rows.append(['PA000001', 'left', factor, 140., 500, 4.])
        rows.append(['PA000001', 'right', factor, 141., 500, 4.])
    summary = script.summarise(_sweep(rows), {})

    assert list(summary.x_ratio) == [1.0, 5.0, 9.0]
    assert list(summary.n) == [2, 2, 2]


def test_named_factors_are_the_pipeline_formula_and_the_slice_aspect(script):
    """The two named constants must stay the two values the item is about.

    5.46875 is ``spacing_z / 2 * spacing_x`` and 9.142857 is ``spacing_z / spacing_x``
    at the Augsburg hip geometry (1.09375 mm in plane, 10 mm slices); both are in the
    swept list, so the sweep brackets them rather than reporting them alone.
    """
    spacing_x, spacing_z = 1.09375, 10.0
    assert script.PIPELINE_FACTOR == pytest.approx(spacing_z / 2 * spacing_x)
    assert script.ASPECT_FACTOR == pytest.approx(spacing_z / spacing_x)
    assert script.PIPELINE_FACTOR in script.SWEEP_FACTORS
    assert script.ASPECT_FACTOR in script.SWEEP_FACTORS
