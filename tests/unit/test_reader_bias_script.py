"""Data-free tests for the pure half of ``scripts/analyse_reader_bias.py``.

Follows the pattern of ``tests/unit/test_knee_cohort_report_script.py``: the
script is loaded by path (``scripts/`` is not a package) and only the functions
that transform already-loaded tables are exercised, so no cohort is needed.

The summary-row drop in :func:`load_readers` is the one that earns its test --
the readers' xlsx files carry two trailing rows holding the column means and
SDs, which index as ``NaN`` patients and silently corrupt every join.
"""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parent.parent.parent
SCRIPT = REPO / 'scripts' / 'analyse_reader_bias.py'


@pytest.fixture(scope='module')
def script():
    """Import ``analyse_reader_bias.py`` by path, as the other script tests do."""
    spec = importlib.util.spec_from_file_location('analyse_reader_bias', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _reader_frame(tmp_path, name, rows):
    """Write a minimal reader xlsx with the real column names and summary rows."""
    columns = ['Unnamed: 0', 'patient_id',
               'CCD rechts', 'CCD links',
               'AT (Lee) rechts', 'AT (Lee) links',
               'AT (Murphy) rechts', 'AT (Murphy) links']
    frame = pd.DataFrame(rows, columns=columns)
    path = tmp_path / f'reference_measurements_{name}.xlsx'
    frame.to_excel(path, index=False)
    return path


def test_load_readers_drops_the_trailing_mean_and_sd_rows(script, tmp_path):
    """The two summary rows index as NaN patients and must not reach the join."""
    for name in script.READERS:
        _reader_frame(tmp_path, name, [
            ['PA000001', 1, 130.0, 131.0, 10.0, 11.0, 20.0, 21.0],
            ['PA000002', 2, 140.0, 141.0, 12.0, 13.0, 22.0, 23.0],
            [np.nan, np.nan, 135.0, 136.0, 11.0, 12.0, 21.0, 22.0],   # means
            [np.nan, np.nan, 5.0, 5.0, 1.0, 1.0, 1.0, 1.0],           # SDs
        ])
    readers = script.load_readers(tmp_path)

    assert set(readers) == set(script.READERS)
    for frame in readers.values():
        assert list(frame.index) == ['PA000001', 'PA000002']
        assert not frame.index.isna().any()


def test_reader_means_average_the_two_readers_per_leg(script, tmp_path):
    """The reference is the per-leg mean of whichever readers have that column."""
    _reader_frame(tmp_path, 'felix', [['PA000001', 1, 130.0, 131.0, 10.0, 11.0, 20.0, 21.0]])
    _reader_frame(tmp_path, 'jonathan', [['PA000001', 1, 140.0, 141.0, 20.0, 21.0, 30.0, 31.0]])
    means = script.reader_means(script.load_readers(tmp_path))

    # 'links' is the patient's left leg; the two readers are averaged.
    assert means[('ccd', 'left')]['PA000001'] == pytest.approx(136.0)
    assert means[('ccd', 'right')]['PA000001'] == pytest.approx(135.0)
    assert means[('lee', 'left')]['PA000001'] == pytest.approx(16.0)
    assert means[('murphy', 'right')]['PA000001'] == pytest.approx(25.0)


def test_bias_table_reports_signed_and_unsigned_separately(script, tmp_path):
    """A retrotorted leg disagrees in sign but may agree in magnitude."""
    patients = [f'PA{i:06d}' for i in range(6)]
    # The pipeline calls one leg retrotorted at -12; the reader recorded +12.
    measured = pd.DataFrame({
        'lee_left': [12.0, 12.0, 12.0, 12.0, 12.0, -12.0],
        'lee_right': [12.0] * 6,
        'ccd_left': [130.0] * 6, 'ccd_right': [130.0] * 6,
        'murphy_left': [20.0] * 6, 'murphy_right': [20.0] * 6,
    }, index=pd.Index(patients, name='patient'))
    for name in script.READERS:
        _reader_frame(tmp_path, name, [[p, 1, 130.0, 130.0, 12.0, 12.0, 20.0, 20.0]
                                       for p in patients])

    bias = script.bias_table(measured, script.load_readers(tmp_path))
    lee_left = bias[(bias.measure == 'AT (Lee)') & (bias.side == 'left')].iloc[0]

    assert lee_left.n == 6
    assert lee_left.pipe_negative == 1 and lee_left.read_negative == 0
    # One leg off by -24 out of six: signed -4, but the magnitudes all agree.
    assert lee_left.bias == pytest.approx(-4.0)
    assert lee_left.bias_unsigned == pytest.approx(0.0)
    # A measure that agrees exactly has no bias at all.
    ccd_left = bias[(bias.measure == 'CCD') & (bias.side == 'left')].iloc[0]
    assert ccd_left.bias == pytest.approx(0.0)


def test_bias_table_skips_a_measure_with_too_few_paired_legs(script, tmp_path):
    """Fewer than five paired legs is not worth a row (and breaks the Wilcoxon)."""
    patients = ['PA000001', 'PA000002']
    measured = pd.DataFrame({f'{k}_{s}': [10.0, 11.0]
                             for k in script.MEASURES for s in ('left', 'right')},
                            index=pd.Index(patients, name='patient'))
    for name in script.READERS:
        _reader_frame(tmp_path, name, [[p, 1, 130.0, 130.0, 12.0, 12.0, 20.0, 20.0]
                                       for p in patients])

    assert script.bias_table(measured, script.load_readers(tmp_path)).empty
