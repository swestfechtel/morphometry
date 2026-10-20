"""The Murphy neck-base QC flags reach the batch scripts' output tables (no image data needed).

Every cohort script that reports a Murphy-derived angle carries the flags of
:func:`morphometry.femur.murphy_neck_base_qc` in a ``qc_flags`` column, appended after the columns
it already had: ``scripts/process_nmdid.py`` (which already had such a column for segmentation
integrity, so the Murphy flags are joined onto it), ``scripts/process_nako_large.py``,
``scripts/process_augsburg_reader_study.py`` and ``scripts/process_augsburg_large_dataset.py``.
A flagged case is measured and written exactly as before; only the column changes.
"""
import importlib.util
import pathlib
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

_SCRIPTS = pathlib.Path(__file__).resolve().parents[2] / 'scripts'


def _load(name: str):
    """Import one script by path, without running its ``__main__`` block.

    :param name: The script's file name.
    :return: The imported module.
    """
    spec = importlib.util.spec_from_file_location(f'script_{name.replace(".py", "")}',
                                                  _SCRIPTS / name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --- process_nmdid.py ----------------------------------------------------------------------

nmdid = _load('process_nmdid.py')


def test_nmdid_joins_the_murphy_flags_onto_the_segmentation_flags(monkeypatch):
    """One ``qc_flags`` column holds both flag families, ``;``-joined, per image side."""
    leg = SimpleNamespace(name='fake leg')
    monkeypatch.setattr(nmdid, 'load_case', lambda path: leg)
    monkeypatch.setattr(nmdid, 'split_legs',
                        lambda seg: ({'left': leg, 'right': leg}, leg,
                                     {'left': 'femur:2pieces', 'right': ''}))
    monkeypatch.setattr(nmdid, 'split_ct_image', lambda *a, **k: (None, [None, None, None]))
    # the measurements themselves are covered elsewhere; here they are stubbed out as NaN
    monkeypatch.setattr(nmdid, '_measure',
                        lambda results, failures, key, fn: [results.__setitem__(k, np.nan)
                                                            for k in ((key,) if isinstance(key, str) else key)])
    monkeypatch.setattr(nmdid, 'get_murphy_neck_qc_ct',
                        lambda leg, side, segmentation_label: 'murphy_axis_short:7.00'
                        if side == 'left' else '')

    rows = {row['side']: row for row in nmdid.measure_case(pathlib.Path('case-1.nii.gz'))}
    assert rows['left']['qc_flags'] == 'femur:2pieces;murphy_axis_short:7.00'
    assert rows['right']['qc_flags'] == ''
    assert 'qc_flags' in nmdid.FIELDNAMES and set(rows['left']) <= set(nmdid.FIELDNAMES)


def test_nmdid_murphy_qc_never_costs_a_row(monkeypatch):
    """A QC helper that blows up must not take the case down with it."""
    def boom(*args, **kwargs):
        raise AttributeError('no array')

    monkeypatch.setattr(nmdid, 'get_murphy_neck_qc_ct', boom)
    assert nmdid.murphy_qc(SimpleNamespace(), 'left') == ''


# --- process_nako_large.py -----------------------------------------------------------------

nako = _load('process_nako_large.py')

#: The chunk-CSV header: the original cohort-table columns with ``qc_flags`` appended last.
NAKO_HEADER = ('Patient,Side,CCD,AT_murphy,AA_anterior,AA_posterior,AAV,CE,Offset,'
               'Cartilage_thickness,Cartilage_thickness (mean),Cartilage_thickness (std),'
               'Cartilage_thickness (min),Cartilage_thickness (max),qc_flags')


def _nako_result(patient: str, left_flags: str = '', right_flags: str = '') -> dict:
    """A per-patient NaKo result with the measurement values stubbed out.

    :param patient: Patient id.
    :param left_flags: QC flags of the image-left half (the patient's right hip).
    :param right_flags: QC flags of the image-right half.
    :return: The result dict :func:`f` would return.
    """
    result = nako._nan_result(patient)
    result.update(murphy_qc_left=left_flags, murphy_qc_right=right_flags)
    return result


def test_nako_appends_the_qc_column_with_the_patient_side_mapping(tmp_path):
    """``qc_flags`` is the last column and follows the image-left = patient-right mapping."""
    df = nako.build_table([_nako_result('p1', left_flags='murphy_base_in_head:13.00'),
                           _nako_result('p2')], ['p1', 'p2'])
    assert list(df.columns)[-1] == 'qc_flags'
    out = tmp_path / 'chunk.csv'
    df.to_csv(out)
    assert out.read_text().splitlines()[0] == NAKO_HEADER

    assert df.loc[('p1', 'right'), 'qc_flags'] == 'murphy_base_in_head:13.00'
    assert df.loc[('p1', 'left'), 'qc_flags'] == ''
    assert (df.loc[('p2',), 'qc_flags'] == '').all()
    # a flagged hip keeps its (here stubbed) AT_murphy value rather than losing it
    assert 'AT_murphy' in df.columns


def test_nako_table_survives_a_result_without_flags():
    """A load failure (``_nan_result``) carries empty flags, so the column is always filled."""
    df = nako.build_table([nako._nan_result('p1', load_failed=True)], ['p1'])
    assert (df['qc_flags'] == '').all()


# --- the two Augsburg scripts --------------------------------------------------------------

@pytest.mark.parametrize('script', ['process_augsburg_reader_study.py',
                                    'process_augsburg_large_dataset.py'])
def test_augsburg_scripts_append_the_qc_column(script):
    """``add_qc_flags`` appends the column with the patient-side mapping and skips missing cases."""
    module = _load(script)
    index = pd.MultiIndex.from_product([['p1', 'p2'], ['right', 'left']], names=['Patient', 'Side'])
    df = pd.DataFrame({'AT (Murphy)': 1.0}, index=index)
    module.add_qc_flags(df, [{'patient': 'p1', 'murphy_qc_left': 'murphy_axis_short:6.97',
                              'murphy_qc_right': ''},
                             {'patient': 'p2'}])  # a patient whose files were missing

    assert list(df.columns)[-1] == 'qc_flags'
    assert df.loc[('p1', 'right'), 'qc_flags'] == 'murphy_axis_short:6.97'   # image-left half
    assert df.loc[('p1', 'left'), 'qc_flags'] == ''
    assert (df.loc[('p2',), 'qc_flags'] == '').all()
    assert (df['AT (Murphy)'] == 1.0).all()  # no measurement touched
