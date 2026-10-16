"""Unit tests for the pure helpers in scripts/analyse_thinnest_point_flexion.py.

Filename parsing, the articular-surface arc-length fraction and the per-compartment
bookkeeping (thinnest-point search with per-compartment failure isolation, long-format
detail rows, per-compartment sign test) are unit-tested here with synthetic inputs; the
end-to-end cohort analysis is exercised against real segmentations when the script is run.
"""
import importlib.util
import math
import pathlib

import numpy as np
import pytest

_SCRIPT = pathlib.Path(__file__).resolve().parents[2] / 'scripts' / 'analyse_thinnest_point_flexion.py'
_spec = importlib.util.spec_from_file_location('thinnest_flexion', _SCRIPT)
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)


def test_parse_filename():
    assert mod.parse_filename('K1_0_belastet.nii.gz') == ('K1', 0, 'belastet')
    assert mod.parse_filename('K12_90_unbelastet.nii.gz') == ('K12', 90, 'unbelastet')
    assert mod.parse_filename('K3_30_belastet.nii') == ('K3', 30, 'belastet')
    assert mod.parse_filename('not_a_segmentation.txt') is None
    assert mod.parse_filename('K1_0.nii.gz') is None  # missing load


class _Femur:
    def __init__(self, point_cloud):
        self.point_cloud = point_cloud


def test_articular_ap_fraction_anterior_and_posterior():
    """A straight anterior->posterior articular surface gives fraction = y-position along it."""
    # One sagittal column x=5; the tibia-facing surface runs y=0..10 at constant z (flat),
    # so arc length is proportional to y and the fraction equals y / y_max.
    pts = np.array([[5, y, 20] for y in range(11)], float)
    femur = _Femur(pts)
    spacing = (1.0, 1.0, 1.0)
    assert mod.articular_ap_fraction(femur, (5.0, 0.0), spacing) == 0.0    # anterior end
    assert mod.articular_ap_fraction(femur, (5.0, 10.0), spacing) == 1.0   # posterior end
    assert mod.articular_ap_fraction(femur, (5.0, 5.0), spacing) == 0.5    # middle


def test_articular_ap_fraction_degenerate_returns_nan():
    """Too few points in the column -> NaN (skipped downstream)."""
    femur = _Femur(np.array([[5, 3, 20]], float))
    assert np.isnan(mod.articular_ap_fraction(femur, (5.0, 3.0), (1.0, 1.0, 1.0)))


# --- per-compartment bookkeeping ------------------------------------------------------

def test_csv_columns_carry_compartment():
    """Detail rows are long format (compartment after angle); summaries are per compartment."""
    assert mod.COMPARTMENTS == ('both', 'left', 'right')
    assert mod.DETAIL_COLUMNS[:4] == ['subject', 'load', 'angle', 'compartment']
    assert mod.SUMMARY_COLUMNS[:3] == ['subject', 'load', 'compartment']


def _fake_thinnest(x):
    return {'point': (float(x), 1.0), 'center': (float(x), 1.0), 'center_thickness': 2.0,
            'combined_thickness': 1.5, 'tibial_thickness': 0.7, 'femoral_thickness': 0.8,
            'kernel_size': 3}


def test_locate_thinnest_points_isolates_compartment_failures(monkeypatch, capsys):
    """A failing compartment becomes ``None`` (with a warning); the others are still found."""
    tibia = {'clt': {(30.0, 1.0): 1.0}, 'crt': {(5.0, 1.0): 1.0}}
    femur = {'c': {(30.0, 1.0): 1.0, (5.0, 1.0): 1.0}}

    femur_args = []

    def fake_find(tibia_subset, femur_thickness, image, **kwargs):
        femur_args.append(femur_thickness)
        keys = set(tibia_subset)
        if keys == {'clt'}:
            raise ValueError('no overlap')
        return _fake_thinnest(99 if keys == {'clt', 'crt'} else 5)

    monkeypatch.setattr(mod, 'find_thinnest_area', fake_find)
    found = mod.locate_thinnest_points(tibia, femur, image=None, label='K1 0 belastet')

    assert set(found) == set(mod.COMPARTMENTS)
    assert found['left'] is None
    assert found['both']['point'] == (99.0, 1.0)
    assert found['right']['point'] == (5.0, 1.0)
    # Every compartment searches against the *full* femoral map.
    assert len(femur_args) == 3 and all(arg is femur for arg in femur_args)
    err = capsys.readouterr().err
    assert 'K1 0 belastet' in err and 'left' in err and 'no overlap' in err


def test_compartment_row_nan_for_missing_result():
    """A missing (failed) compartment yields an all-NaN measurement row."""
    row = mod.compartment_row(None, None, (1.0, 1.0, 1.0))
    assert set(row) == {'thin_x', 'thin_y', 'ap_fraction', 'combined_thickness',
                        'center_thickness'}
    assert all(math.isnan(v) for v in row.values())


def test_detail_rows_one_per_compartment():
    """``detail_rows`` emits one row per compartment in COMPARTMENTS order."""
    nan_row = mod.compartment_row(None, None, (1.0, 1.0, 1.0))
    case = {'compartments': {
        'both': {'thin_x': 1.0, 'thin_y': 2.0, 'ap_fraction': 0.3,
                 'combined_thickness': 1.1, 'center_thickness': 1.2},
        'left': nan_row,
        'right': {'thin_x': 5.0, 'thin_y': 6.0, 'ap_fraction': 0.7,
                  'combined_thickness': 2.1, 'center_thickness': 2.2},
    }}
    rows = mod.detail_rows('K1', 'belastet', 30, case)
    assert [r[:4] for r in rows] == [['K1', 'belastet', 30, 'both'],
                                     ['K1', 'belastet', 30, 'left'],
                                     ['K1', 'belastet', 30, 'right']]
    assert rows[0][4:9] == [1.0, 2.0, 0.3, 1.1, 1.2]
    assert all(math.isnan(v) for v in rows[1][4:9])
    assert len(rows[0]) == len(mod.DETAIL_COLUMNS)


def test_failed_case_yields_three_nan_detail_rows():
    """A scan whose thickness pipeline failed still yields one all-NaN row per compartment."""
    rows = mod.detail_rows('K2', 'unbelastet', 60, mod.failed_case())
    assert [r[3] for r in rows] == list(mod.COMPARTMENTS)
    for row in rows:
        assert row[:3] == ['K2', 'unbelastet', 60]
        assert all(math.isnan(v) for v in row[4:4 + len(mod.MEASUREMENT_FIELDS)])


def test_summary_rows_keep_insufficient_series():
    """Series with too few finite angles still appear (NaN stats) in summary.csv rows."""
    insufficient = mod.analyse_series([0, 30, 60], [0.1, np.nan, 0.5])
    rows = mod.summary_rows([('K1', 'belastet', 'left', insufficient)])
    assert len(rows) == 1 and rows[0][:4] == ['K1', 'belastet', 'left', 2]
    assert rows[0][mod.SUMMARY_COLUMNS.index('supports_hypothesis')] is None
    assert len(rows[0]) == len(mod.SUMMARY_COLUMNS)


def test_sign_test_by_compartment_counts_each_compartment_separately():
    """Supporting/tested counts and p-values are per compartment; untestable series skipped."""
    def stats(supports):
        if supports is None:                           # mirrors analyse_series' < 3 points
            return {'n_angles': 2, 'spearman_rho': np.nan, 'spearman_p': np.nan,
                    'slope_per_deg': np.nan, 'supports_hypothesis': None}
        return {'n_angles': 4, 'spearman_rho': 0.5, 'spearman_p': 0.5,
                'slope_per_deg': 0.01 if supports else -0.01, 'supports_hypothesis': supports}

    summaries = [
        ('K1', 'belastet', 'both', stats(True)),
        ('K1', 'belastet', 'left', stats(True)),
        ('K1', 'belastet', 'right', stats(False)),
        ('K2', 'belastet', 'both', stats(True)),
        ('K2', 'belastet', 'left', stats(None)),       # insufficient points
        ('K2', 'belastet', 'right', stats(False)),
    ]
    tests = mod.sign_test_by_compartment(summaries)
    assert tests['both'][:2] == (2, 2)
    assert tests['left'][:2] == (1, 1)
    assert tests['right'][:2] == (0, 2)
    assert tests['both'][2] == pytest.approx(0.25)
    assert tests['right'][2] == pytest.approx(1.0)


# --- QC flags ------------------------------------------------------------------------

def test_qc_columns_are_appended_last():
    """The QC columns follow the existing ones, which keep their positions."""
    assert mod.DETAIL_COLUMNS == (['subject', 'load', 'angle', 'compartment']
                                  + mod.MEASUREMENT_FIELDS + ['qc_flags'])
    assert mod.SUMMARY_COLUMNS[:8] == ['subject', 'load', 'compartment', 'n_angles',
                                       'spearman_rho', 'spearman_p', 'slope_per_deg',
                                       'supports_hypothesis']
    assert mod.SUMMARY_COLUMNS[-1] == 'qc_flagged_angles'


def test_detail_rows_repeat_scan_qc_flags():
    """A scan's flags appear on each compartment row; a failed scan keeps its flags."""
    rows = mod.detail_rows('K7', 'belastet', 90, mod.failed_case('non_lpi_orientation:LSP'))
    assert [r[-1] for r in rows] == ['non_lpi_orientation:LSP'] * 3
    assert [r[-1] for r in mod.detail_rows('K1', 'belastet', 0, mod.failed_case())] == [''] * 3


def test_main_flags_scans_and_series(monkeypatch, tmp_path):
    """End to end through main: flagged scans are measured, flagged in detail.csv, and their
    angles listed in summary.csv; a scan whose pipeline fails keeps its orientation flag."""
    import csv
    from types import SimpleNamespace

    for angle in (0, 30, 60, 90):
        (tmp_path / f'K7_{angle}_belastet.nii.gz').write_bytes(b'')
    oblique = SimpleNamespace(orientation_ok=False, achieved_axcodes=('L', 'S', 'P'))
    clean = SimpleNamespace(orientation_ok=True, achieved_axcodes=('L', 'P', 'I'))
    monkeypatch.setattr(mod, 'load_scan',
                        lambda path: oblique if ('_90_' in path or '_60_' in path) else clean)

    def fake_process(path, *a, seg=None, **kw):
        if '_60_' in path:
            raise RuntimeError('broken scan')
        angle = int(pathlib.Path(path).name.split('_')[1])
        row = {'thin_x': 1.0, 'thin_y': 2.0, 'ap_fraction': angle / 100,
               'combined_thickness': 1.0, 'center_thickness': 1.0}
        return {'compartments': {c: dict(row) for c in mod.COMPARTMENTS},
                'qc_flags': mod.scan_qc_flags(seg)}

    monkeypatch.setattr(mod, 'process_file', fake_process)
    args = SimpleNamespace(input=str(tmp_path), femur_label=3, tibia_label=4,
                           femur_bone_label=1,
                           tibia_bone_label=2, method='knn', output=str(tmp_path / 'out'),
                           visualize=False, screenshot=None, html=None)
    mod.main(args)

    with open(tmp_path / 'out' / 'detail.csv') as fh:
        detail = list(csv.DictReader(fh))
    flags = {(row['angle'], row['compartment']): row['qc_flags'] for row in detail}
    assert len(detail) == 12
    assert {flags[('0', c)] for c in mod.COMPARTMENTS} == {''}
    assert {flags[('60', c)] for c in mod.COMPARTMENTS} == {'non_lpi_orientation:LSP'}
    assert {flags[('90', c)] for c in mod.COMPARTMENTS} == {'non_lpi_orientation:LSP'}
    # the flagged 90 deg scan is still measured
    assert all(row['ap_fraction'] == '0.9' for row in detail if row['angle'] == '90')

    with open(tmp_path / 'out' / 'summary.csv') as fh:
        summary = list(csv.DictReader(fh))
    assert [row['qc_flagged_angles'] for row in summary] == ['60;90'] * 3
    assert all(row['n_angles'] == '3' for row in summary)      # 60 failed -> NaN, dropped
