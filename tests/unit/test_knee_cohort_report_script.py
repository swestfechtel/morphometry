"""Unit tests for ``scripts/report_knee_cartilage_cohort.py``.

The script's measurement half needs real knee segmentations (that is what the
``needs_dfg_knees`` goldens and the cohort run cover); everything tested here is the
aggregation half, which is pure and runs on synthetic records: scan-name parsing, which scans
enter the statistics, how the loaded-vs-unloaded pairs and the flexion series are formed, the
plate / condyle / whole-bone aggregates, the pooling rules behind the documented cohort
figures, and the CSV writers.
"""
import csv
import importlib.util
import pathlib

import numpy as np
import pytest

_SCRIPT = (pathlib.Path(__file__).resolve().parents[2] / 'scripts'
           / 'report_knee_cartilage_cohort.py')
_spec = importlib.util.spec_from_file_location('knee_cohort_report_script', _SCRIPT)
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)


def _row(bone: str, region: str, share: float = float('nan'), tab: float = 100.0,
         cab: float = 100.0, aMe: float = 2.0, ac: float = None, **statistics) -> dict:
    """One region row of a synthetic record.

    The two measurement directions default to ``aMe`` each, so that a row with only ``aMe``
    given is internally consistent (``aMe = (cMe + bMe) / 2``).

    :param bone: ``'tibia'`` or ``'femur'``.
    :param region: Region key or aggregate name.
    :param share: The region's share of its plate / condyle.
    :param tab: tAB area in mm^2.
    :param cab: cAB area in mm^2.
    :param aMe: ThC.cAB.aMe in mm.
    :param ac: Articular-surface area in mm^2 (defaults to ``cab``).
    :param statistics: Further statistics to override (e.g. ``aMin=0.1``, ``bMe=1.0``).
    :return: The row dict.
    """
    stats = {'tab_area_mm2': tab, 'cab_area_mm2': cab,
             'ac_area_mm2': cab if ac is None else ac,
             'denuded_fraction': (1.0 - cab / tab) if tab else float('nan'),
             'aMe': aMe, 'cMe': aMe, 'bMe': aMe}
    stats.update(statistics)
    return mod._region_row(bone, region, share, stats, n_cells=7)


def _record(scan: str, subject: str, angle: int, load: str, rows=(), qc_flags: str = '',
            status: str = 'ok') -> dict:
    """A synthetic per-scan record of the shape :func:`mod.measure_scan` returns.

    :param scan: Scan name.
    :param subject: Subject id.
    :param angle: Flexion angle in degrees.
    :param load: ``'belastet'`` or ``'unbelastet'``.
    :param rows: The scan's region rows.
    :param qc_flags: The scan's QC flag string.
    :param status: The scan's status.
    :return: The record dict.
    """
    return {'scan': scan, 'subject': subject, 'angle': angle, 'load': load, 'status': status,
            'error': '', 'qc_flags': qc_flags, 'regions': list(rows), 'geometry': {}}


# --- scan names -----------------------------------------------------------------------

def test_parse_scan_name():
    """``K<subject>_<angle>_<load>`` parses; anything else gives three Nones."""
    assert mod.parse_scan_name('K11_90_belastet') == ('K11', 90, 'belastet')
    assert mod.parse_scan_name('K1_0_unbelastet') == ('K1', 0, 'unbelastet')
    assert mod.parse_scan_name('P07_1Relaxed') == (None, None, None)
    assert mod.parse_scan_name('K1_0_belastet_v2') == (None, None, None)


def test_natural_subject_key_orders_k2_before_k10():
    """Subjects sort numerically, and an unparsed scan sorts last."""
    assert sorted(['K10', 'K2', 'K1'], key=mod._natural_subject_key) == ['K1', 'K2', 'K10']
    assert mod._natural_subject_key(None) > mod._natural_subject_key('K99')


# --- which scans are aggregated -------------------------------------------------------

def test_select_scans_drops_flagged_and_failed():
    """Flagged scans are dropped by default, kept with --include-flagged; failures never count."""
    records = [_record('a', 'K1', 0, 'belastet'),
               _record('b', 'K1', 90, 'belastet', qc_flags='non_lpi_orientation:LSP'),
               _record('c', 'K2', 0, 'belastet', status='failed'),
               _record('d', 'K2', 90, 'belastet', status='tibia_failed')]
    assert [r['scan'] for r in mod.select_scans(records)] == ['a', 'd']
    assert [r['scan'] for r in mod.select_scans(records, include_flagged=True)] == \
        ['a', 'b', 'd']
    assert [r['scan'] for r in
            mod.select_scans(records, include_flagged=True, exclude=['b'])] == ['a', 'd']


def test_load_pairs_and_flexion_series():
    """Pairs need both loads at one angle; a series needs ``min_angles`` of them."""
    records = [_record('K1_0_belastet', 'K1', 0, 'belastet'),
               _record('K1_0_unbelastet', 'K1', 0, 'unbelastet'),
               _record('K1_30_belastet', 'K1', 30, 'belastet'),
               _record('K1_60_belastet', 'K1', 60, 'belastet'),
               _record('K2_0_belastet', 'K2', 0, 'belastet')]
    assert mod.load_pairs(records) == [('K1_0_belastet', 'K1_0_unbelastet')]
    assert mod.flexion_series(records, min_angles=3) == \
        [['K1_0_belastet', 'K1_30_belastet', 'K1_60_belastet']]
    assert len(mod.flexion_series(records, min_angles=2)) == 1
    assert mod.flexion_series(records, min_angles=4) == []


# --- aggregates -----------------------------------------------------------------------

def test_aggregate_row_sums_areas_and_area_weights_the_thickness():
    """A plate / condyle / whole-bone row adds areas and weights each direction by its own area."""
    rows = [_row('tibia', 'clt', tab=100.0, cab=100.0, aMe=3.0, aMin=1.0, aMax=4.0),
            _row('tibia', 'ilt', tab=300.0, cab=200.0, aMe=1.0, aMin=0.1, aMax=2.0)]
    aggregate = mod._aggregate_row('tibia', 'plate_left', rows)
    assert aggregate['tab_area_mm2'] == pytest.approx(400.0)
    assert aggregate['cab_area_mm2'] == pytest.approx(300.0)
    assert aggregate['denuded_fraction'] == pytest.approx(0.25)
    # Both directions have cAB = AC here, so both halves give (3.0 * 100 + 1.0 * 200) / 300.
    assert aggregate['aMe_mm'] == pytest.approx(5.0 / 3.0)
    assert aggregate['tab_aMe_mm'] == pytest.approx(5.0 / 3.0 * 0.75)
    # Extremes are extremes of the parts, not weighted means.
    assert aggregate['aMin_mm'] == pytest.approx(0.1)
    assert aggregate['aMax_mm'] == pytest.approx(4.0)
    assert aggregate['share'] != aggregate['share']   # NaN: an aggregate has no share
    assert aggregate['n_cells'] == 14


def test_aggregate_row_weights_the_two_directions_by_their_own_areas():
    """``aMe`` is exact: ``cMe`` is cAB-weighted, ``bMe`` AC-weighted, not both cAB-weighted.

    Pinning the fix for a real defect: taking the union ``aMe`` as a cAB-weighted mean of the
    subregion ``aMe``s is wrong whenever a subregion's AC area differs from its cAB area, which
    it always does, and it moved the documented whole-bone figures.
    """
    rows = [_row('tibia', 'clt', tab=100.0, cab=100.0, ac=100.0, cMe=3.0, bMe=3.0, aMe=3.0),
            _row('tibia', 'ilt', tab=100.0, cab=100.0, ac=300.0, cMe=1.0, bMe=1.0, aMe=1.0)]
    aggregate = mod._aggregate_row('tibia', 'plate_left', rows)
    assert aggregate['cMe_mm'] == pytest.approx(2.0)                     # cAB 100 : 100
    assert aggregate['bMe_mm'] == pytest.approx((3.0 * 100 + 1.0 * 300) / 400)   # AC 100 : 300
    assert aggregate['aMe_mm'] == pytest.approx(0.5 * (2.0 + 1.5))
    # The old (cAB-weighted-aMe) answer would have been 2.0 flat.
    assert aggregate['aMe_mm'] != pytest.approx(2.0)


def test_aggregate_row_takes_the_tab_minimum_over_the_parts():
    """``tab_aMin`` is a min over the union, so one denuded subregion makes it exactly 0.0.

    It used to fall through to the weighted mean, because the branch tested the literal
    ``'aMin'``; that reported a whole-bone ThC.tAB.aMin of a few tenths of a millimetre where
    the true union value is 0.0 on every scan with any denuded bone.
    """
    rows = [_row('tibia', 'clt', tab=100.0, cab=100.0, tab_aMin=1.5, aMin=1.5),
            _row('tibia', 'ilt', tab=200.0, cab=190.0, tab_aMin=0.0, aMin=0.2)]
    aggregate = mod._aggregate_row('tibia', 'plate_left', rows)
    assert aggregate['tab_aMin_mm'] == 0.0
    assert aggregate['aMin_mm'] == pytest.approx(0.2)
    # The dispersion statistics have no exact union value and stay weighted means.
    assert 'aSD' not in mod.EXACT_ON_AGGREGATES and 'tab_aMin' in mod.EXACT_ON_AGGREGATES


def test_aggregate_row_tolerates_a_missing_statistic():
    """A region whose statistics are absent (NaN) does not poison the aggregate."""
    rows = [_row('femur', 'iclf', tab=100.0, cab=100.0, aMe=2.0),
            _row('femur', 'cclf', tab=100.0, cab=100.0, aMe=float('nan'),
                 cMe=float('nan'), bMe=float('nan'))]
    aggregate = mod._aggregate_row('femur', 'cf_left', rows)
    assert aggregate['aMe_mm'] == pytest.approx(2.0)
    assert aggregate['tab_area_mm2'] == pytest.approx(200.0)


# --- cohort tables --------------------------------------------------------------------

def test_region_tables_report_shares_in_percent_and_keep_the_paper_separate():
    """Shares are reported in %, and the paper's values are their own rows, never paired."""
    records = [_record('K1_0_belastet', 'K1', 0, 'belastet',
                       [_row('tibia', 'clt', share=0.2), _row('tibia', 'ilt', share=0.16),
                        _row('femur', 'cclf', share=1 / 3)])]
    rows = mod.region_tables(records)
    central = mod._lookup(rows, 'tibia_region_shares', 'plate_left', 'central')
    assert central['mean'] == pytest.approx(20.0)
    assert central['paper'] == ''      # our plates are patient-side, the paper's are medial/lateral
    third = mod._lookup(rows, 'femur_region_shares', 'condyle_left', 'central')
    assert third['mean'] == pytest.approx(100 / 3)
    paper = mod.paper_comparison_rows(records)
    assert mod._lookup(paper, 'paper_tibia_region_shares', 'MT', 'anterior')['mean'] == 24.2
    assert mod._lookup(paper, 'paper_femur_region_shares', 'cLF', 'internal')['mean'] == 33.4


def test_summary_row_statistics():
    """``_summary_row`` drops NaNs and reports n / mean / sd / min / max / p50 / p90."""
    row = mod._summary_row('t', 's', 'r', 'mm', [1.0, 2.0, 3.0, float('nan')])
    assert (row['n'], row['mean'], row['min'], row['max'], row['p50']) == (3, 2.0, 1.0, 3.0, 2.0)
    assert row['sd'] == pytest.approx(1.0)                 # ddof = 1
    empty = mod._summary_row('t', 's', 'r', 'mm', [float('nan')])
    assert empty['n'] == 0 and np.isnan(empty['mean'])


# --- reproducibility ------------------------------------------------------------------

def _two_load_records(**kw):
    """A loaded / unloaded pair of one subject and angle, with one tibial subregion each.

    :param kw: ``belastet`` / ``unbelastet`` aMe values.
    :return: The two records.
    """
    return [
        _record('K1_0_belastet', 'K1', 0, 'belastet',
                [_row('tibia', 'clt', share=0.2, tab=100.0, aMe=kw['belastet']),
                 _row('tibia', 'plate_left', tab=100.0, aMe=kw['belastet'])]),
        _record('K1_0_unbelastet', 'K1', 0, 'unbelastet',
                [_row('tibia', 'clt', share=0.2, tab=110.0, aMe=kw['unbelastet']),
                 _row('tibia', 'plate_left', tab=110.0, aMe=kw['unbelastet'])]),
    ]


def test_load_spread_is_absolute_for_thickness_and_relative_for_area():
    """The loaded-vs-unloaded difference is mm for aMe and % of the mean for an area."""
    rows, counts = mod.reproducibility_tables(_two_load_records(belastet=2.0, unbelastet=2.5),
                                              min_angles=3)
    assert counts == {'load_pairs': 1, 'flexion_series': 0}
    assert mod._lookup(rows, 'load_spread_aMe', 'tibia', 'clt')['mean'] == pytest.approx(0.5)
    area = mod._lookup(rows, 'load_spread_tab_area', 'tibia', 'clt')
    assert area['unit'] == '%'
    assert area['mean'] == pytest.approx(200 * 10 / 210)   # |100-110| / mean(100, 110)


def test_pooled_reproducibility_covers_the_subregions_only():
    """The pooled samples hold one value per (pair, subregion): aggregates would double-count.

    This is the convention the documented cohort figures use (e.g. 45 load pairs x 20
    subregions = 900 aMe differences), so a change here would silently move them.
    """
    rows, _counts = mod.reproducibility_tables(_two_load_records(belastet=2.0, unbelastet=3.0),
                                               min_angles=3)
    pooled = mod._lookup(rows, 'load_spread_aMe', 'pooled', '__all__')
    assert pooled['n'] == 1                                 # clt only, not plate_left
    assert pooled['mean'] == pytest.approx(1.0)
    assert mod._lookup(rows, 'load_spread_aMe', 'pooled_tibia', '__all__')['n'] == 1
    assert mod._lookup(rows, 'load_spread_aMe', 'pooled_femur', '__all__')['n'] == 0
    # The aggregate still gets its own row, it is just not pooled.
    assert mod._lookup(rows, 'load_spread_aMe', 'tibia', 'plate_left')['n'] == 1


def test_flexion_spread_is_the_within_series_sd():
    """The flexion spread is the SD of the series' values (CV% for an area)."""
    records = [_record(f'K1_{angle}_belastet', 'K1', angle, 'belastet',
                       [_row('tibia', 'clt', share=0.2, tab=100.0, aMe=value)])
               for angle, value in ((0, 1.0), (30, 2.0), (60, 3.0))]
    rows, counts = mod.reproducibility_tables(records, min_angles=3)
    assert counts['flexion_series'] == 1
    assert mod._lookup(rows, 'flexion_spread_aMe', 'tibia', 'clt')['mean'] == pytest.approx(1.0)
    # Constant area: no spread at all, and the CV is 0 rather than NaN.
    assert mod._lookup(rows, 'flexion_spread_tab_area', 'tibia', 'clt')['mean'] == \
        pytest.approx(0.0)
    assert mod._lookup(rows, 'flexion_cv_aMe', 'tibia', 'clt')['mean'] == pytest.approx(50.0)


# --- output ---------------------------------------------------------------------------

def test_write_scan_csv_keeps_failed_scans_visible(tmp_path):
    """A scan with no region rows keeps one placeholder row, with its status and flags."""
    records = [_record('K1_0_belastet', 'K1', 0, 'belastet',
                       [_row('tibia', 'clt', share=0.2, aMe=2.0)]),
               _record('K11_90_belastet', 'K11', 90, 'belastet', qc_flags='non_lpi_orientation:LIA',
                       status='failed')]
    path = tmp_path / 'scan_metrics.csv'
    mod.write_scan_csv(records, path)
    with open(path, newline='') as fh:
        table = list(csv.DictReader(fh))
    assert list(table[0]) == mod.SCAN_COLUMNS
    assert [r['scan'] for r in table] == ['K1_0_belastet', 'K11_90_belastet']
    assert table[0]['region'] == 'clt' and float(table[0]['aMe_mm']) == pytest.approx(2.0)
    assert table[1]['region'] == '' and table[1]['status'] == 'failed'
    assert table[1]['qc_flags'] == 'non_lpi_orientation:LIA'


def test_write_summary_csv_round_trips(tmp_path):
    """Every summary row is written with the full column set, NaNs included."""
    rows = [mod._summary_row('tab_area', 'tibia', 'clt', 'mm2', [1.0, 2.0]),
            mod._summary_row('tab_area', 'tibia', 'ilt', 'mm2', [])]
    path = tmp_path / 'cohort_summary.csv'
    mod.write_summary_csv(rows, path)
    with open(path, newline='') as fh:
        table = list(csv.DictReader(fh))
    assert list(table[0]) == mod.SUMMARY_COLUMNS
    assert float(table[0]['mean']) == pytest.approx(1.5)
    assert table[1]['n'] == '0' and table[1]['mean'] == 'nan'


def test_statistic_families_are_both_reported():
    """Both ThC families reach the per-scan table; mixing them is what keeping them apart avoids."""
    assert {'aMe', 'aSD', 'aMax', 'aMav', 'aMin', 'aMiv'} <= set(mod.THICKNESS_STATISTICS)
    assert {'tab_aMe', 'tab_aSD', 'tab_aMin', 'tab_aMiv'} <= set(mod.THICKNESS_STATISTICS)
    assert all(f'{s}_mm' in mod.SCAN_COLUMNS for s in mod.THICKNESS_STATISTICS)


def test_relative_quantities_emit_no_duplicate_cv_row():
    """For an area the spread already *is* the CV, so only one row is written.

    ``_flexion_cv`` falls back to the spread row, so the report's CV column still fills.
    """
    records = [_record(f'K1_{angle}_belastet', 'K1', angle, 'belastet',
                       [_row('tibia', 'clt', share=0.2, tab=float(100 + angle))])
               for angle in (0, 30, 60)]
    rows, _counts = mod.reproducibility_tables(records, min_angles=3)
    assert mod._lookup(rows, 'flexion_cv_tab_area', 'tibia', 'clt') is None
    spread = mod._lookup(rows, 'flexion_spread_tab_area', 'tibia', 'clt')
    assert mod._flexion_cv(rows, 'tab_area', 'tibia', 'clt') is spread
    # A thickness keeps its own CV row, which is a different quantity from its mm spread.
    assert mod._lookup(rows, 'flexion_cv_aMe', 'tibia', 'clt') is not None


def test_relative_spread_of_a_vanishing_quantity_is_not_reported_as_absolute():
    """A '%' column must not silently carry millimetres when both values are 0."""
    records = [_record('K1_0_belastet', 'K1', 0, 'belastet',
                       [_row('tibia', 'clt', share=0.2, tab=0.0, cab=0.0)]),
               _record('K1_0_unbelastet', 'K1', 0, 'unbelastet',
                       [_row('tibia', 'clt', share=0.2, tab=0.0, cab=0.0)])]
    rows, _counts = mod.reproducibility_tables(records, min_angles=3)
    assert mod._lookup(rows, 'load_spread_tab_area', 'tibia', 'clt')['n'] == 0


def test_geometry_summary_leaves_the_timings_out():
    """``cohort_summary.csv`` carries no timing, so it is comparable between runs."""
    records = [_record('K1_0_belastet', 'K1', 0, 'belastet')]
    records[0]['geometry'] = {'tibia_seconds': 7.0, 'femur_seconds': 13.0,
                              'tibia_transfer_max_mm': 3.0,
                              'tibia_right_plate_larger': 1.0}
    rows = mod.geometry_tables(records)
    assert [r['region'] for r in rows] == ['tibia_right_plate_larger', 'tibia_transfer_max_mm']
    assert 'share of scans' in mod._lookup(rows, 'geometry', '', 'tibia_right_plate_larger')['unit']
    assert mod.timing_summary(records)['tibia_seconds']['median'] == pytest.approx(7.0)


def test_write_meta_records_provenance(tmp_path):
    """The run's meta file names the code, the inputs and which scans were left out."""
    import argparse
    import json
    records = [_record('K1_0_belastet', 'K1', 0, 'belastet', [_row('tibia', 'clt')]),
               _record('K7_90_belastet', 'K7', 90, 'belastet', [_row('tibia', 'clt')],
                       qc_flags='non_lpi_orientation:LSP')]
    selected = mod.select_scans(records)
    args = argparse.Namespace(input=['/data/knees'], output=tmp_path, exclude=[],
                              include_flagged=False, min_angles=3, workers=4)
    path = tmp_path / 'report.meta.json'
    mod.write_meta(path, args, ['/data/knees/K1_0_belastet.nii.gz'], records, selected,
                   {'load_pairs': 0, 'flexion_series': 0})
    meta = json.loads(path.read_text())
    assert meta['n_measured'] == 2 and meta['n_aggregated'] == 1
    assert meta['excluded_by_flag'] == ['K7_90_belastet'] and meta['failed'] == []
    assert meta['inputs'] == ['/data/knees'] and meta['args']['workers'] == 4
    assert set(meta) >= {'commit', 'dirty', 'branch', 'date', 'host', 'seconds_per_bone'}
