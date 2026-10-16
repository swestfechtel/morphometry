"""Unit tests for the laterality support of ``scripts/compute_knee_cartilage_thickness.py``."""
import importlib.util
import pathlib

import numpy as np
import pytest

_SCRIPT = pathlib.Path(__file__).resolve().parents[2] / 'scripts' / 'compute_knee_cartilage_thickness.py'
_spec = importlib.util.spec_from_file_location('knee_thickness_script', _SCRIPT)
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)


def test_laterality_table(tmp_path):
    """subject,laterality CSV: case-insensitive, empty = unknown, junk raises."""
    table = tmp_path / 'lat.csv'
    table.write_text('subject,laterality\nK1_0_belastet,Right\nK2_0_belastet,\nK3_0_belastet,left\n')
    assert mod.load_laterality_table(str(table)) == {'K1_0_belastet': 'right', 'K3_0_belastet': 'left'}
    assert mod.load_laterality_table(None) == {}
    table.write_text('subject,laterality\nK1,medial\n')
    with pytest.raises(ValueError, match='left or right'):
        mod.load_laterality_table(str(table))
    table.write_text('subject,side\nK1,left\n')
    with pytest.raises(ValueError, match='subject,laterality'):
        mod.load_laterality_table(str(table))


def test_summarize_adds_anatomical_names_only_with_laterality():
    """Rows get the medial/lateral names only with a laterality; 'subregion' stays patient-side."""
    results = {'tibia': {'clt': {(0.0, 0.0): 1.0}, 'crt': {(1.0, 0.0): 2.0}},
               'femur': {'iclf': {(0.0, 0.0): 3.0}}}
    rows = mod.summarize(results, 'K1', 'right')        # right knee: patient-left plate medial
    assert [r[2] for r in rows] == ['clt', 'crt', 'iclf']
    col = mod.SUMMARY_COLUMNS.index('anatomical_subregion')
    assert [r[col] for r in rows] == ['cMT', 'cLT', 'icMF']
    assert len(rows[0]) == len(mod.SUMMARY_COLUMNS)
    assert [r[col] for r in mod.summarize(results, 'K1')] == ['', '', '']
    assert np.isclose(rows[1][3], 2.0)


# --- QC flags ------------------------------------------------------------------------

def test_column_order_is_preserved_and_statistics_appended():
    """The original columns keep their positions; the paper's statistics follow them."""
    assert mod.SUMMARY_COLUMNS[:10] == ['subject', 'bone', 'subregion', 'mean_mm', 'std_mm',
                                        'min_mm', 'max_mm', 'n_points', 'anatomical_subregion',
                                        'qc_flags']
    assert mod.SUMMARY_COLUMNS[10:16] == ['aMe_mm', 'aSD_mm', 'aMax_mm', 'aMav_mm',
                                          'aMin_mm', 'aMiv_mm']
    # The tAB family is kept separate from the cAB one, never interleaved.
    assert {'tab_aMe_mm', 'tab_aSD_mm', 'tab_aMin_mm', 'tab_aMiv_mm'} <= set(mod.SUMMARY_COLUMNS)


def test_qc_flags_on_every_row():
    """qc_flags is repeated on each of a subject's rows, including the NaN ones."""
    flag_col = mod.SUMMARY_COLUMNS.index('qc_flags')
    results = {'tibia': {'clt': {(0.0, 0.0): 1.0}, 'crt': {}}, 'femur': {'iclf': {(0.0, 0.0): 3.0}}}
    rows = mod.summarize(results, 'K7', qc_flags='non_lpi_orientation:LSP')
    assert [r[flag_col] for r in rows] == ['non_lpi_orientation:LSP'] * 3   # incl. the NaN row
    assert [r[flag_col] for r in mod.summarize(results, 'K1')] == ['', '', '']


def test_summarize_carries_the_area_weighted_statistics():
    """The appended columns come from ThicknessMaps.statistics; a plain dict gives NaN."""
    import numpy as np
    from morphometry.cartilage.thickness import KneeFrame, ThicknessMaps

    frame = KneeFrame(origin=np.zeros(3), axes=np.eye(3), grid_mm=0.5)
    maps = ThicknessMaps({'clt': {(0, 0): 1.0, (1, 0): 3.0}}, frame=frame, bone='tibia',
                         statistics={'clt': {'aMe': 2.25, 'tab_aMe': 2.0, 'aMax': 3.5,
                                             'aMav': 3.4, 'aSD': 0.5, 'aMin': 0.75,
                                             'aMiv': 0.8, 'tab_aSD': 0.9,
                                             'tab_aMin': 0.0, 'tab_aMiv': 0.1,
                                             'tab_area_mm2': 160.0,
                                             'cab_area_mm2': 142.0,
                                             'denuded_fraction': 0.1125}})
    row = mod.summarize({'tibia': maps}, 'K1')[0]
    assert row[mod.SUMMARY_COLUMNS.index('mean_mm')] == pytest.approx(2.0)   # the plain map mean
    assert row[mod.SUMMARY_COLUMNS.index('aMe_mm')] == pytest.approx(2.25)
    assert row[mod.SUMMARY_COLUMNS.index('tab_aMe_mm')] == pytest.approx(2.0)
    assert row[mod.SUMMARY_COLUMNS.index('aMiv_mm')] == pytest.approx(0.8)
    # cAB and tAB minima are different numbers and go in different columns.
    assert row[mod.SUMMARY_COLUMNS.index('aMin_mm')] == pytest.approx(0.75)
    assert row[mod.SUMMARY_COLUMNS.index('tab_aMin_mm')] == pytest.approx(0.0)
    assert row[mod.SUMMARY_COLUMNS.index('denuded_fraction')] == pytest.approx(0.1125)
    # A plain dict (no statistics) leaves the appended columns NaN but keeps the row shape.
    plain = mod.summarize({'tibia': {'clt': {(0, 0): 1.0}}}, 'K1')[0]
    assert len(plain) == len(mod.SUMMARY_COLUMNS)
    assert np.isnan(plain[mod.SUMMARY_COLUMNS.index('aMe_mm')])


def test_process_one_writes_qc_flags_to_csv_and_json(monkeypatch, tmp_path):
    """End to end through process_one/main: the flags reach the CSV column and the JSON."""
    import csv
    import json
    from types import SimpleNamespace

    oblique = SimpleNamespace(orientation_ok=False, achieved_axcodes=('L', 'S', 'P'))
    clean = SimpleNamespace(orientation_ok=True, achieved_axcodes=('L', 'P', 'I'))
    failing = SimpleNamespace(orientation_ok=False, achieved_axcodes=('L', 'I', 'A'))
    segs = {'K7_90_belastet': oblique, 'K8_90_unbelastet': clean, 'K11_90_belastet': failing}
    monkeypatch.setattr(mod, 'load_segmentation',
                        lambda path, remove_outliers=False: segs[mod._strip_nifti_suffix(
                            pathlib.Path(path).name)])

    def fake_compute(seg, **kwargs):
        if seg is failing:
            return None, None, {}, {}     # both bones failed (like K11_90_belastet)
        # the oblique scan also has a bad region transfer, the clean one a good one
        far = 0.2 if seg is oblique else 0.0
        tibia = SimpleNamespace(transfer_qc={'far_fraction': far})
        return tibia, None, {'clt': {(0.0, 0.0): 1.0}}, {}

    monkeypatch.setattr(mod, 'compute_thickness', fake_compute)
    for name in segs:
        (tmp_path / f'{name}.nii.gz').write_bytes(b'')
    args = SimpleNamespace(input=[str(tmp_path)], laterality_table=None, femur_label=3,
                           tibia_label=4, tibia_bone_label=2, femur_bone_label=1,
                           remove_outliers=False,
                           visualize='none', screenshot=None, html=None,
                           output=str(tmp_path / 'out.csv'), dump_json=str(tmp_path / 'out.json'))
    mod.main(args)

    with open(tmp_path / 'out.csv') as fh:
        table = list(csv.DictReader(fh))
    flags = {row['subject']: row['qc_flags'] for row in table}
    assert flags == {'K7_90_belastet': 'non_lpi_orientation:LSP;tibia_transfer_far:0.200',
                     'K8_90_unbelastet': '', 'K11_90_belastet': 'non_lpi_orientation:LIA'}
    # the scan without any result keeps one placeholder row carrying its flags
    failed = [row for row in table if row['subject'] == 'K11_90_belastet']
    assert len(failed) == 1 and failed[0]['bone'] == '' and failed[0]['n_points'] == '0'
    payload = json.loads((tmp_path / 'out.json').read_text())
    assert payload['K7_90_belastet']['qc_flags'] == 'non_lpi_orientation:LSP;tibia_transfer_far:0.200'
    assert payload['K8_90_unbelastet']['qc_flags'] == ''
    assert set(payload['K8_90_unbelastet']) == {'tibia', 'femur', 'qc_flags', 'statistics',
                                                'frames'}


def test_scan_that_raises_keeps_placeholder_row(monkeypatch, tmp_path):
    """A scan whose processing raises stays in the CSV as one placeholder row; it keeps
    its orientation flag when it loaded, and has no flags when loading itself failed."""
    import csv
    from types import SimpleNamespace

    oblique = SimpleNamespace(orientation_ok=False, achieved_axcodes=('L', 'S', 'P'))

    def fake_load(path, remove_outliers=False):
        if 'unreadable' in path:
            raise OSError('cannot read')
        return oblique

    def fake_compute(seg, **kwargs):
        raise MemoryError('pipeline blew up')       # not caught by compute_thickness

    monkeypatch.setattr(mod, 'load_segmentation', fake_load)
    monkeypatch.setattr(mod, 'compute_thickness', fake_compute)
    for name in ('K7_90_belastet', 'unreadable'):
        (tmp_path / f'{name}.nii.gz').write_bytes(b'')
    args = SimpleNamespace(input=[str(tmp_path)], laterality_table=None, femur_label=3,
                           tibia_label=4, tibia_bone_label=2, femur_bone_label=1,
                           remove_outliers=False,
                           visualize='none', screenshot=None, html=None,
                           output=str(tmp_path / 'out.csv'), dump_json=None)
    mod.main(args)

    with open(tmp_path / 'out.csv') as fh:
        table = list(csv.DictReader(fh))
    assert {row['subject']: row['qc_flags'] for row in table} == {
        'K7_90_belastet': 'non_lpi_orientation:LSP', 'unreadable': ''}
    assert all(row['bone'] == '' and row['n_points'] == '0' for row in table)
