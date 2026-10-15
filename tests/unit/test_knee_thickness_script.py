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
    """Tibial rows get cMT ... only with a laterality; 'subregion' stays patient-side."""
    results = {'tibia': {'clt': {(0.0, 0.0): 1.0}, 'crt': {(1.0, 0.0): 2.0}},
               'femur': {'iclf': {(0.0, 0.0): 3.0}}}
    rows = mod.summarize(results, 'K1', 'right')        # right knee: patient-left plate medial
    assert [r[2] for r in rows] == ['clt', 'crt', 'iclf']
    assert [r[-1] for r in rows] == ['cMT', 'cLT', '']
    assert len(rows[0]) == len(mod.SUMMARY_COLUMNS)
    assert [r[-1] for r in mod.summarize(results, 'K1')] == ['', '', '']
    assert np.isclose(rows[1][3], 2.0)
