"""Unit tests for the pure halves of scripts/analyse_cartilage_deformation.py.

No image data: the per-scan measurements are stand-ins carrying only what
:func:`process_knee` reads (``statistics``), and :func:`knee_delta_maps` is monkeypatched to
return small synthetic :class:`BoneDeltaMaps`, so the status bookkeeping, the QC flags, the
per-subregion deltas and the CSV rows are tested without registering a single bone.
"""
import importlib.util
import math
import os
import pathlib
import pickle
from types import SimpleNamespace

import numpy as np
import pytest

from morphometry.cartilage.deformation import COMPARISONS, REFERENCE, BoneDeltaMaps, MapGrid
from morphometry.cartilage.femur_regions import FEMUR_REGION_KEYS, ZONE_THICKNESS_KEYS
from morphometry.cartilage.registration import RegistrationResult, RigidTransform
from morphometry.cartilage.tibia_regions import TIBIA_REGION_KEYS

_SCRIPT = (pathlib.Path(__file__).resolve().parents[2] / 'scripts'
           / 'analyse_cartilage_deformation.py')
_spec = importlib.util.spec_from_file_location('analyse_cartilage_deformation', _SCRIPT)
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)

ALL_CONFIGS = (REFERENCE,) + tuple(COMPARISONS)


def _bone(bone, ame=2.0):
    """A stand-in BoneMeasurement: every subregion of ``bone`` at thickness ``ame``."""
    if bone == 'tibia':
        stats = {key: {'aMe': ame} for key in TIBIA_REGION_KEYS}
    else:
        # The femoral statistics key the extension zones by their long names.
        stats = {ZONE_THICKNESS_KEYS.get(key, key): {'aMe': ame} for key in FEMUR_REGION_KEYS}
    return SimpleNamespace(bone=bone, statistics=stats)


def _record(scan, tibia=2.0, femur=2.0, qc_flags='', errors=None):
    """A measure_scan-shaped record; ``None`` thickness = that bone failed."""
    return {'scan': scan,
            'tibia': _bone('tibia', tibia) if tibia is not None else None,
            'femur': _bone('femur', femur) if femur is not None else None,
            'qc_flags': qc_flags, 'errors': dict(errors or {})}


def _knee(configs=ALL_CONFIGS, **overrides):
    """``{(angle, load): record}`` for one knee K1; ``overrides`` maps a scan name to kwargs."""
    out = {}
    for angle, load in configs:
        scan = f'K1_{angle}_{load}'
        out[(angle, load)] = _record(scan, **overrides.get(scan, {}))
    return out


def _registration(rms=0.3):
    return RegistrationResult(RigidTransform(np.eye(3), np.zeros(3)), rms, rms * 0.8, rms * 2,
                              500, 12, True)


def _fake_maps(bone, configurations, rms=None, rotation=None, errors=None):
    """A BoneDeltaMaps over a 12 x 12 grid with a uniform -0.1 mm delta per configuration."""
    rms, rotation, errors = dict(rms or {}), dict(rotation or {}), dict(errors or {})
    grid = MapGrid(np.array([0.0, 0.0]), 1.0, (12, 12))
    keys = TIBIA_REGION_KEYS if bone == 'tibia' else FEMUR_REGION_KEYS
    left = keys.index('clt' if bone == 'tibia' else 'cclf')
    right = keys.index('crt' if bone == 'tibia' else 'ccrf')
    codes = np.full(grid.shape, left, dtype=np.int64)
    codes[6:, :] = right
    reference = np.full(grid.shape, 2.0)
    positions = np.stack(np.meshgrid(np.arange(12.0), np.arange(12.0), indexing='ij')
                         + [np.full(grid.shape, 7.0)], axis=-1)
    ok = [k for k in configurations if k not in errors]
    deltas = {k: np.full(grid.shape, -0.1) for k in ok}
    deltas_cfg = {k: reference + deltas[k] for k in ok}
    return BoneDeltaMaps(bone=bone, grid=grid, reference=reference, region_codes=codes,
                         configurations=deltas_cfg, deltas=deltas,
                         registrations={k: _registration(rms.get(k, 0.3)) for k in ok},
                         dropped_fraction={k: 0.01 for k in ok}, reference_positions=positions,
                         rotation_change_deg={k: rotation.get(k, 2.0) for k in ok},
                         errors=errors)


@pytest.fixture
def fake_delta_maps(monkeypatch):
    """Replace knee_delta_maps by _fake_maps; returns the dict of per-bone overrides."""
    overrides = {'tibia': {}, 'femur': {}}
    calls = []

    def fake(bone, reference, configurations, grid_mm=1.0, **kwargs):
        calls.append((bone, tuple(configurations)))
        spec = overrides[bone]
        if spec.get('raise'):
            raise ValueError(spec['raise'])
        return _fake_maps(bone, configurations, rms=spec.get('rms'),
                          rotation=spec.get('rotation'), errors=spec.get('errors'))

    monkeypatch.setattr(mod, 'knee_delta_maps', fake)
    overrides['calls'] = calls
    return overrides


def _process(measurements, max_rms=1.0):
    return mod.process_knee('K1', measurements, grid_mm=1.0, neighbourhood_mm=3.0,
                            max_rms_mm=max_rms)


def _rows(result, kind, **match):
    return [r for r in result[kind] if all(r[k] == v for k, v in match.items())]


# ---------------------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------------------

def test_parse_scan_name():
    assert mod.parse_scan_name('K1_30_unbelastet') == ('K1', 30, 'unbelastet')
    assert mod.parse_scan_name('K12_90_belastet.nii.gz') == ('K12', 90, 'belastet')
    assert mod.parse_scan_name('K1_30') == (None, None, None)
    assert mod.parse_scan_name('notes.nii.gz') == (None, None, None)


def test_discover_knees_groups_by_knee_and_configuration(tmp_path):
    names = ['K10_0_belastet.nii.gz', 'K2_30_unbelastet.nii.gz', 'K2_0_belastet.nii',
             'K10_30_unbelastet.nii.gz', 'readme.nii.gz']
    paths = [str(tmp_path / n) for n in names]
    knees = mod.discover_knees(paths)
    assert list(knees) == ['K2', 'K10']  # natural order, unparsable name skipped
    assert knees['K2'] == {(30, 'unbelastet'): paths[1], (0, 'belastet'): paths[2]}
    assert knees['K10'] == {(0, 'belastet'): paths[0], (30, 'unbelastet'): paths[3]}


def test_discover_knees_refuses_a_configuration_twice(tmp_path):
    paths = [str(tmp_path / 'a' / 'K1_0_belastet.nii.gz'),
             str(tmp_path / 'b' / 'K1_0_belastet.nii.gz')]
    with pytest.raises(ValueError, match='K1_0_belastet'):
        mod.discover_knees(paths)


# ---------------------------------------------------------------------------------------
# process_knee: statuses
# ---------------------------------------------------------------------------------------

def test_every_knee_gets_one_row_per_comparison_bone_subregion(fake_delta_maps):
    result = _process(_knee())
    assert len(result['subregion_rows']) == 7 * (len(TIBIA_REGION_KEYS) + len(FEMUR_REGION_KEYS))
    assert len(result['hotspot_rows']) == 7 * 2 * 3
    assert len(result['registration_rows']) == 7 * 2
    assert {r['status'] for r in result['subregion_rows']} == {'ok'}
    assert set(result['subregion_rows'][0]) == set(mod.SUBREGION_COLUMNS)
    assert set(result['hotspot_rows'][0]) == set(mod.HOTSPOT_COLUMNS)
    assert set(result['registration_rows'][0]) == set(mod.REGISTRATION_COLUMNS)
    assert mod.REGISTRATION_COLUMNS.index('rotation_change_deg') == \
        mod.REGISTRATION_COLUMNS.index('converged') + 1


def test_missing_configuration_is_status_missing_not_failed(fake_delta_maps):
    configs = [k for k in ALL_CONFIGS if k != (90, 'belastet')]  # K7 has no 90 loaded
    result = _process(_knee(configs))
    missing = _rows(result, 'subregion_rows', angle=90, load='belastet')
    assert len(missing) == len(TIBIA_REGION_KEYS) + len(FEMUR_REGION_KEYS)
    assert {r['status'] for r in missing} == {'missing'}
    assert all(math.isnan(r['delta_mm']) for r in missing)
    hotspots = _rows(result, 'hotspot_rows', angle=90, load='belastet')
    assert len(hotspots) == 6 and {r['status'] for r in hotspots} == {'missing'}
    others = [r for r in result['subregion_rows'] if (r['angle'], r['load']) != (90, 'belastet')]
    assert {r['status'] for r in others} == {'ok'}
    # The missing configuration is never handed to the registration.
    assert all((90, 'belastet') not in keys for _, keys in fake_delta_maps['calls'])
    assert result['panel_titles']['tibia'][(90, 'belastet')] == 'no scan'


def test_failed_reference_fails_every_comparison_of_that_knee(fake_delta_maps):
    measurements = _knee(**{'K1_30_unbelastet': {'tibia': None, 'femur': None,
                                                 'errors': {'tibia': 'ValueError: x',
                                                            'femur': 'ValueError: y'}}})
    result = _process(measurements)
    assert {r['status'] for r in result['subregion_rows']} == {'failed'}
    assert {r['status'] for r in result['hotspot_rows']} == {'failed'}
    assert result['registration_rows'] == []
    assert result['maps'] == {'tibia': None, 'femur': None}
    assert fake_delta_maps['calls'] == []
    assert any('reference' in f['error'] for f in result['failures'])


def test_reference_that_cannot_be_mapped_fails_that_bone(fake_delta_maps):
    fake_delta_maps['femur']['raise'] = 'the reference femur has too few cross-sections'
    result = _process(_knee())
    assert {r['status'] for r in _rows(result, 'subregion_rows', bone='femur')} == {'failed'}
    assert {r['status'] for r in _rows(result, 'subregion_rows', bone='tibia')} == {'ok'}
    assert result['maps']['femur'] is None and result['maps']['tibia'] is not None


def test_one_bone_failing_keeps_the_other_bones_rows(fake_delta_maps):
    measurements = _knee(**{'K1_60_belastet': {'tibia': None,
                                               'errors': {'tibia': 'ValueError: plates overlap'}}})
    result = _process(measurements)
    tibia = _rows(result, 'subregion_rows', angle=60, load='belastet', bone='tibia')
    femur = _rows(result, 'subregion_rows', angle=60, load='belastet', bone='femur')
    assert {r['status'] for r in tibia} == {'failed'}
    assert {r['status'] for r in femur} == {'ok'}
    assert all(r['delta_mm'] == pytest.approx(0.0) for r in femur)
    assert {r['status'] for r in _rows(result, 'hotspot_rows', angle=60, load='belastet',
                                       bone='femur')} == {'ok'}
    # The failed configuration is not registered, but the tibia's other comparisons are.
    tibia_call = [keys for bone, keys in fake_delta_maps['calls'] if bone == 'tibia'][0]
    assert (60, 'belastet') not in tibia_call and len(tibia_call) == 6
    assert result['panel_titles']['tibia'][(60, 'belastet')] == 'failed'
    assert result['maps']['tibia'] is not None and result['maps']['femur'] is not None


def test_registration_error_fails_that_comparison_only(fake_delta_maps):
    fake_delta_maps['tibia']['errors'] = {(0, 'belastet'): 'RuntimeError: ICP diverged'}
    result = _process(_knee())
    assert {r['status'] for r in _rows(result, 'subregion_rows', angle=0, load='belastet',
                                       bone='tibia')} == {'failed'}
    assert {r['status'] for r in _rows(result, 'subregion_rows', angle=0, load='unbelastet',
                                       bone='tibia')} == {'ok'}
    assert {r['status'] for r in _rows(result, 'subregion_rows', angle=0, load='belastet',
                                       bone='femur')} == {'ok'}
    assert any('ICP diverged' in f['error'] for f in result['failures'])


# ---------------------------------------------------------------------------------------
# process_knee: QC flags
# ---------------------------------------------------------------------------------------

def test_registration_residual_over_threshold_flags_the_comparison(fake_delta_maps):
    fake_delta_maps['femur']['rms'] = {(90, 'unbelastet'): 1.73}
    result = _process(_knee(), max_rms=1.0)
    flagged = _rows(result, 'subregion_rows', angle=90, load='unbelastet', bone='femur')
    assert all('registration_rms:1.730' in r['qc_flags'] for r in flagged)
    assert all(r['status'] == 'ok' for r in flagged)  # flagged, still written
    clean = _rows(result, 'subregion_rows', angle=90, load='unbelastet', bone='tibia')
    assert all(r['qc_flags'] == '' for r in clean)
    reg = _rows(result, 'registration_rows', angle=90, load='unbelastet')
    assert {r['bone']: r['flagged'] for r in reg} == {'tibia': False, 'femur': True}
    hot = _rows(result, 'hotspot_rows', angle=90, load='unbelastet', bone='femur')
    assert all('registration_rms:1.730' in r['qc_flags'] for r in hot)


def test_large_rotation_change_flags_the_comparison(fake_delta_maps):
    fake_delta_maps['tibia']['rotation'] = {(30, 'belastet'): 170.0}
    result = _process(_knee())
    rows = _rows(result, 'subregion_rows', angle=30, load='belastet', bone='tibia')
    assert all('registration_rotation:170.0' in r['qc_flags'] for r in rows)
    reg = _rows(result, 'registration_rows', angle=30, load='belastet', bone='tibia')[0]
    assert reg['flagged'] is True and reg['rotation_change_deg'] == 170.0


def test_scan_flags_of_reference_and_configuration_are_joined(fake_delta_maps):
    measurements = _knee(**{'K1_30_unbelastet': {'qc_flags': 'femur_transfer_far:0.020'},
                            'K1_90_belastet': {'qc_flags': 'non_lpi_orientation:LSP'}})
    result = _process(measurements)
    row = _rows(result, 'subregion_rows', angle=90, load='belastet', bone='tibia')[0]
    assert row['qc_flags'] == 'femur_transfer_far:0.020;non_lpi_orientation:LSP'
    row = _rows(result, 'subregion_rows', angle=0, load='belastet', bone='tibia')[0]
    assert row['qc_flags'] == 'femur_transfer_far:0.020'


# ---------------------------------------------------------------------------------------
# process_knee: values
# ---------------------------------------------------------------------------------------

def test_subregion_delta_and_percent(fake_delta_maps):
    measurements = _knee(**{'K1_0_belastet': {'tibia': 1.8, 'femur': 2.5}})
    result = _process(measurements)
    for row in _rows(result, 'subregion_rows', angle=0, load='belastet', bone='tibia'):
        assert row['aMe_ref_mm'] == pytest.approx(2.0)
        assert row['aMe_mm'] == pytest.approx(1.8)
        assert row['delta_mm'] == pytest.approx(-0.2)
        assert row['delta_pct'] == pytest.approx(-10.0)
        assert row['anatomical_subregion'] == ''
    femur = _rows(result, 'subregion_rows', angle=0, load='belastet', bone='femur')
    assert [r['subregion'] for r in femur] == list(FEMUR_REGION_KEYS)  # region keys, not map keys
    assert all(r['delta_pct'] == pytest.approx(25.0) for r in femur)


def test_hotspot_rows_carry_map_and_world_position(fake_delta_maps):
    result = _process(_knee())
    rows = _rows(result, 'hotspot_rows', angle=0, load='unbelastet', bone='tibia')
    assert [r['compartment'] for r in rows] == ['both', 'left', 'right']
    for row in rows:
        assert row['status'] == 'ok'
        assert row['smoothed_delta_mm'] == pytest.approx(-0.1)
        # Ruling 1: world = the reference position of the hotspot cell (x, y = cell index here).
        assert (row['world_x'], row['world_y'], row['world_z']) == \
            pytest.approx((row['map_x_mm'], row['map_y_mm'], 7.0))
    assert rows[1]['subregion'] == 'clt' and rows[2]['subregion'] == 'crt'
    assert result['hotspots']['tibia'][(0, 'unbelastet')]['subregion'] in ('clt', 'crt')


# ---------------------------------------------------------------------------------------
# Cache and pool robustness
# ---------------------------------------------------------------------------------------

def test_cache_is_reused_only_for_the_same_commit_clean_checkout_and_file(tmp_path, monkeypatch):
    scan = tmp_path / 'K1_0_belastet.nii.gz'
    scan.write_bytes(b'abc')
    calls = []

    def fake_measure(path, **labels):
        calls.append(path)
        return _record('K1_0_belastet')

    monkeypatch.setattr(mod, 'measure_scan', fake_measure)
    provenance = {'commit': 'abc123', 'dirty': False}
    labels = dict(femur_label=3, tibia_label=4, femur_bone_label=1, tibia_bone_label=2)
    out = tmp_path / 'out'

    first = mod.measure_scans([str(scan)], out, workers=1, use_cache=True,
                              provenance=provenance, labels=labels)
    assert len(calls) == 1 and first[str(scan)]['scan'] == 'K1_0_belastet'
    cache_file = out / 'cache' / 'K1_0_belastet.pkl'
    assert cache_file.exists()
    assert oct(cache_file.stat().st_mode & 0o777) == oct(0o600)
    mod.measure_scans([str(scan)], out, workers=1, use_cache=True, provenance=provenance,
                      labels=labels)
    assert len(calls) == 1  # reused

    mod.measure_scans([str(scan)], out, workers=1, use_cache=True,
                      provenance={'commit': 'other', 'dirty': False}, labels=labels)
    assert len(calls) == 2  # another commit
    mod.measure_scans([str(scan)], out, workers=1, use_cache=True,
                      provenance={'commit': 'other', 'dirty': True}, labels=labels)
    assert len(calls) == 3  # a dirty checkout is never reused (nor trusted)
    mod.measure_scans([str(scan)], out, workers=1, use_cache=True,
                      provenance={'commit': 'other', 'dirty': True}, labels=labels)
    assert len(calls) == 4
    mod.measure_scans([str(scan)], out, workers=1, use_cache=True,
                      provenance=provenance, labels=labels)
    assert len(calls) == 5  # the dirty run overwrote the cache entry
    scan.write_bytes(b'abcdef')  # the file changed
    mod.measure_scans([str(scan)], out, workers=1, use_cache=True, provenance=provenance,
                      labels=labels)
    assert len(calls) == 6
    mod.measure_scans([str(scan)], out, workers=1, use_cache=True, provenance=provenance,
                      labels=dict(labels, tibia_bone_label=5))
    assert len(calls) == 7  # other labels
    mod.measure_scans([str(scan)], out, workers=1, use_cache=False, provenance=provenance,
                      labels=dict(labels, tibia_bone_label=5))
    assert len(calls) == 8  # --no-cache


def test_worker_crash_is_recorded_not_raised(tmp_path, monkeypatch):
    def boom(path, **labels):
        raise MemoryError('out of memory')

    monkeypatch.setattr(mod, 'measure_scan', boom)
    scan = tmp_path / 'K1_0_belastet.nii.gz'
    scan.write_bytes(b'x')
    records = mod.measure_scans([str(scan)], tmp_path / 'out', workers=1, use_cache=True,
                                provenance={'commit': 'c', 'dirty': False},
                                labels=dict(femur_label=3, tibia_label=4, femur_bone_label=1,
                                            tibia_bone_label=2))
    record = records[str(scan)]
    assert record['tibia'] is None and record['femur'] is None
    assert record['errors']['tibia'].startswith('MemoryError')
    assert not (tmp_path / 'out' / 'cache' / 'K1_0_belastet.pkl').exists()  # never cached


def test_write_csv_round_trips_columns(tmp_path, fake_delta_maps):
    result = _process(_knee())
    path = tmp_path / 'subregion_delta.csv'
    mod.write_csv(result['subregion_rows'], mod.SUBREGION_COLUMNS, path)
    header = path.read_text().splitlines()[0].split(',')
    assert header == mod.SUBREGION_COLUMNS


# ---------------------------------------------------------------------------------------
# Isolation: figures and per-bone crashes
# ---------------------------------------------------------------------------------------

def test_figure_failure_keeps_the_knee_rows_and_is_recorded(fake_delta_maps, monkeypatch,
                                                            tmp_path):
    def broken_plot(*args, **kwargs):
        raise OSError('No space left on device')

    monkeypatch.setattr(mod, 'plot_knee_deltas', broken_plot)
    knee, result = mod._knee_job(('K1', _knee(), 1.0, 3.0, 1.0, tmp_path))
    assert knee == 'K1'
    assert {r['status'] for r in result['subregion_rows']} == {'ok'}
    assert {r['status'] for r in result['hotspot_rows']} == {'ok'}
    figure = [f for f in result['failures'] if f['stage'] == 'figure']
    assert [f['bone'] for f in figure] == ['tibia', 'femur']
    assert all('OSError' in f['error'] for f in figure)
    assert not list(tmp_path.glob('*.png'))


def test_unexpected_crash_of_one_bone_keeps_the_other_bone(monkeypatch, fake_delta_maps):
    real = mod.knee_delta_maps

    def femur_crashes(bone, *args, **kwargs):
        if bone == 'femur':
            raise IndexError('index 40 is out of bounds')
        return real(bone, *args, **kwargs)

    monkeypatch.setattr(mod, 'knee_delta_maps', femur_crashes)
    result = _process(_knee())
    assert {r['status'] for r in _rows(result, 'subregion_rows', bone='tibia')} == {'ok'}
    assert {r['status'] for r in _rows(result, 'subregion_rows', bone='femur')} == {'failed'}
    assert result['maps']['tibia'] is not None and result['maps']['femur'] is None
    crash = [f for f in result['failures'] if f['stage'] == 'delta_maps']
    assert len(crash) == 1 and crash[0]['bone'] == 'femur'
    assert 'IndexError' in crash[0]['error']
