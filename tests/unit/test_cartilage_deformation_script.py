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


def test_reference_that_cannot_be_mapped_fails_that_bones_maps_only(fake_delta_maps):
    """
    An un-unrollable reference femur costs that bone's maps and hotspots, not its subregion
    Δ, which does not go through the registration or the map (controller ruling 8).
    """
    fake_delta_maps['femur']['raise'] = 'the reference femur has too few cross-sections'
    result = _process(_knee())
    femur = _rows(result, 'subregion_rows', bone='femur')
    assert {r['status'] for r in femur} == {'ok'}
    assert all(r['delta_mm'] == pytest.approx(0.0) for r in femur)
    assert {r['status'] for r in _rows(result, 'hotspot_rows', bone='femur')} == {'failed'}
    assert {r['status'] for r in _rows(result, 'subregion_rows', bone='tibia')} == {'ok'}
    assert {r['status'] for r in _rows(result, 'hotspot_rows', bone='tibia')} == {'ok'}
    assert result['maps']['femur'] is None and result['maps']['tibia'] is not None
    assert set(result['panel_titles']['femur'].values()) == {'failed'}
    assert not _rows(result, 'registration_rows', bone='femur')


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


def test_registration_error_fails_that_comparisons_maps_only(fake_delta_maps):
    """A registration error fails that comparison's hotspots and panel; its subregion Δ stays."""
    fake_delta_maps['tibia']['errors'] = {(0, 'belastet'): 'RuntimeError: ICP diverged'}
    result = _process(_knee())
    rows = _rows(result, 'subregion_rows', angle=0, load='belastet', bone='tibia')
    assert {r['status'] for r in rows} == {'ok'}
    assert all(math.isfinite(r['delta_mm']) for r in rows)
    assert {r['status'] for r in _rows(result, 'hotspot_rows', angle=0, load='belastet',
                                       bone='tibia')} == {'failed'}
    assert result['panel_titles']['tibia'][(0, 'belastet')] == 'failed'
    assert {r['status'] for r in _rows(result, 'hotspot_rows', angle=0, load='unbelastet',
                                       bone='tibia')} == {'ok'}
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

    for name in ('plot_knee_deltas', 'plot_knee_residuals', 'plot_registration_sections'):
        monkeypatch.setattr(mod, name, broken_plot)
    knee, result = mod._knee_job(('K1', _knee(), 1.0, 3.0, 1.0, 2.0, tmp_path))
    assert knee == 'K1'
    assert {r['status'] for r in result['subregion_rows']} == {'ok'}
    assert {r['status'] for r in result['hotspot_rows']} == {'ok'}
    figure = [f for f in result['failures'] if f['stage'] == 'figure']
    # Every figure of every bone is attempted and recorded on its own.
    by_bone = {bone: sorted(f['error'].split(':')[0] for f in figure if f['bone'] == bone)
               for bone in ('tibia', 'femur')}
    assert by_bone['tibia'] == by_bone['femur']
    assert {'delta', 'registration_residual'} <= set(by_bone['tibia'])
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
    # The subregion Δ does not need the maps: kept; the femoral hotspots fail.
    assert {r['status'] for r in _rows(result, 'subregion_rows', bone='femur')} == {'ok'}
    assert {r['status'] for r in _rows(result, 'hotspot_rows', bone='femur')} == {'failed'}
    assert result['maps']['tibia'] is not None and result['maps']['femur'] is None
    crash = [f for f in result['failures'] if f['stage'] == 'delta_maps']
    assert len(crash) == 1 and crash[0]['bone'] == 'femur'
    assert 'IndexError' in crash[0]['error']


# ---------------------------------------------------------------------------------------
# Cohort stage: selection, laterality, statistics, report
# ---------------------------------------------------------------------------------------

def _srow(knee, region, delta, angle=0, load='unbelastet', bone='tibia', status='ok',
          qc_flags='', ref=2.0):
    """A subregion_delta row as process_knee writes it."""
    pct = 100.0 * delta / ref if np.isfinite(delta) else np.nan
    return {'knee': knee, 'angle': angle, 'load': load, 'bone': bone, 'subregion': region,
            'anatomical_subregion': '', 'aMe_ref_mm': ref, 'aMe_mm': ref + delta,
            'delta_mm': delta, 'delta_pct': pct, 'status': status, 'qc_flags': qc_flags}


def _hrow(knee, compartment, region, x=3.0, y=1.0, angle=0, load='unbelastet', bone='tibia',
          status='ok', qc_flags=''):
    """A hotspots row as process_knee writes it."""
    return {'knee': knee, 'angle': angle, 'load': load, 'bone': bone,
            'compartment': compartment, 'map_x_mm': x, 'map_y_mm': y, 'world_x': 10.0,
            'world_y': 20.0, 'world_z': 30.0, 'subregion': region,
            'anatomical_subregion': '', 'smoothed_delta_mm': -0.2, 'delta_mm': -0.25,
            'kernel_size': 3, 'fallback': False, 'status': status, 'qc_flags': qc_flags}


def test_flagged_rows_are_excluded_by_default_and_included_on_request():
    rows = [_srow('K1', 'clt', -0.1),
            _srow('K2', 'clt', -0.1, qc_flags='registration_rms:1.300'),
            _srow('K3', 'clt', np.nan, status='failed'),
            _srow('K4', 'clt', np.nan, status='missing'),
            _srow('K5', 'clt', -0.1, angle=60),
            _srow('K6', 'clt', -0.1)]
    assert [r['knee'] for r in mod.select_rows(rows, False, [])] == ['K1', 'K5', 'K6']
    assert [r['knee'] for r in mod.select_rows(rows, True, [])] == ['K1', 'K2', 'K5', 'K6']
    # --exclude names a scan: the configuration's own scan or the knee's reference scan.
    kept = mod.select_rows(rows, True, ['K5_60_unbelastet', 'K6_30_unbelastet'])
    assert [r['knee'] for r in kept] == ['K1', 'K2']
    # A scan name that is neither leaves the row alone.
    assert [r['knee'] for r in mod.select_rows(rows, False, ['K1_60_unbelastet'])] == \
        ['K1', 'K5', 'K6']
    # Hotspot rows are selected the same way.
    hot = [_hrow('K1', 'both', 'clt'), _hrow('K2', 'both', 'clt', qc_flags='x:1')]
    assert [r['knee'] for r in mod.select_rows(hot, False, [])] == ['K1']


def test_wilcoxon_is_reported_uncorrected_and_nan_for_tiny_n():
    from scipy.stats import wilcoxon
    deltas = [-0.1, -0.25, -0.3, -0.42, -0.5]
    rows = [_srow(f'K{i}', 'clt', d) for i, d in enumerate(deltas, start=1)]
    rows.append(_srow('K6', 'clt', np.nan))       # an ok row with a NaN aMe: not counted
    rows.append(_srow('K1', 'crt', -0.2))         # n = 1
    rows += [_srow(f'K{i}', 'elt', 0.0) for i in range(1, 4)]  # all zero
    summary = mod.subregion_summary(rows)
    assert set(mod.SUMMARY_COLUMNS) <= set(summary[0])
    assert 'wilcoxon_p_uncorrected' in mod.SUMMARY_COLUMNS

    def get(region, unit):
        [row] = [r for r in summary if r['region'] == region and r['unit'] == unit]
        return row

    clt = get('clt', 'mm')
    assert clt['table'] == 'subregion_delta'
    assert (clt['angle'], clt['load'], clt['bone']) == (0, 'unbelastet', 'tibia')
    assert clt['n'] == 5
    assert clt['mean'] == pytest.approx(np.mean(deltas))
    assert clt['sd'] == pytest.approx(np.std(deltas, ddof=1))
    assert clt['median'] == pytest.approx(-0.3)
    assert (clt['min'], clt['max']) == (pytest.approx(-0.5), pytest.approx(-0.1))
    assert clt['wilcoxon_p_uncorrected'] == pytest.approx(wilcoxon(deltas).pvalue)
    pct = get('clt', '%')
    assert pct['n'] == 5 and pct['mean'] == pytest.approx(100 * np.mean(deltas) / 2.0)
    assert get('crt', 'mm')['n'] == 1 and math.isnan(get('crt', 'mm')['wilcoxon_p_uncorrected'])
    assert math.isnan(get('crt', 'mm')['sd'])
    assert get('elt', 'mm')['n'] == 3 and math.isnan(get('elt', 'mm')['wilcoxon_p_uncorrected'])


def test_laterality_renames_and_mirrors():
    rows = [_srow('K1', 'clt', -0.1), _srow('K2', 'clt', -0.1),
            _srow('K1', 'alf', -0.1, bone='femur')]
    hot = [_hrow('K1', 'left', 'clt', x=3.0), _hrow('K1', 'both', 'crt', x=-2.0),
           _hrow('K2', 'left', 'clt', x=3.0), _hrow('K2', 'right', 'crt', x=-2.0),
           _hrow('K1', 'right', '', x=1.0)]
    table = {'K1': 'right', 'K2': 'left'}
    out = mod.to_anatomical(rows, table)
    # Right knee: the patient-left plate is medial; left knee: it is lateral.
    assert [r['subregion'] for r in out] == ['cMT', 'cLT', 'aMF']
    assert [r['anatomical_subregion'] for r in out] == ['cMT', 'cLT', 'aMF']
    assert rows[0]['subregion'] == 'clt'  # the input rows are not modified
    hout = mod.to_anatomical(hot, table)
    assert [r['compartment'] for r in hout] == ['medial', 'both', 'lateral', 'medial', 'lateral']
    assert [r['subregion'] for r in hout] == ['cMT', 'cLT', 'cLT', 'cMT', '']
    # Medial positive: a right knee's x is negated, a left knee's kept.
    assert [r['map_x_mm'] for r in hout] == [-3.0, 2.0, 3.0, -2.0, -1.0]
    assert all(r['map_y_mm'] == 1.0 for r in hout)
    # No table: the rows come back unchanged (patient side).
    assert mod.to_anatomical(hot, {}) == hot


def test_knee_missing_from_laterality_table_is_dropped_and_listed():
    rows = [_srow('K1', 'clt', -0.1), _srow('K3', 'clt', -0.1), _srow('K10', 'crt', -0.2)]
    table = {'K1': 'right'}
    assert mod.missing_laterality(rows, table) == ['K3', 'K10']
    assert [r['knee'] for r in mod.to_anatomical(rows, table)] == ['K1']
    assert mod.missing_laterality(rows, {}) == []  # without a table nothing is missing


def test_hotspot_frequency_counts_per_subregion():
    hot = [_hrow('K1', 'both', 'clt'), _hrow('K2', 'both', 'clt'), _hrow('K3', 'both', 'crt'),
           _hrow('K1', 'left', 'clt'), _hrow('K2', 'left', 'plt'),
           _hrow('K4', 'both', 'clt', angle=60),
           _hrow('K5', 'both', ''),
           _hrow('K1', 'both', 'cclf', bone='femur')]
    freq = mod.hotspot_frequency(hot)
    assert all(r['table'] == 'hotspot_frequency' for r in freq)

    def count(angle, bone, compartment, region):
        [row] = [r for r in freq if (r['angle'], r['bone'], r['compartment'], r['region'])
                 == (angle, bone, compartment, region)]
        return row['count'], row['n']

    assert count(0, 'tibia', 'both', 'clt') == (2, 4)
    assert count(0, 'tibia', 'both', 'crt') == (1, 4)
    assert count(0, 'tibia', 'both', 'none') == (1, 4)
    assert count(0, 'tibia', 'left', 'plt') == (1, 2)
    assert count(60, 'tibia', 'both', 'clt') == (1, 1)
    assert count(0, 'femur', 'both', 'cclf') == (1, 1)
    assert len(freq) == 7
    # Comparisons come out in reporting order.
    keys = [(r['angle'], r['load']) for r in freq]
    assert keys == sorted(keys, key=list(COMPARISONS).index)


def test_load_laterality_table(tmp_path):
    path = tmp_path / 'lat.csv'
    path.write_text('subject,laterality\nK1,Right\nK2, left \nK3,\n')
    assert mod.load_laterality_table(str(path)) == {'K1': 'right', 'K2': 'left'}
    assert mod.load_laterality_table(None) == {}
    path.write_text('subject,side\nK1,right\n')
    with pytest.raises(ValueError, match='subject,laterality'):
        mod.load_laterality_table(str(path))
    path.write_text('subject,laterality\nK1,up\n')
    with pytest.raises(ValueError, match='left or right'):
        mod.load_laterality_table(str(path))


def test_per_knee_rows_gain_anatomical_subregion_but_keep_patient_side_keys():
    rows = [_srow('K1', 'clt', -0.1), _srow('K2', 'clt', -0.1), _srow('K1', 'prf', 0.1,
                                                                       bone='femur'),
            _hrow('K1', 'both', '')]
    mod.fill_anatomical_subregion(rows, {'K1': 'left'})
    assert [r['subregion'] for r in rows] == ['clt', 'clt', 'prf', '']
    assert [r['anatomical_subregion'] for r in rows] == ['cLT', '', 'pMF', '']
    assert 'anatomical_subregion' in mod.HOTSPOT_COLUMNS


def _cohort_context(fake_delta_maps, tmp_path):
    """A per_knee_stage-shaped context for two synthetic knees (K1 complete, K2 flagged 60°)."""
    k1 = _process(_knee())
    k2_measurements = {key: dict(r, scan=r['scan'].replace('K1_', 'K2_'))
                       for key, r in _knee(configs=ALL_CONFIGS[:-1]).items()}
    k2_measurements[(60, 'unbelastet')]['qc_flags'] = 'tibia_transfer_far:0.02'
    k2 = mod.process_knee('K2', k2_measurements, 1.0, 3.0, 1.0)
    return {'paths': [str(tmp_path / f'K1_{a}_{l}.nii.gz') for a, l in ALL_CONFIGS],
            'knees': {'K1': {}, 'K2': {}}, 'records': {}, 'results': {'K1': k1, 'K2': k2},
            'provenance': {'commit': 'abc', 'dirty': False}}


def test_report_states_same_side_assumption_without_table(tmp_path, fake_delta_maps):
    context = _cohort_context(fake_delta_maps, tmp_path)
    out = tmp_path / 'out'
    out.mkdir()
    args = mod.build_parser().parse_args(['-i', str(tmp_path), '-o', str(out)])
    cohort = mod.cohort_stage(args, context, {})
    report = (out / 'report.md').read_text()
    assert 'same side' in report
    assert 'uncorrected' in report
    assert 'acquisition' in report  # Δ is acquisition-to-acquisition spread
    assert str(tmp_path) in report  # the label set used
    assert 'K2 90° loaded' in report  # the missing comparison is listed
    assert 'K2 60° unloaded' in report and 'tibia_transfer_far' in report  # flagged, listed
    assert (out / 'cohort_summary.csv').read_text().splitlines()[0].split(',') == \
        mod.SUMMARY_COLUMNS
    assert (out / 'tibia_hotspots_cohort.png').exists()
    assert (out / 'femur_hotspots_cohort.png').exists()
    import json
    meta = json.loads((out / 'report.meta.json').read_text())
    assert {'commit', 'dirty', 'branch', 'host', 'date', 'args', 'inputs'} <= set(meta)
    assert meta['n_knees'] == 2
    assert meta['hotspot_edge_margin_mm'] == 2.0
    # Flagged K2 60° unloaded is out of the statistics: the clt row of 60° unloaded has n = 1.
    [row] = [r for r in cohort['summary'] if r['table'] == 'subregion_delta'
             and (r['angle'], r['load'], r['bone'], r['region'], r['unit'])
             == (60, 'unbelastet', 'tibia', 'clt', 'mm')]
    assert row['n'] == 1
    # The 90° loaded comparison only has K1.
    [row] = [r for r in cohort['summary'] if r['table'] == 'subregion_delta'
             and (r['angle'], r['load'], r['bone'], r['region'], r['unit'])
             == (90, 'belastet', 'tibia', 'clt', 'mm')]
    assert row['n'] == 1

    # With a table: no same-side assumption, anatomical names, and K2 (not in it) listed.
    lat = tmp_path / 'lat.csv'
    lat.write_text('subject,laterality\nK1,right\n')
    args = mod.build_parser().parse_args(['-i', str(tmp_path), '-o', str(out),
                                          '--laterality-table', str(lat)])
    cohort = mod.cohort_stage(args, context, mod.load_laterality_table(str(lat)))
    report = (out / 'report.md').read_text()
    assert 'same side' not in report
    assert 'cMT' in report and 'K2' in report
    assert cohort['missing_laterality'] == ['K2']


def test_main_runs_both_stages_and_fills_the_per_knee_anatomical_column(tmp_path, monkeypatch,
                                                                         fake_delta_maps):
    data = tmp_path / 'segs'
    data.mkdir()
    for angle, load in ALL_CONFIGS:
        (data / f'K1_{angle}_{load}.nii.gz').write_bytes(b'')
    knee = _knee()

    def fake_measure(paths, *args, **kwargs):
        return {p: knee[mod.parse_scan_name(p)[1:]] for p in paths}

    monkeypatch.setattr(mod, 'measure_scans', fake_measure)
    lat = tmp_path / 'lat.csv'
    lat.write_text('subject,laterality\nK1,left\n')
    out = tmp_path / 'out'
    context = mod.main(['-i', str(data), '-o', str(out), '--workers', '1',
                        '--laterality-table', str(lat)])
    import csv as _csv
    with open(out / 'subregion_delta.csv') as handle:
        rows = list(_csv.DictReader(handle))
    clt = [r for r in rows if r['subregion'] == 'clt']
    assert clt and {r['anatomical_subregion'] for r in clt} == {'cLT'}  # left knee: lateral
    for name in ('cohort_summary.csv', 'report.md', 'report.meta.json',
                 'tibia_hotspots_cohort.png'):
        assert (out / name).exists()
    assert {r['compartment'] for r in context['cohort']['hotspot_rows']} == \
        {'both', 'medial', 'lateral'}


# ---------------------------------------------------------------------------------------
# Hotspot edge margin
# ---------------------------------------------------------------------------------------

def test_hotspot_edge_margin_reaches_the_search_and_the_rows(fake_delta_maps, tmp_path):
    # a uniform delta: the C-order-first searchable cell wins, so the margin decides which.
    # 12 x 12 footprint at 1 mm: the window rule alone leaves (1, 1), 2 mm from the edge;
    # the default 2 mm margin moves it to (2, 2), 3 mm from it.
    assert 'edge_distance_mm' in mod.HOTSPOT_COLUMNS
    default = mod.process_knee('K1', _knee(), 1.0, 3.0, 1.0)
    off = mod.process_knee('K1', _knee(), 1.0, 3.0, 1.0, edge_margin_mm=0.0)
    for result, expected in ((default, 3.0), (off, 2.0)):
        both = _rows(result, 'hotspot_rows', compartment='both', status='ok')
        assert both and {r['edge_distance_mm'] for r in both} == {expected}
    _, result = mod._knee_job(('K1', _knee(), 1.0, 3.0, 1.0, 0.0, tmp_path))
    assert {r['edge_distance_mm'] for r in _rows(result, 'hotspot_rows', compartment='both',
                                                  status='ok')} == {2.0}


def test_edge_margin_option_default_and_meta(tmp_path, fake_delta_maps):
    args = mod.build_parser().parse_args(['-i', 'x', '-o', str(tmp_path)])
    assert args.edge_margin_mm == 2.0
    args = mod.build_parser().parse_args(['-i', 'x', '-o', str(tmp_path),
                                          '--edge-margin-mm', '0'])
    assert args.edge_margin_mm == 0.0


# ---------------------------------------------------------------------------------------
# Reference unmapped fraction, cartilage cavities, report caveats
# ---------------------------------------------------------------------------------------

def test_registration_rows_carry_the_reference_unmapped_fraction(fake_delta_maps):
    assert mod.REGISTRATION_COLUMNS.index('reference_unmapped_fraction') == \
        mod.REGISTRATION_COLUMNS.index('dropped_fraction') + 1
    result = _process(_knee())
    rows = result['registration_rows']
    assert rows and all(r['reference_unmapped_fraction'] == 0.0 for r in rows)  # fake maps


def test_count_enclosed_cavities():
    array = np.zeros((20, 20, 20), dtype=np.int16)
    array[3:12, 3:12, 3:12] = 3            # a femoral cartilage block ...
    array[6, 6, 6] = 0                     # ... with one single-voxel cavity
    array[8:10, 8:10, 8] = 0               # ... and a second, 4 voxels
    array[11, 3:12, 3:12] = 0              # an open notch on the surface: not enclosed
    array[14:18, 14:18, 14:18] = 4         # tibial cartilage, solid
    assert mod.count_enclosed_cavities(array, 3) == 2
    assert mod.count_enclosed_cavities(array, 4) == 0
    assert mod.count_enclosed_cavities(array, 9) == 0          # absent label


def test_report_lists_hotspot_crowding_and_cavity_caveats(tmp_path, fake_delta_maps):
    context = _cohort_context(fake_delta_maps, tmp_path)
    context['records'] = {
        'a': {'scan': 'K1_30_unbelastet', 'cartilage_cavities': {'femur': 8, 'tibia': 4}},
        'b': {'scan': 'K1_0_belastet', 'cartilage_cavities': {'femur': 0, 'tibia': 0}},
        'c': {'scan': 'K1_60_belastet'},                       # an older cached record
    }
    out = tmp_path / 'out'
    out.mkdir()
    args = mod.build_parser().parse_args(['-i', str(tmp_path), '-o', str(out)])
    cohort = mod.cohort_stage(args, context, {})
    report = (out / 'report.md').read_text()
    assert '## Open caveats' in report
    # fake maps: every hotspot is 3.0 mm from an edge -> all within margin + 1 mm
    assert cohort['crowding']['femur']['n_near'] == cohort['crowding']['femur']['n']
    # Only the left/right hotspots count (the whole-bone one duplicates one of them).
    sides = [r for r in cohort['hotspot_rows'] if r['bone'] == 'femur'
             and r['compartment'] in ('left', 'right') and r['status'] == 'ok']
    assert cohort['crowding']['femur']['n'] == len(sides) > 0
    assert 'Left/right compartment hotspots' in report
    assert cohort['crowding']['femur']['median_mm'] == 3.0
    assert 'K1_30_unbelastet (femur 8, tibia 4)' in report
    assert cohort['cavities'] == [('K1_30_unbelastet', 8, 4)]
    assert 'reference unmapped' in report


# ---------------------------------------------------------------------------------------
# Final-review fixes: unmapped causes, maps-only failures, provenance, cache, report wording
# ---------------------------------------------------------------------------------------

def test_registration_rows_split_the_reference_unmapped_fraction_by_cause(fake_delta_maps):
    from morphometry.cartilage.deformation import UNMAPPED_CAUSES
    columns = [f'reference_unmapped_{cause}' for cause in UNMAPPED_CAUSES]
    assert columns == [c for c in mod.REGISTRATION_COLUMNS if c.startswith('reference_unmapped_')
                       and c != 'reference_unmapped_fraction']
    rows = [{'knee': 'K1', 'bone': 'femur', 'reference_unmapped_fraction': 0.03,
             'reference_unmapped_detour': 0.02, 'reference_unmapped_no_crossing': 0.01,
             'reference_unmapped_no_profile_point': 0.0, 'reference_unmapped_off_bone': 0.0},
            {'knee': 'K2', 'bone': 'femur', 'reference_unmapped_fraction': 0.01,
             'reference_unmapped_detour': 0.0, 'reference_unmapped_no_crossing': 0.004,
             'reference_unmapped_no_profile_point': 0.006, 'reference_unmapped_off_bone': 0.0}]
    summary = mod.reference_unmapped(rows)['femur']
    assert summary['max'] == 0.03 and summary['max_knee'] == 'K1'
    assert summary['causes']['detour']['max'] == 0.02
    assert summary['causes']['no_profile_point']['max_knee'] == 'K2'
    result = _process(_knee())
    assert all(column in row for row in result['registration_rows'] for column in columns)


def test_report_lists_maps_only_failures_and_states_the_one_sided_map_thickness(
        tmp_path, fake_delta_maps):
    fake_delta_maps['tibia']['errors'] = {(0, 'belastet'): 'RuntimeError: ICP diverged'}
    context = _cohort_context(fake_delta_maps, tmp_path)
    out = tmp_path / 'out'
    out.mkdir()
    args = mod.build_parser().parse_args(['-i', str(tmp_path), '-o', str(out)])
    cohort = mod.cohort_stage(args, context, {})
    report = (out / 'report.md').read_text()
    assert [(r['knee'], r['bone']) for r in cohort['maps_failed']
            if (r['angle'], r['load']) == (0, 'belastet')] == [('K1', 'tibia'), ('K2', 'tibia')]
    assert 'Maps and hotspots failed, subregion Δ kept' in report
    assert 'K1 0° loaded (tibia)' in report
    # The registration error does not take the subregion rows out of the statistics.
    [row] = [r for r in cohort['summary'] if r['table'] == 'subregion_delta'
             and (r['angle'], r['load'], r['bone'], r['region'], r['unit'])
             == (0, 'belastet', 'tibia', 'clt', 'mm')]
    assert row['n'] == 2
    assert 'cMe direction' in report and 'not directly comparable' in report
    assert '140 p-values per bone (280 over both bones)' in report


def test_cache_entry_that_cannot_be_read_is_recomputed(tmp_path, monkeypatch):
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
    cache_file = out / 'cache' / 'K1_0_belastet.pkl'
    cache_file.parent.mkdir(parents=True)
    bad_entries = [b'not a pickle at all',                       # UnpicklingError
                   pickle.dumps(['a', 'list']),                    # not a dict
                   pickle.dumps({'record': 'not a dict'}),         # malformed record
                   pickle.dumps(SimpleNamespace(x=1))[:-3],        # truncated
                   b'\x80\x04\x95' + b'\xff' * 20]                 # garbage after a header
    for n, content in enumerate(bad_entries, start=1):
        cache_file.write_bytes(content)
        records = mod.measure_scans([str(scan)], out, workers=1, use_cache=True,
                                    provenance=provenance, labels=labels)
        assert len(calls) == n and records[str(scan)]['scan'] == 'K1_0_belastet'


def test_provenance_counts_untracked_code_as_dirty(monkeypatch):
    answers = {}

    def fake_git(*args):
        if args[0] == 'rev-parse':
            return 'abc123'
        return answers['untracked' if '--untracked-files=all' in args else 'tracked']

    monkeypatch.setattr(mod, '_git', fake_git)
    answers.update(tracked='', untracked='')
    assert mod.provenance() == {'commit': 'abc123', 'dirty': False}
    answers.update(tracked='', untracked='?? morphometry/cartilage/new_module.py')
    assert mod.provenance()['dirty'] is True
    answers.update(tracked=' M scripts/x.py', untracked='')
    assert mod.provenance()['dirty'] is True
    answers.update(tracked='unknown', untracked='')
    assert mod.provenance()['dirty'] is True

    calls = []

    def recording_git(*args):
        calls.append(args)
        return 'abc123' if args[0] == 'rev-parse' else ''

    monkeypatch.setattr(mod, '_git', recording_git)
    mod.provenance()
    untracked = [a for a in calls if '--untracked-files=all' in a]
    assert untracked and untracked[0][-2:] == mod.PROVENANCE_CODE_DIRS


def test_meta_records_the_runs_own_provenance(tmp_path, fake_delta_maps, monkeypatch):
    context = _cohort_context(fake_delta_maps, tmp_path)
    context['provenance'] = {'commit': 'deadbeef', 'dirty': True}
    monkeypatch.setattr(mod, '_git', lambda *args: 'from-a-second-git-call')
    out = tmp_path / 'out'
    out.mkdir()
    args = mod.build_parser().parse_args(['-i', str(tmp_path), '-o', str(out)])
    mod.cohort_stage(args, context, {})
    import json
    meta = json.loads((out / 'report.meta.json').read_text())
    assert meta['commit'] == 'deadbeef' and meta['dirty'] is True
