"""Characterization of the knee-cartilage deformation (thickness-change) pipeline on real data.

Runs the per-scan measurement of ``scripts/analyse_cartilage_deformation.py``
(:func:`measure_scan`: LPI, no ``remove_outliers()``, both bones' subdivision and per-triangle
thickness) on knee ``K1`` of the **current** ``dfg_knees`` label set
(:data:`tests.data_paths.DFG_KNEES_CURRENT_DIR`, override with ``MORPH_DFG_KNEES_CURRENT_DIR``;
the module skips when it is absent), then the library's
:func:`~morphometry.cartilage.deformation.knee_delta_maps` against the reference
(unloaded 30 degrees). Two scans are measured once per module (~90 s for the module).

What is pinned, and why it is not a golden: the pipeline's numbers move with any change to
the subdivision or the thickness, which have their own goldens
(``test_knee_cartilage.py``). These tests pin *properties* the deformation pipeline adds:

* the reference compared with itself registers to the identity and gives a delta of exactly 0
  on every cell of both bones (the registration and the map construction add no error of
  their own);
* the loaded 30 degree scan of the same knee (same flexion, a different acquisition) registers
  below :data:`~morphometry.cartilage.registration.REGISTRATION_MAX_RMS_MM`, converges, and
  its median per-cell delta is under 0.3 mm in magnitude on both bones -- a gross
  registration or mapping error (a flipped bone, a shifted coordinate) would put the whole
  map off by millimetres;
* the femoral map has no strip of side-wall-anchored levels (the K1 defect fixed with
  :data:`~morphometry.cartilage.deformation.PROFILE_MAX_DETOUR_MM`): before the fix 64.8 mm^2
  of K1's covered reference tAB read an arc beyond 50 mm, after it 3.6 mm^2;
* every hotspot of the loaded 30 degree comparison lies more than the default edge margin
  from both scans' footprint edges.
"""
import importlib.util
import pathlib
import warnings

import numpy as np
import pytest

from morphometry.cartilage.deformation import (
    COMPARTMENTS, DEFAULT_HOTSPOT_EDGE_MARGIN_MM, REFERENCE, UnrolledFemur,
    find_largest_thinning, knee_delta_maps,
)
from morphometry.cartilage.registration import REGISTRATION_MAX_RMS_MM
from morphometry.cartilage.surface import triangle_areas, triangle_centroids
from tests import data_paths

pytestmark = pytest.mark.needs_dfg_knees

_SCRIPT = (pathlib.Path(__file__).resolve().parents[2] / 'scripts'
           / 'analyse_cartilage_deformation.py')

LOADED_30 = (30, 'belastet')
BONES = ('tibia', 'femur')


def _script():
    """Import the deformation script as a module (it is not a package member)."""
    spec = importlib.util.spec_from_file_location('analyse_cartilage_deformation', _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope='module')
def k1_records():
    """
    ``{(angle, load): measure_scan record}`` of K1's unloaded and loaded 30 degree scans.

    :return: The two records; skips the module when the current label set or a scan is absent.
    """
    directory = data_paths.DFG_KNEES_CURRENT_DIR
    if not directory.exists():
        pytest.skip(f'current dfg_knees label set not found at {directory} '
                    '(set MORPH_DFG_KNEES_CURRENT_DIR)')
    script = _script()
    records = {}
    for angle, load in (REFERENCE, LOADED_30):
        path = directory / f'K1_{angle}_{load}.nii.gz'
        if not path.exists():
            pytest.skip(f'{path} not found')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            record = script.measure_scan(str(path))
        for bone in BONES:
            assert record[bone] is not None, record['errors']
        records[(angle, load)] = record
    return records


@pytest.fixture(scope='module')
def k1_maps(k1_records):
    """
    Per bone, the delta maps of K1 against its reference for two configurations: the
    reference itself (keyed ``REFERENCE``) and the loaded 30 degree scan.

    :return: ``{bone: BoneDeltaMaps}``.
    """
    reference = k1_records[REFERENCE]
    return {bone: knee_delta_maps(bone, reference[bone],
                                  {REFERENCE: reference[bone],
                                   LOADED_30: k1_records[LOADED_30][bone]})
            for bone in BONES}


@pytest.mark.parametrize('bone', BONES)
def test_reference_against_itself_is_exactly_zero(k1_maps, bone):
    """The reference registered onto itself: identity transform, every delta exactly 0."""
    maps = k1_maps[bone]
    assert not maps.errors
    result = maps.registrations[REFERENCE]
    assert result.converged
    assert np.allclose(result.transform.rotation, np.eye(3), atol=1e-9)
    assert np.allclose(result.transform.translation, 0.0, atol=1e-9)
    assert result.median_mm < 1e-6
    delta = maps.deltas[REFERENCE]
    footprint = np.isfinite(maps.reference)
    assert footprint.sum() > 1000
    assert np.array_equal(np.isfinite(delta), footprint)
    assert np.all(delta[footprint] == 0.0)
    assert maps.dropped_fraction[REFERENCE] == pytest.approx(maps.reference_dropped_fraction)


@pytest.mark.parametrize('bone', BONES)
def test_k1_loaded_30_registers_and_stays_plausible(k1_maps, bone):
    """Same flexion, other load: a clean bone registration and a small median delta."""
    maps = k1_maps[bone]
    result = maps.registrations[LOADED_30]
    assert result.converged
    assert result.rms_mm < REGISTRATION_MAX_RMS_MM
    delta = maps.deltas[LOADED_30]
    finite = delta[np.isfinite(delta)]
    assert finite.size > 1000
    assert abs(float(np.median(finite))) < 0.3


def test_k1_femoral_map_has_no_side_wall_strip(k1_records):
    """
    No covered reference tAB beyond an arc of 50 mm (K1's posterior condyles end at ~48 mm;
    the side-wall-anchored levels put 64.8 mm^2 of it at 50-85 mm before the fix).
    """
    femur = k1_records[REFERENCE]['femur']
    unrolled = UnrolledFemur.from_reference(femur.bone_mesh, femur.femoral_frame)
    covered = np.asarray(femur.tab_mesh.cell_data['covered'], dtype=bool)
    coords = unrolled.coordinates(triangle_centroids(femur.tab_mesh)[covered])
    areas = triangle_areas(femur.tab_mesh)[covered]
    beyond = np.nan_to_num(coords[:, 1], nan=-np.inf) > 50.0
    assert areas[beyond].sum() < 10.0


@pytest.mark.parametrize('bone', BONES)
def test_k1_hotspots_keep_off_the_footprint_rims(k1_maps, bone):
    """Every compartment's hotspot is more than the edge margin inside both footprints."""
    for compartment in COMPARTMENTS:
        hotspot = find_largest_thinning(k1_maps[bone], LOADED_30, compartment)
        assert hotspot is not None and not hotspot['fallback']
        assert hotspot['edge_distance_mm'] > DEFAULT_HOTSPOT_EDGE_MARGIN_MM


def test_k1_reference_unmapped_area_is_attributed_and_not_cut_proximally(k1_records, k1_maps):
    """
    The femoral reference's unmapped covered tAB is split by cause in full (nothing off the
    bone), and none of it lies beyond the registration crop: the unrolled coordinate is built
    on the whole bone, not on the distal ``REGISTRATION_CROP_MM`` (final review, finding 1).
    """
    from morphometry.cartilage.deformation import UNMAPPED_CAUSES
    from morphometry.cartilage.registration import REGISTRATION_CROP_MM
    maps = k1_maps['femur']
    causes = maps.reference_unmapped_by_cause
    assert set(causes) == set(UNMAPPED_CAUSES)
    assert sum(causes.values()) == pytest.approx(maps.reference_dropped_fraction, abs=1e-12)
    assert causes['off_bone'] == 0.0
    assert maps.reference_dropped_fraction < 0.05
    femur = k1_records[REFERENCE]['femur']
    frame = femur.femoral_frame
    unrolled = UnrolledFemur.from_reference(femur.bone_mesh, frame)
    covered = np.asarray(femur.tab_mesh.cell_data['covered'], dtype=bool)
    centroids = triangle_centroids(femur.tab_mesh)[covered]
    distal = (np.asarray(femur.bone_mesh.points) @ frame.proximal_axis).min()
    high = centroids @ frame.proximal_axis - distal > REGISTRATION_CROP_MM - 1.0
    assert high.any()
    assert np.isfinite(unrolled.coordinates(centroids[high])).all()
    assert all(m.reference_unmapped_by_cause == dict.fromkeys(UNMAPPED_CAUSES, 0.0)
               for m in (k1_maps['tibia'],))
