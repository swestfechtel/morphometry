"""Characterization of the knee cartilage subregions and thickness (Wirth & Eckstein 2008).

These are the first committed reference values for any knee cartilage number: steps 1-4 of
``docs/wirth_eckstein_2008_plan.md`` are covered by phantom unit tests
(``tests/unit/test_cartilage_{surface,thickness}.py``, ``test_{tibia,femur}_regions.py``) and by
cohort tables written into that document, but nothing pinned the output on a real scan. A
refactor could therefore move every cohort number without a single test failing.

Two scans are pinned, both ``dfg_knees`` (labels 1 = femur bone, 2 = tibia bone,
3 = femoral cartilage, 4 = tibial cartilage):

* ``K1_0_unbelastet`` (:data:`tests.data_paths.DFG_KNEES_GOLDEN_SCAN`) -- a clean, unloaded,
  extended knee with no QC flag, the scan the plan document quotes its old-vs-new thickness
  comparison on (``clt`` 1.800 -> 2.487 mm, ``cclf`` 1.976 -> 3.236 mm), so the goldens here
  are directly comparable with that text. Everything is pinned on it: the tibial plate frames,
  both bones' region areas and shares, the full thickness statistics of all 20 subregions, and
  the thinnest-point search in all three compartments.
* ``K1_90_unbelastet`` -- the 90 deg flexion scan of the same subject and load, pinned on a
  cheaper subset (region shares and areas, the femoral frame, ``aMe`` per subregion and the
  thinnest point). It costs ~15 s and is worth it: at 90 deg of flexion the femoral frame, the
  posterior-condyle zones, ``_restrict_to_contact_cluster`` and the shared in-plane frame all
  do work that an extended knee never exercises -- the posterior zones hold 76% of the
  tibiofemoral overlap there against 9% at 0 deg (plan step 3). A regression confined to the
  flexed geometry would otherwise pass unnoticed.

Not pinned, deliberately:

* ``n_tab_triangles`` / ``n_ac_triangles`` -- mesh triangle counts from marching cubes and the
  exact mesh splitting. They are integers with no tolerance to give, and they would turn a
  harmless ``scikit-image`` / VTK upgrade into a golden failure while adding nothing the areas
  do not already cover.
* ``CondyleFrame.n_slices``, for the same reason: it is how many 0.5 mm cross-sections fit in
  the weight-bearing A-P window, i.e. a quantisation of a continuous length, and the window has
  only 0.049 mm of headroom at 0 deg (0.139 at 90). A change of a twentieth of a millimetre --
  a sixth of an in-plane voxel, well inside what a marching-cubes or smoothing change can do --
  flips 31 to 30 without any measurement moving. It takes 13 distinct values (29-43) over the
  cohort, so it is genuinely mobile; ``FemoralFrame.shaft_bins`` is pinned instead, being 10 on
  all 96 scans.
* the *neighbourhood centre* of the flexed scan's thinnest-point search (see
  :func:`_thinnest_values`).
* The region *point clouds* (``Tibia.clt`` etc.): voxel counts of the transferred regions move
  with the fragment-removal safety net (plan open issue 5) and say nothing about the
  measurement, which is computed on the surfaces.

Scans are measured once per module run and cached: a subdivision plus both bones' thickness is
~15 s, and the measurement is deterministic (the only stochastic step, the two-means plate-split
fallback in ``find_plates``, does not fire on any ``dfg_knees`` scan -- and the autouse seed
fixture in ``conftest`` covers the first call anyway).
"""
import warnings

import numpy as np
import pytest

from morphometry.cartilage.femur_regions import (
    FEMUR_REGION_KEYS, ZONE_THICKNESS_KEYS, articular_arc_fraction,
)
from morphometry.cartilage.knee import (
    Femur, Tibia, find_thinnest_area, find_thinnest_area_by_compartment, scan_qc_flags,
    thinnest_point_position,
)
from morphometry.cartilage.thickness import STATISTIC_KEYS
from morphometry.cartilage.tibia_regions import TIBIA_REGION_KEYS
from tests import data_paths
from tests.conftest import assert_golden

pytestmark = pytest.mark.needs_dfg_knees

SCAN = data_paths.DFG_KNEES_GOLDEN_SCAN
SCAN_FLEXED = data_paths.DFG_KNEES_GOLDEN_SCAN_FLEXED

# Labels of the dfg_knees segmentations.
FEMUR_BONE, TIBIA_BONE = 1, 2
FEMUR_CARTILAGE, TIBIA_CARTILAGE = 3, 4

# The femoral thickness maps key the two extension zones by their long names.
FEMUR_THICKNESS_KEYS = tuple(ZONE_THICKNESS_KEYS.get(key, key) for key in FEMUR_REGION_KEYS)

# The statistics pinned per subregion: everything measure_thickness reports except the two
# triangle counts (see the module docstring).
PINNED_STATISTICS = tuple(k for k in STATISTIC_KEYS
                          if k not in ('n_tab_triangles', 'n_ac_triangles'))

# Thinnest-point results are reported for the whole knee and both patient-side compartments.
COMPARTMENTS = ('both', 'left', 'right')

_MEASURED: dict = {}


def _measure(seg) -> dict:
    """Run the whole knee cartilage pipeline on one loaded scan.

    :param seg: The LPI-oriented knee ``Segmentation``.
    :return: ``{'tibia', 'femur', 'tibia_thickness', 'femur_thickness', 'thinnest',
        'qc_flags'}``, where ``thinnest`` maps ``'both'`` / ``'left'`` / ``'right'`` to the
        :func:`~morphometry.cartilage.knee.find_thinnest_area` result of that compartment,
        each extended by the ``'arc_fraction'`` of its point along the femoral articular
        surface.
    """
    tibia = Tibia(seg, cartilage_label=TIBIA_CARTILAGE, bone_label=TIBIA_BONE)
    tibia_thickness = tibia.calculate_thickness()
    femur = Femur(seg, cartilage_label=FEMUR_CARTILAGE, bone_label=FEMUR_BONE)
    femur_thickness = femur.calculate_thickness(tibia)

    thinnest = {'both': find_thinnest_area(tibia_thickness, femur_thickness, seg)}
    thinnest.update(find_thinnest_area_by_compartment(tibia_thickness, femur_thickness, seg))
    for result in thinnest.values():
        world = thinnest_point_position(tibia_thickness, femur_thickness, result['point'])
        result['arc_fraction'] = float(
            articular_arc_fraction(femur.subdivision, world)[0])
    return {'tibia': tibia, 'femur': femur, 'tibia_thickness': tibia_thickness,
            'femur_thickness': femur_thickness, 'thinnest': thinnest,
            'qc_flags': scan_qc_flags(seg, tibia, femur)}


@pytest.fixture
def measured(dfg_knee):
    """A cached accessor for the measured knees.

    :return: A function ``get(scan_name) -> dict`` (see :func:`_measure`); the scan is loaded
        and measured on first use and reused by the later tests of this module.
    """

    def get(name: str) -> dict:
        if name not in _MEASURED:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                _MEASURED[name] = _measure(dfg_knee(name))
        return _MEASURED[name]

    return get


def _plate_frame_values(tibia: Tibia) -> list:
    """The key scalars of both tibial plate frames, patient left then patient right.

    Per plate, in order: the centre of gravity S (3), the principal axes ``b1, b2, b3``
    row-major (9), the area-weighted variances along them (3), the in-plane internal and
    anterior directions (3 + 3), the central-region level ``rho*`` (1), the ellipse semi-axes
    (2) and the plate's tAB and cAB areas (2) -- 26 numbers per plate.

    The two in-plane variances are well separated on a tibial plate (K1_0_unbelastet: 90.5 vs
    43.9 mm^2 patient-left, 133.8 vs 61.1 patient-right), so the eigenvector order, and with it
    the sign conventions ``b3`` superior / ``b1`` towards the other plate, is not a near tie.

    :param tibia: The measured ``Tibia``.
    :return: The flat list of 52 values.
    """
    values: list = []
    for side in ('left', 'right'):
        plate = tibia.subdivision.plates[side]
        values += list(plate.centroid) + list(np.asarray(plate.axes).ravel())
        values += list(plate.variances) + list(plate.ml_direction)
        values += list(plate.anterior_direction) + [plate.ellipse_level]
        values += list(plate.semi_axes) + [plate.tab_area, plate.cab_area]
    return values


def _femur_frame_values(femur: Femur) -> list:
    """The key scalars of the femoral frame and of both condyle frames.

    In order: the proximal, medial-lateral and anterior-posterior axes (3 x 3), the trochlear
    notch point (3) and its A-P coordinate (1), the condyle divider's slope and intercept (2),
    the shaft-fit residual and bin count (2), the whole femoral tAB area (1), and per condyle
    (patient left then right) the posterior and 60% border A-P coordinates, the cF tAB and cAB
    areas and the trochlea and posterior-zone areas (6 each).

    ``CondyleFrame.n_slices`` is **not** pinned (see the module docstring); ``shaft_bins`` is,
    because it is 10 on all 96 scans of the cohort.

    :param femur: The measured ``Femur``.
    :return: The flat list of 30 values.
    """
    frame = femur.subdivision.frame
    values = (list(frame.proximal_axis) + list(frame.ml_axis) + list(frame.ap_axis)
              + list(frame.notch_point) + [frame.notch_ap]
              + list(frame.divider) + [frame.shaft_residual_mm, frame.shaft_bins]
              + [femur.subdivision.tab_area])
    for side in ('left', 'right'):
        condyle = femur.subdivision.condyles[side]
        values += [condyle.posterior_ap, condyle.boundary_ap, condyle.tab_area,
                   condyle.cab_area, condyle.anterior_area, condyle.posterior_area]
    return values


def _statistics_values(thickness, keys) -> list:
    """Every pinned thickness statistic of every subregion, subregion-major.

    :param thickness: One bone's ``ThicknessMaps``.
    :param keys: The subregion keys, in the order they are reported.
    :return: ``len(keys) * len(PINNED_STATISTICS)`` values (NaN where a statistic is missing).
    """
    return [thickness.statistics[key].get(stat, np.nan)
            for key in keys for stat in PINNED_STATISTICS]


def _thinnest_values(measurement: dict, include_center: bool = True) -> list:
    """The thinnest-point result of all three compartments.

    Per compartment, in order: the thinnest cell ``(u, v)``, optionally the neighbourhood
    centre ``(u, v)``, the smoothed ``center_thickness``, the raw ``combined_thickness``, the
    tibial and femoral contributions, the median kernel size and the point's arc fraction along
    the femoral articular surface -- 10 numbers per compartment with the centre, 8 without.

    :param measurement: The scan's measurement bundle (see :func:`_measure`).
    :param include_center: Whether to pin the neighbourhood centre as well. ``False`` on the
        flexed scan, where the smoothed minimum is attained **bitwise identically** at six
        cells spanning 2.5 mm, so which of them is returned is decided by the C-order
        traversal of the grid and not by the data: a pinned centre would fail on any upstream
        reordering while meaning nothing. The returned ``point`` is refined inside the disc
        around it and is stable, so it stays pinned.
    :return: The flat list of 30 values (24 without the centres).
    """
    values: list = []
    for compartment in COMPARTMENTS:
        result = measurement['thinnest'][compartment]
        values += [float(result['point'][0]), float(result['point'][1])]
        if include_center:
            values += [float(result['center'][0]), float(result['center'][1])]
        values += [result['center_thickness'], result['combined_thickness'],
                   result['tibial_thickness'], result['femoral_thickness'],
                   float(result['kernel_size']), result['arc_fraction']]
    return values


# ------------------------------------------------------------------ K1_0_unbelastet (0 deg)

def test_scan_is_clean(measured):
    """The pinned scan carries no QC flag, so its goldens are not pinned on a broken scan."""
    assert measured(SCAN)['qc_flags'] == ''


def test_tibia_region_shares(measured):
    """The five regions' shares of their own plate's tAB (the paper's Table I quantity)."""
    tibia = measured(SCAN)['tibia']
    shares = tibia.subdivision.region_shares
    assert_golden('knee_tibia_region_shares', [shares[k] for k in TIBIA_REGION_KEYS])


def test_tibia_region_areas(measured):
    """The ten tibial subregions' tAB areas in mm^2, plus the area dropped outside both plates."""
    subdivision = measured(SCAN)['tibia'].subdivision
    areas = subdivision.region_areas
    assert_golden('knee_tibia_region_areas',
                  [areas[k] for k in TIBIA_REGION_KEYS] + [subdivision.dropped_area])


def test_tibia_plate_frames(measured):
    """The two plate frames: S, the principal axes and variances, rho*, the semi-axes, areas."""
    assert_golden('knee_tibia_plate_frames', _plate_frame_values(measured(SCAN)['tibia']))


def test_femur_region_shares(measured):
    """Each third's share of its own weight-bearing condyle's tAB (the paper's Table II)."""
    shares = measured(SCAN)['femur'].subdivision.region_shares
    assert_golden('knee_femur_region_shares',
                  [shares[k] for k in FEMUR_REGION_KEYS if k in shares])


def test_femur_region_areas(measured):
    """The ten femoral subregions' tAB areas in mm^2 (thirds, trochlea, posterior condyles)."""
    areas = measured(SCAN)['femur'].subdivision.region_areas
    assert_golden('knee_femur_region_areas', [areas[k] for k in FEMUR_REGION_KEYS])


def test_femur_frame(measured):
    """The femoral frame: axes, trochlear notch, the 60% borders, the shaft fit, cF areas."""
    assert_golden('knee_femur_frame', _femur_frame_values(measured(SCAN)['femur']))


def test_tibia_thickness_statistics(measured):
    """All pinned ThC statistics (cAB and tAB families, areas, denuded fraction) per region."""
    assert_golden('knee_tibia_thickness_statistics',
                  _statistics_values(measured(SCAN)['tibia_thickness'], TIBIA_REGION_KEYS))


def test_femur_thickness_statistics(measured):
    """The same for the femur's six thirds and four extension zones."""
    assert_golden('knee_femur_thickness_statistics',
                  _statistics_values(measured(SCAN)['femur_thickness'], FEMUR_THICKNESS_KEYS))


def test_thinnest_point(measured):
    """The thinnest-point search in the whole knee and both compartments, with its arc position.

    This is the most sensitive golden in the module by construction: the search takes an
    ``argmin`` over a smoothed map, so a change anywhere upstream can move the chosen cell a
    long way rather than a little. That is intended -- it is exactly the quantity
    ``scripts/analyse_thinnest_point_flexion.py`` reports, and the plan document records that
    several earlier fixes moved it on nearly every scan.
    """
    assert_golden('knee_thinnest_point', _thinnest_values(measured(SCAN)))


# --------------------------------------------------------------- K1_90_unbelastet (90 deg)

def test_flexed_region_shares(measured):
    """Tibial and femoral region shares of a 90 deg flexed knee (the femoral frame's hard case)."""
    measurement = measured(SCAN_FLEXED)
    tibia_shares = measurement['tibia'].subdivision.region_shares
    femur_shares = measurement['femur'].subdivision.region_shares
    assert_golden('knee90_region_shares',
                  [tibia_shares[k] for k in TIBIA_REGION_KEYS]
                  + [femur_shares[k] for k in FEMUR_REGION_KEYS if k in femur_shares])


def test_flexed_region_areas(measured):
    """The flexed knee's twenty subregion tAB areas in mm^2 (tibia first, then femur)."""
    measurement = measured(SCAN_FLEXED)
    tibia_areas = measurement['tibia'].subdivision.region_areas
    femur_areas = measurement['femur'].subdivision.region_areas
    assert_golden('knee90_region_areas',
                  [tibia_areas[k] for k in TIBIA_REGION_KEYS]
                  + [femur_areas[k] for k in FEMUR_REGION_KEYS])


def test_flexed_femur_frame(measured):
    """The flexed knee's femoral frame -- built from the bone, so it must not follow flexion."""
    assert_golden('knee90_femur_frame', _femur_frame_values(measured(SCAN_FLEXED)['femur']))


def test_flexed_thickness_mean(measured):
    """``aMe`` (ThC.cAB) of all twenty subregions of the flexed knee.

    The posterior-condyle zones are the point of this golden: their thickness used to be
    measured along an image axis that crossed the whole condyle at 90 deg of flexion (plan
    step 3), and only a flexed scan can catch that coming back.
    """
    measurement = measured(SCAN_FLEXED)
    tibia_statistics = measurement['tibia_thickness'].statistics
    femur_statistics = measurement['femur_thickness'].statistics
    assert_golden('knee90_thickness_mean',
                  [tibia_statistics[k]['aMe'] for k in TIBIA_REGION_KEYS]
                  + [femur_statistics[k]['aMe'] for k in FEMUR_THICKNESS_KEYS])


def test_flexed_thinnest_point(measured):
    """The flexed knee's thinnest point in all three compartments.

    Same caveat as :func:`test_thinnest_point`: an ``argmin`` can move a long way.
    """
    assert_golden('knee90_thinnest_point',
                  _thinnest_values(measured(SCAN_FLEXED), include_center=False))
