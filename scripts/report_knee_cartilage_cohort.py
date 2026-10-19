"""Regenerate the cohort tables of the knee cartilage subregions (Wirth & Eckstein 2008).

Every cohort number quoted in ``CLAUDE.md`` and ``docs/wirth_eckstein_2008_plan.md`` about the
``dfg_knees`` knees -- the Table I / Table II analogues, the subregion areas, the thickness per
subregion, the loaded-vs-unloaded and flexion spreads, the QC counts -- came out of ad-hoc
sessions. This script derives all of them from a cohort in one run, so they can be re-checked
after any change to :mod:`morphometry.cartilage`.

It measures each scan exactly as ``scripts/compute_knee_cartilage_thickness.py`` does (load,
reorient to LPI, no outlier removal; the subregions and the thickness come from
:class:`~morphometry.cartilage.knee.Tibia` / ``Femur``), then aggregates:

* **Table I analogue** -- the five tibial regions' shares of their own plate's tAB, per plate,
  mean / SD / min / max, next to the paper's published values;
* **Table II analogue** -- the three thirds' shares of their own weight-bearing condyle's tAB;
* **areas** -- tAB and cAB per subregion, per plate / condyle and per whole bone, and the
  denuded fraction;
* **thickness** -- ThC.cAB ``aMe`` / ``aSD`` / ``aMin`` and ThC.tAB ``aMe`` per subregion, plus
  the whole-bone ``aMe`` per flexion angle;
* **reproducibility** -- loaded vs unloaded for one subject and angle, and the spread across
  the four flexion angles of one subject and load. These are **spread, not precision**: the
  scans are different acquisitions of a knee in different positions and loading states, not
  repeat scans, so they are not comparable with the paper's 19-84 um test-retest rms SD;
* **QC** -- which scans failed, which carry which :func:`~morphometry.cartilage.knee.scan_qc_flags`
  flag, and the geometry checks behind them (plate split, transfer distances, dropped islands).

**Laterality.** The regions are keyed by patient side (``clt`` = patient-left plate), and the
``dfg_knees`` knees' laterality is not recorded anywhere, so the paper's medial / lateral
columns cannot be matched to ours. The paper's values are printed alongside as a sanity check
only; which of our plates is MT and which LT is unknown (plan open issue 3).

**Which scans are aggregated.** By default only scans that measured without an error *and*
carry no QC flag -- on ``dfg_knees`` that is 93 of 96, excluding ``K7_90_belastet`` and
``K8_90_belastet`` (non-LPI) and ``K11_90_belastet`` (broken segmentation, its tibia raises).
``--include-flagged`` keeps the flagged ones and ``--exclude`` drops named scans, which is how
an older table computed over a different set can be reproduced (e.g. the tibial shares in
``CLAUDE.md`` were computed over 94 scans: ``--include-flagged --exclude K7_90_belastet
K11_90_belastet``). Failures never abort the run and always appear in the QC section.

Outputs (into ``-o``, created if needed):

* ``scan_metrics.csv`` -- long format, one row per (scan, bone, region): share, tAB / cAB area,
  denuded fraction and the thickness statistics. ``region`` also carries the aggregates
  ``plate_left`` / ``plate_right`` / ``cf_left`` / ``cf_right`` / ``whole``.
* ``scan_geometry.csv`` -- long format, one row per (scan, metric): the per-scan geometry and QC
  scalars (A-P extent, shaft fit, plate ratio, transfer distances, dropped area, timings).
* ``cohort_summary.csv`` -- long format, one row per statistic: ``table, scope, region, unit, n,
  mean, sd, min, max, p50, p90, paper``. Every number in the markdown report is in here. It
  holds no timings, so two runs that computed the same numbers produce the same file whatever
  ``--workers`` was.
* ``report.md`` -- the human-readable tables.
* ``report.meta.json`` -- provenance: the git commit and whether the checkout was dirty, the
  branch, host, date, the full argument list, the inputs, the scan counts (measured,
  aggregated, excluded by flag or by name, failed) and the per-bone wall-clock timings.

Examples
--------
::

    python scripts/report_knee_cartilage_cohort.py \\
        -i /home/simon/Data/dfg_knees/monai_segmentations/labels/original \\
        -o out/knee_cohort --workers 16

    # the historical 94-scan tibial table
    python scripts/report_knee_cartilage_cohort.py -i <dir> -o out/94 --include-flagged \\
        --exclude K7_90_belastet K11_90_belastet

Scan names are parsed as ``K<subject>_<0|30|60|90>_<belastet|unbelastet>``; the reproducibility
sections need that pattern and are skipped (with a note) for a cohort that does not follow it.
"""
import argparse
import csv
import datetime
import itertools
import json
import logging
import math
import os
import re
import socket
import subprocess
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# Allow running the script standalone (repo root on the path), matching the other
# scripts in this directory.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from morphometry.image_io import Segmentation
from morphometry.cartilage.femur_regions import (
    CONDYLE_KEYS, FEMUR_REGION_KEYS, ZONE_THICKNESS_KEYS,
)
from morphometry.cartilage.knee import Femur, Tibia, scan_qc_flags
from morphometry.cartilage.tibia_regions import TIBIA_REGION_KEYS

log = logging.getLogger('report_knee_cartilage_cohort')

NIFTI_SUFFIXES = ('.nii', '.nii.gz')

# K<subject>_<flexion angle>_<load>.nii[.gz], the dfg_knees naming (the same pattern
# scripts/analyse_thinnest_point_flexion.py parses).
_NAME_RE = re.compile(r'^(K\d+)_(\d+)_(belastet|unbelastet)$')

# The paper's published region shares (Table I, % of plate tAB; Table II, % of cF tAB).
# Ours are patient-side and the knees' laterality is unknown, so these are a sanity check,
# not a paired comparison (see the module docstring).
PAPER_TIBIA_SHARES = {
    'MT': {'c': 20.0, 'e': 16.0, 'i': 15.6, 'a': 24.2, 'p': 24.1},
    'LT': {'c': 20.0, 'e': 17.7, 'i': 17.5, 'a': 22.6, 'p': 22.2},
}
PAPER_FEMUR_SHARES = {
    'cMF': {'i': 33.5, 'c': 33.2, 'e': 33.3},
    'cLF': {'i': 33.4, 'c': 33.2, 'e': 33.4},
}

# Region letter -> name, for the report's row labels.
TIBIA_REGION_NAMES = {'c': 'central', 'i': 'internal', 'e': 'external',
                      'a': 'anterior', 'p': 'posterior'}
FEMUR_REGION_NAMES = {'i': 'internal', 'c': 'central', 'e': 'external'}

# The femoral thickness maps key the extension zones by their long names.
FEMUR_THICKNESS_KEY = {key: ZONE_THICKNESS_KEYS.get(key, key) for key in FEMUR_REGION_KEYS}

# Geometry metrics that are a 0/1 flag per scan rather than a measurement: only their mean
# (the share of scans) is meaningful, so the summary labels them.
INDICATOR_METRICS = ('tibia_right_plate_larger', 'tibia_plate_split_components')

# Aggregate "regions" added to the per-scan table beside the 20 subregions.
TIBIA_AGGREGATES = ('plate_left', 'plate_right', 'whole')
FEMUR_AGGREGATES = ('cf_left', 'cf_right', 'whole')

# The thickness statistics carried per region in scan_metrics.csv and aggregated. The bare
# names are ThC.cAB and the tab_ ones ThC.tAB (denuded bone counted as 0.0 mm); the two
# families must not be mixed (see thickness.STATISTIC_KEYS). ``cMe`` / ``bMe`` are the two
# measurement directions aMe is the mean of; they are carried because an aggregate's aMe can
# only be formed exactly from them (see _aggregate_row).
THICKNESS_STATISTICS = ('cMe', 'bMe', 'aMe', 'aSD', 'aMax', 'aMav', 'aMin', 'aMiv',
                        'tab_aMe', 'tab_aSD', 'tab_aMin', 'tab_aMiv')

# The statistics an aggregate (a plate, a condyle, a whole bone) has an *exact* union value
# for; everything else in THICKNESS_STATISTICS is an area-weighted mean of the subregion
# values on an aggregate row, which is an approximation (see _aggregate_row).
EXACT_ON_AGGREGATES = ('cMe', 'bMe', 'aMe', 'aMax', 'aMin', 'tab_aMe', 'tab_aMin')

# Areas reported per region. ``ac_area_mm2`` (the articular-surface area) is needed for the
# exact aggregate ``bMe``, which is AC-area-weighted, not cAB-weighted.
AREA_COLUMNS = ('tab_area_mm2', 'cab_area_mm2', 'ac_area_mm2', 'denuded_fraction')

SCAN_COLUMNS = (['scan', 'subject', 'angle', 'load', 'status', 'qc_flags', 'bone', 'region',
                 'share'] + list(AREA_COLUMNS)
                + [f'{s}_mm' for s in THICKNESS_STATISTICS] + ['n_cells'])
GEOMETRY_COLUMNS = ['scan', 'subject', 'angle', 'load', 'metric', 'value']
SUMMARY_COLUMNS = ['table', 'scope', 'region', 'unit', 'n', 'mean', 'sd', 'min', 'max',
                   'p50', 'p90', 'paper']

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_SCAN_CSV = 'scan_metrics.csv'
DEFAULT_GEOMETRY_CSV = 'scan_geometry.csv'
DEFAULT_SUMMARY_CSV = 'cohort_summary.csv'
DEFAULT_REPORT = 'report.md'
DEFAULT_META = 'report.meta.json'


# ----------------------------------------------------------------------------------------
# Input handling
# ----------------------------------------------------------------------------------------

def _strip_nifti_suffix(name: str) -> str:
    """
    Strip a ``.nii`` / ``.nii.gz`` suffix from a filename.

    :param name: A filename (not a full path).
    :return: The name without its NIfTI suffix.
    """
    for suffix in ('.nii.gz', '.nii'):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def resolve_inputs(inputs: Sequence[str]) -> List[str]:
    """
    Expand the given input paths into a sorted, de-duplicated list of NIfTI files.

    :param inputs: The raw ``--input`` values (files and/or directories, searched
        recursively).
    :return: Sorted, de-duplicated list of absolute NIfTI file paths.
    :raises SystemExit: If a path does not exist or no NIfTI files are found.
    """
    found = set()
    for entry in inputs:
        if os.path.isfile(entry):
            found.add(os.path.abspath(entry))
        elif os.path.isdir(entry):
            for suffix in NIFTI_SUFFIXES:
                for match in Path(entry).rglob(f'*{suffix}'):
                    found.add(os.path.abspath(str(match)))
        else:
            log.error('input path not found: %s', entry)
            sys.exit(1)
    if not found:
        log.error('no NIfTI (.nii/.nii.gz) files found in the given input(s).')
        sys.exit(1)
    return sorted(found)


def parse_scan_name(name: str) -> Tuple[Optional[str], Optional[int], Optional[str]]:
    """
    Parse a ``K<subject>_<flexion>_<load>`` scan name.

    :param name: The scan id (filename without its NIfTI suffix).
    :return: ``(subject, angle, load)``, e.g. ``('K1', 60, 'belastet')``; all three are
        ``None`` for a name that does not match, in which case the reproducibility sections
        skip that scan.
    """
    match = _NAME_RE.match(name)
    if match is None:
        return None, None, None
    subject, angle, load = match.groups()
    return subject, int(angle), load


def _natural_subject_key(subject: Optional[str]) -> tuple:
    """Sort key so K2 precedes K10 (numeric part of a ``K<n>`` subject id)."""
    if subject is None:
        return (1, 0, '')
    return (0, int(subject[1:]), subject)


# ----------------------------------------------------------------------------------------
# Per-scan measurement
# ----------------------------------------------------------------------------------------

def load_scan(path: str) -> Segmentation:
    """
    Load one knee segmentation and reorient it to LPI.

    Matches the other knee scripts: no ``remove_outliers()``, the bones clean their own
    cartilage masks. Kept separate from :func:`measure_scan` so a scan whose measurement
    fails can still report its orientation QC flag.

    :param path: Path to the segmentation NIfTI.
    :return: The reoriented ``Segmentation`` (check ``orientation_ok`` for whether it is LPI).
    """
    seg = Segmentation('nibabel')
    seg.read_image(path)
    seg.transform_coordinate_system()
    return seg


def _weighted_mean(values: Sequence[float], weights: Sequence[float]) -> float:
    """
    Area-weighted mean that tolerates NaN entries and a zero total weight.

    :param values: The values to average.
    :param weights: Their weights (areas in mm^2).
    :return: The weighted mean, or ``NaN`` if nothing finite is left.
    """
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    ok = np.isfinite(v) & np.isfinite(w) & (w > 0)
    if not ok.any():
        return float('nan')
    return float(np.average(v[ok], weights=w[ok]))


def _region_row(bone: str, region: str, share: float, statistics: dict,
                n_cells: int) -> dict:
    """
    One (bone, region) row of the per-scan table.

    :param bone: ``'tibia'`` or ``'femur'``.
    :param region: The region key or aggregate name.
    :param share: The region's share of its plate / condyle tAB (``NaN`` for aggregates).
    :param statistics: That region's entry of ``ThicknessMaps.statistics`` (``{}`` if the
        bone's thickness is unavailable).
    :param n_cells: Number of in-plane map cells the region covers.
    :return: A dict with the measurement columns of :data:`SCAN_COLUMNS`.
    """
    row = {'bone': bone, 'region': region, 'share': share, 'n_cells': n_cells}
    for area in AREA_COLUMNS:
        row[area] = statistics.get(area, float('nan'))
    for statistic in THICKNESS_STATISTICS:
        row[f'{statistic}_mm'] = statistics.get(statistic, float('nan'))
    return row


def _aggregate_row(bone: str, region: str, rows: List[dict]) -> dict:
    """
    An aggregate row (a plate, a condyle, a whole bone) built from its subregion rows.

    **Which of these are exact, and which are not.** Areas add up exactly, and so do the
    statistics in :data:`EXACT_ON_AGGREGATES`:

    * ``cMe`` is cAB-area-weighted and ``bMe`` AC-area-weighted by definition
      (:func:`morphometry.cartilage.thickness._region_statistics`), so the union's are the
      respective weighted means of the parts', ``aMe = (cMe + bMe) / 2`` follows, and
      ``tab_aMe = aMe * cAB / tAB`` holds for the union exactly as it does per region. Taking
      the union ``aMe`` as a cAB-weighted mean of the subregion ``aMe``s instead -- which this
      function used to do -- is wrong by up to 0.025 mm (1.3%) on a whole bone, because the
      ``bMe`` half carries the wrong weights.
    * ``aMin`` / ``aMax`` (and ``tab_aMin``) are the min / max over the parts, which is the
      min / max over their union. Note the consequence for ``tab_aMin``: a single denuded
      triangle anywhere in the union makes it exactly 0.0, which is the denuded-as-0.0 mm
      convention and not a thin spot.

    Everything else -- ``aSD``, ``aMav``, ``aMiv``, ``tab_aSD``, ``tab_aMiv`` -- has **no**
    exact union value computable from the parts, and is reported as an area-weighted mean of
    the subregion values. Read those aggregate cells with care:

    * an aggregate ``aSD`` is a mean of *within*-subregion spreads and drops all
      between-subregion variance, so it under-states the union's heterogeneity (on
      ``K1_0_unbelastet`` the subregion ``aMe``s span 1.37-2.97 mm, none of which it sees);
    * an aggregate ``aMav`` / ``aMiv`` / ``tab_aMiv`` is a mean of the subregions' own 1%
      tails, not the union's 1% tail.

    :param bone: ``'tibia'`` or ``'femur'``.
    :param region: The aggregate's name.
    :param rows: The subregion rows it covers.
    :return: A dict with the measurement columns of :data:`SCAN_COLUMNS`.
    """
    cab_weights = [r['cab_area_mm2'] for r in rows]
    ac_weights = [r['ac_area_mm2'] for r in rows]
    tab = float(np.nansum([r['tab_area_mm2'] for r in rows]))
    cab = float(np.nansum(cab_weights))
    covered_fraction = cab / tab if tab > 0 else float('nan')
    row = {'bone': bone, 'region': region, 'share': float('nan'),
           'tab_area_mm2': tab, 'cab_area_mm2': cab,
           'ac_area_mm2': float(np.nansum(ac_weights)),
           'denuded_fraction': (1.0 - covered_fraction) if tab > 0 else float('nan'),
           'n_cells': int(np.nansum([r['n_cells'] for r in rows]))}
    for statistic in THICKNESS_STATISTICS:
        key = f'{statistic}_mm'
        values = [r[key] for r in rows]
        finite = np.isfinite(np.asarray(values, dtype=float)).any()
        if statistic.endswith('aMax'):
            row[key] = float(np.nanmax(values)) if finite else float('nan')
        elif statistic.endswith('aMin'):
            row[key] = float(np.nanmin(values)) if finite else float('nan')
        elif statistic == 'bMe':
            row[key] = _weighted_mean(values, ac_weights)        # AC-area-weighted
        else:
            row[key] = _weighted_mean(values, cab_weights)
    # The exact two-directional means, from the two directions rather than from aMe.
    row['aMe_mm'] = 0.5 * (row['cMe_mm'] + row['bMe_mm'])
    row['tab_aMe_mm'] = row['aMe_mm'] * covered_fraction
    return row


def _tibia_metrics(tibia: Tibia, thickness, rows: List[dict], geometry: dict):
    """
    Fill the tibial region rows and geometry scalars of one scan.

    :param tibia: The measured ``Tibia``.
    :param thickness: Its ``ThicknessMaps``.
    :param rows: The scan's region row list, appended to in place.
    :param geometry: The scan's geometry dict, filled in place.
    """
    subdivision = tibia.subdivision
    shares, areas = subdivision.region_shares, subdivision.region_areas
    per_region = []
    for key in TIBIA_REGION_KEYS:
        statistics = dict(thickness.statistics.get(key, {}))
        statistics.setdefault('tab_area_mm2', areas.get(key, float('nan')))
        per_region.append(_region_row('tibia', key, shares.get(key, float('nan')), statistics,
                                      len(thickness.get(key, {}))))
    rows.extend(per_region)
    for side in ('left', 'right'):
        plate_rows = [r for r in per_region if r['region'].endswith(f'{side[0]}t')]
        rows.append(_aggregate_row('tibia', f'plate_{side}', plate_rows))
    rows.append(_aggregate_row('tibia', 'whole', per_region))

    left, right = subdivision.plates['left'], subdivision.plates['right']
    areas_by_side = {'left': left.tab_area, 'right': right.tab_area}
    geometry.update({
        'tibia_dropped_area_mm2': subdivision.dropped_area,
        'tibia_plate_split_components': float(subdivision.plate_split == 'components'),
        'tibia_plate_area_ratio': (min(areas_by_side.values()) / max(areas_by_side.values())
                                   if max(areas_by_side.values()) > 0 else float('nan')),
        # The ratio find_plates itself checks against min_plate_ratio (0.2) is on the
        # *covered* component areas, before the per-plate hole filling, so it is reported
        # separately from the tAB ratio above.
        'tibia_plate_cab_ratio': (min(left.cab_area, right.cab_area)
                                  / max(left.cab_area, right.cab_area)
                                  if max(left.cab_area, right.cab_area) > 0 else float('nan')),
        'tibia_right_plate_larger': float(right.tab_area > left.tab_area),
    })
    for side, plate in (('left', left), ('right', right)):
        geometry[f'tibia_filled_area_{side}_mm2'] = plate.tab_area - plate.cab_area
        geometry[f'tibia_plate_variance1_{side}_mm2'] = float(plate.variances[0])
        geometry[f'tibia_plate_variance2_{side}_mm2'] = float(plate.variances[1])
        geometry[f'tibia_ellipse_level_{side}'] = float(plate.ellipse_level)
        # How well the plate's in-plane anterior direction lines up with world +y: the
        # subdivision raises above 60 deg off, and the plan quotes the cohort minimum.
        geometry[f'tibia_anterior_cos_{side}'] = abs(float(plate.anterior_direction[1]))
    if tibia.transfer_qc is not None:
        geometry['tibia_transfer_median_mm'] = tibia.transfer_qc['median_mm']
        geometry['tibia_transfer_max_mm'] = tibia.transfer_qc['max_mm']
        geometry['tibia_transfer_far_fraction'] = tibia.transfer_qc['far_fraction']


def _femur_metrics(femur: Femur, thickness, rows: List[dict], geometry: dict):
    """
    Fill the femoral region rows and geometry scalars of one scan.

    :param femur: The measured ``Femur``.
    :param thickness: Its ``ThicknessMaps``.
    :param rows: The scan's region row list, appended to in place.
    :param geometry: The scan's geometry dict, filled in place.
    """
    subdivision = femur.subdivision
    shares, areas = subdivision.region_shares, subdivision.region_areas
    per_region = []
    for key in FEMUR_REGION_KEYS:
        map_key = FEMUR_THICKNESS_KEY[key]
        statistics = dict(thickness.statistics.get(map_key, {}))
        statistics.setdefault('tab_area_mm2', areas.get(key, float('nan')))
        per_region.append(_region_row('femur', key, shares.get(key, float('nan')), statistics,
                                      len(thickness.get(map_key, {}))))
    rows.extend(per_region)
    for side in ('left', 'right'):
        thirds = [r for r in per_region if r['region'].endswith(f'c{side[0]}f')]
        rows.append(_aggregate_row('femur', f'cf_{side}', thirds))
    rows.append(_aggregate_row('femur', 'whole', per_region))

    frame = subdivision.frame
    geometry.update({
        'femur_tab_area_mm2': subdivision.tab_area,
        'femur_notch_ap_mm': frame.notch_ap,
        'femur_shaft_residual_mm': frame.shaft_residual_mm,
        'femur_shaft_bins': float(frame.shaft_bins),
        'femur_divider_slope': float(frame.divider[0]),
    })
    for side in ('left', 'right'):
        condyle = subdivision.condyles[side]
        geometry[f'femur_ap_extent_{side}_mm'] = frame.notch_ap - condyle.posterior_ap
        geometry[f'femur_cf_slices_{side}'] = float(condyle.n_slices)
    if femur.transfer_qc is not None:
        geometry['femur_transfer_median_mm'] = femur.transfer_qc['median_mm']
        geometry['femur_transfer_max_mm'] = femur.transfer_qc['max_mm']
        geometry['femur_transfer_far_fraction'] = femur.transfer_qc['far_fraction']


def measure_scan(path: str, femur_label: int = 3, tibia_label: int = 4,
                 tibia_bone_label: int = 2, femur_bone_label: int = 1) -> dict:
    """
    Measure one knee scan: both bones' subregions, areas and thickness.

    Each bone is wrapped separately, as in ``compute_knee_cartilage_thickness.py``: a tibia
    that cannot be split into two plates does not cost the femoral numbers, and a scan that
    cannot be loaded at all still produces a record (with its error) instead of aborting the
    run.

    :param path: Path to the segmentation NIfTI.
    :param femur_label: Label of the femoral cartilage.
    :param tibia_label: Label of the tibial cartilage.
    :param tibia_bone_label: Label of the tibial bone (the tibial subregions are defined on
        its subchondral surface).
    :param femur_bone_label: Label of the femoral bone (ditto, plus the femoral frame).
    :return: ``{'scan', 'subject', 'angle', 'load', 'status', 'error', 'qc_flags', 'regions',
        'geometry'}``. ``status`` is ``'ok'``, ``'tibia_failed'``, ``'femur_failed'`` or
        ``'failed'`` (neither bone, e.g. an unreadable file); ``regions`` holds the rows of
        the bones that succeeded and ``geometry`` the per-scan scalars.
    """
    scan = _strip_nifti_suffix(os.path.basename(path))
    subject, angle, load = parse_scan_name(scan)
    record = {'scan': scan, 'subject': subject, 'angle': angle, 'load': load,
              'status': 'ok', 'error': '', 'qc_flags': '', 'regions': [], 'geometry': {}}

    try:
        seg = load_scan(path)
    except Exception as exc:  # unreadable file: keep the scan visible
        record.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        return record

    tibia = femur = None
    failures = []
    started = time.time()
    try:
        tibia = Tibia(seg, cartilage_label=tibia_label, bone_label=tibia_bone_label)
        tibia_thickness = tibia.calculate_thickness()
        _tibia_metrics(tibia, tibia_thickness, record['regions'], record['geometry'])
    except Exception as exc:
        failures.append(f'tibia: {type(exc).__name__}: {exc}')
        tibia = None
    record['geometry']['tibia_seconds'] = time.time() - started

    started = time.time()
    try:
        femur = Femur(seg, cartilage_label=femur_label, bone_label=femur_bone_label)
        femur_thickness = femur.calculate_thickness(tibia)
        _femur_metrics(femur, femur_thickness, record['regions'], record['geometry'])
    except Exception as exc:
        failures.append(f'femur: {type(exc).__name__}: {exc}')
        femur = None
    record['geometry']['femur_seconds'] = time.time() - started

    # Flagged, not fixed or dropped: the measurements stand as computed.
    record['qc_flags'] = scan_qc_flags(seg, tibia, femur)
    record['error'] = ' | '.join(failures)
    if tibia is None and femur is None:
        record['status'] = 'failed'
    elif tibia is None:
        record['status'] = 'tibia_failed'
    elif femur is None:
        record['status'] = 'femur_failed'
    return record


def _worker(arguments: tuple) -> dict:
    """
    Process-pool entry point: :func:`measure_scan` with warnings silenced.

    :param arguments: ``(path, femur_label, tibia_label, tibia_bone_label, femur_bone_label)``.
    :return: The scan's record (see :func:`measure_scan`).
    """
    warnings.simplefilter('ignore')
    path, femur_label, tibia_label, tibia_bone_label, femur_bone_label = arguments
    return measure_scan(path, femur_label=femur_label, tibia_label=tibia_label,
                        tibia_bone_label=tibia_bone_label, femur_bone_label=femur_bone_label)


def measure_cohort(paths: Sequence[str], femur_label: int, tibia_label: int,
                   tibia_bone_label: int, femur_bone_label: int,
                   workers: Optional[int] = None) -> List[dict]:
    """
    Measure every scan of the cohort, in parallel, keeping going past failures.

    :param paths: The scan paths.
    :param femur_label: Label of the femoral cartilage.
    :param tibia_label: Label of the tibial cartilage.
    :param tibia_bone_label: Label of the tibial bone.
    :param femur_bone_label: Label of the femoral bone.
    :param workers: Worker processes; ``None`` uses all visible CPUs, ``1`` runs in-process
        (which makes the per-scan timings in ``scan_geometry.csv`` meaningful).
    :return: The records, sorted by subject, flexion angle and load.
    """
    jobs = [(p, femur_label, tibia_label, tibia_bone_label, femur_bone_label) for p in paths]
    records: List[dict] = []
    started = time.time()
    if workers == 1:
        for n, job in enumerate(jobs, start=1):
            records.append(_worker(job))
            log.info('[%d/%d] %s %s', n, len(jobs), records[-1]['scan'], records[-1]['status'])
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_worker, job): job[0] for job in jobs}
            for n, future in enumerate(as_completed(futures), start=1):
                record = future.result()
                records.append(record)
                log.info('[%d/%d] %s %s (%.0fs elapsed)', n, len(jobs), record['scan'],
                         record['status'], time.time() - started)
    records.sort(key=lambda r: (_natural_subject_key(r['subject']),
                                r['angle'] if r['angle'] is not None else -1,
                                r['load'] or '', r['scan']))
    log.info('measured %d scans in %.0fs', len(records), time.time() - started)
    return records


# ----------------------------------------------------------------------------------------
# Aggregation
# ----------------------------------------------------------------------------------------

def select_scans(records: Sequence[dict], include_flagged: bool = False,
                 exclude: Sequence[str] = ()) -> List[dict]:
    """
    The records the cohort statistics are computed over.

    :param records: All measured records.
    :param include_flagged: Keep scans carrying QC flags (default: drop them).
    :param exclude: Scan names to drop regardless.
    :return: The kept records (a scan whose status is ``'failed'`` is always dropped; a scan
        with one failed bone is kept, and contributes only the bone that worked).
    """
    exclude = set(exclude)
    kept = []
    for record in records:
        if record['scan'] in exclude or record['status'] == 'failed':
            continue
        if record['qc_flags'] and not include_flagged:
            continue
        kept.append(record)
    return kept


def _summary_row(table: str, scope: str, region: str, unit: str, values: Sequence[float],
                 paper=None) -> dict:
    """
    Summarise a sample as one row of ``cohort_summary.csv``.

    :param table: Which table this row belongs to.
    :param scope: The grouping inside the table (plate, condyle, bone, quantity, ...).
    :param region: The region (or metric / angle) the row describes.
    :param unit: Unit of the values.
    :param values: The sample (NaNs are dropped).
    :param paper: The paper's published value, or ``None``.
    :return: A dict with the columns of :data:`SUMMARY_COLUMNS`.
    """
    v = np.asarray([x for x in values if x is not None], dtype=float)
    v = v[np.isfinite(v)]
    row = {'table': table, 'scope': scope, 'region': region, 'unit': unit, 'n': int(len(v)),
           'paper': '' if paper is None else paper}
    if len(v) == 0:
        row.update(mean=float('nan'), sd=float('nan'), min=float('nan'), max=float('nan'),
                   p50=float('nan'), p90=float('nan'))
        return row
    row.update(mean=float(v.mean()), sd=float(v.std(ddof=1)) if len(v) > 1 else 0.0,
               min=float(v.min()), max=float(v.max()),
               p50=float(np.percentile(v, 50)), p90=float(np.percentile(v, 90)))
    return row


def _region_values(records: Sequence[dict], bone: str, region: str, column: str) -> List[float]:
    """
    One column of one (bone, region) over the cohort.

    :param records: The selected records.
    :param bone: ``'tibia'`` or ``'femur'``.
    :param region: The region key or aggregate name.
    :param column: The row key to read (e.g. ``'share'``, ``'aMe_mm'``).
    :return: One value per record that has that row.
    """
    values = []
    for record in records:
        for row in record['regions']:
            if row['bone'] == bone and row['region'] == region:
                values.append(row[column])
    return values


def _by_scan(records: Sequence[dict], bone: str, region: str, column: str) -> Dict[str, float]:
    """
    One column of one (bone, region), keyed by scan name.

    :param records: The selected records.
    :param bone: ``'tibia'`` or ``'femur'``.
    :param region: The region key or aggregate name.
    :param column: The row key to read.
    :return: ``{scan: value}`` for the scans that have that row.
    """
    out = {}
    for record in records:
        for row in record['regions']:
            if row['bone'] == bone and row['region'] == region:
                out[record['scan']] = row[column]
    return out


def region_tables(records: Sequence[dict]) -> List[dict]:
    """
    The Table I / Table II analogues, the area tables and the thickness tables.

    :param records: The selected records.
    :return: Rows for ``cohort_summary.csv``.
    """
    rows: List[dict] = []
    # Table I: tibial region shares per plate, in percent of that plate's tAB. No paper value
    # is attached to the row: our plates are patient-side, the paper's are medial / lateral,
    # and the mapping is unknown (see paper_comparison_rows).
    for side in ('left', 'right'):
        for letter in ('c', 'i', 'e', 'a', 'p'):
            key = f'{letter}{side[0]}t'
            shares = [100 * v for v in _region_values(records, 'tibia', key, 'share')]
            rows.append(_summary_row('tibia_region_shares', f'plate_{side}',
                                     TIBIA_REGION_NAMES[letter], '%', shares))
    # Table II: femoral thirds per condyle.
    for side in ('left', 'right'):
        for letter in ('i', 'c', 'e'):
            key = f'{letter}c{side[0]}f'
            shares = [100 * v for v in _region_values(records, 'femur', key, 'share')]
            rows.append(_summary_row('femur_region_shares', f'condyle_{side}',
                                     FEMUR_REGION_NAMES[letter], '%', shares))
    # Areas and denuded fraction, per subregion and per aggregate.
    for bone, keys, aggregates in (('tibia', TIBIA_REGION_KEYS, TIBIA_AGGREGATES),
                                   ('femur', FEMUR_REGION_KEYS, FEMUR_AGGREGATES)):
        for region in list(keys) + list(aggregates):
            rows.append(_summary_row('tab_area', bone, region, 'mm2',
                                     _region_values(records, bone, region, 'tab_area_mm2')))
            rows.append(_summary_row('cab_area', bone, region, 'mm2',
                                     _region_values(records, bone, region, 'cab_area_mm2')))
            rows.append(_summary_row('denuded_fraction', bone, region, 'fraction',
                                     _region_values(records, bone, region,
                                                    'denuded_fraction')))
            for statistic in THICKNESS_STATISTICS:
                rows.append(_summary_row(f'thickness_{statistic}', bone, region, 'mm',
                                         _region_values(records, bone, region,
                                                        f'{statistic}_mm')))
    # Whole-bone aMe per flexion angle (the plan quotes a range over the four angles).
    angles = sorted({r['angle'] for r in records if r['angle'] is not None})
    for bone in ('tibia', 'femur'):
        for angle in angles:
            at_angle = [r for r in records if r['angle'] == angle]
            rows.append(_summary_row('whole_bone_aMe_by_angle', bone, f'{angle}deg', 'mm',
                                     _region_values(at_angle, bone, 'whole', 'aMe_mm')))
    return rows


def paper_comparison_rows(records: Sequence[dict]) -> List[dict]:
    """
    The paper's published shares, as rows of the summary table.

    Written out as their own rows rather than into the ``paper`` column of ours, because the
    mapping between our patient-side plates and the paper's medial / lateral ones is unknown
    (see the module docstring), so no pairing can be asserted.

    :param records: The selected records (unused; kept so the call site reads uniformly).
    :return: Rows for ``cohort_summary.csv``.
    """
    rows = []
    for plate, shares in PAPER_TIBIA_SHARES.items():
        for letter, value in shares.items():
            rows.append({'table': 'paper_tibia_region_shares', 'scope': plate,
                         'region': TIBIA_REGION_NAMES[letter], 'unit': '%', 'n': 1,
                         'mean': value, 'sd': float('nan'), 'min': value, 'max': value,
                         'p50': value, 'p90': value, 'paper': value})
    for condyle, shares in PAPER_FEMUR_SHARES.items():
        for letter, value in shares.items():
            rows.append({'table': 'paper_femur_region_shares', 'scope': condyle,
                         'region': FEMUR_REGION_NAMES[letter], 'unit': '%', 'n': 1,
                         'mean': value, 'sd': float('nan'), 'min': value, 'max': value,
                         'p50': value, 'p90': value, 'paper': value})
    return rows


def geometry_tables(records: Sequence[dict]) -> List[dict]:
    """
    Summaries of the per-scan geometry and QC scalars.

    The ``*_seconds`` timings are **left out**: they are the only per-scan quantity that
    depends on how the run was scheduled, and keeping them here would make
    ``cohort_summary.csv`` differ between two runs that computed identical numbers. They stay
    per scan in ``scan_geometry.csv`` and are summarised in the run's ``.meta.json``.

    A metric whose name is listed in :data:`INDICATOR_METRICS` is a 0/1 flag per scan, so only
    its ``mean`` (the share of scans) and ``n`` mean anything; its unit says so.

    :param records: The selected records.
    :return: Rows for ``cohort_summary.csv``.
    """
    metrics = sorted({m for r in records for m in r['geometry']
                      if not m.endswith('seconds')})
    rows = []
    for metric in metrics:
        unit = 'mm2' if metric.endswith('mm2') else ('mm' if metric.endswith('mm') else '')
        if metric in INDICATOR_METRICS:
            unit = 'share of scans (0/1 per scan)'
        rows.append(_summary_row('geometry', '', metric, unit,
                                 [r['geometry'].get(metric, float('nan')) for r in records]))
    return rows


def timing_summary(records: Sequence[dict]) -> dict:
    """
    Per-bone wall-clock timings of the run, for the ``.meta.json``.

    :param records: All measured records.
    :return: ``{metric: {'n', 'median', 'min', 'max'}}`` in seconds (empty without timings).
    """
    out = {}
    for metric in ('tibia_seconds', 'femur_seconds'):
        values = np.asarray([r['geometry'][metric] for r in records
                             if metric in r['geometry']], dtype=float)
        values = values[np.isfinite(values)]
        if len(values):
            out[metric] = {'n': int(len(values)), 'median': float(np.median(values)),
                           'min': float(values.min()), 'max': float(values.max())}
    return out


# -------------------------------------------------------------------------- reproducibility

def _quantity_maps(records: Sequence[dict]) -> List[tuple]:
    """
    The quantities the reproducibility sections are computed on.

    :param records: The selected records.
    :return: A list of ``(quantity, unit, relative, entries)`` where ``entries`` is a list of
        ``(bone, region, label, {scan: value})``; ``relative`` says whether the spread is
        expressed as a percentage of the mean rather than in ``unit``. The aggregate regions
        (``plate_*`` / ``cf_*`` / ``whole``) are included as their own rows but kept out of
        the pooled samples, which cover the 20 subregions only.
    """
    quantities = []

    shares = []
    for bone, keys in (('tibia', TIBIA_REGION_KEYS), ('femur', CONDYLE_KEYS)):
        for key in keys:
            shares.append((bone, key, key,
                           {s: 100 * v for s, v in
                            _by_scan(records, bone, key, 'share').items()}))
    quantities.append(('region_share', 'pp', False, shares))

    areas = []
    for bone, keys, aggregates in (('tibia', TIBIA_REGION_KEYS, TIBIA_AGGREGATES),
                                   ('femur', FEMUR_REGION_KEYS, FEMUR_AGGREGATES)):
        for region in list(keys) + list(aggregates):
            areas.append((bone, region, region,
                          _by_scan(records, bone, region, 'tab_area_mm2')))
    quantities.append(('tab_area', '%', True, areas))

    thickness = []
    for bone, keys, aggregates in (('tibia', TIBIA_REGION_KEYS, TIBIA_AGGREGATES),
                                   ('femur', FEMUR_REGION_KEYS, FEMUR_AGGREGATES)):
        for region in list(keys) + list(aggregates):
            thickness.append((bone, region, region, _by_scan(records, bone, region, 'aMe_mm')))
    quantities.append(('aMe', 'mm', False, thickness))
    return quantities


def load_pairs(records: Sequence[dict]) -> List[Tuple[str, str]]:
    """
    The (loaded, unloaded) scan pairs of the cohort: the same subject at the same angle.

    :param records: The selected records.
    :return: ``[(belastet_scan, unbelastet_scan), ...]``, sorted.
    """
    by_key: Dict[tuple, Dict[str, str]] = {}
    for record in records:
        if record['subject'] is None:
            continue
        by_key.setdefault((record['subject'], record['angle']), {})[record['load']] = \
            record['scan']
    pairs = []
    for key in sorted(by_key, key=lambda k: (_natural_subject_key(k[0]), k[1])):
        loads = by_key[key]
        if 'belastet' in loads and 'unbelastet' in loads:
            pairs.append((loads['belastet'], loads['unbelastet']))
    return pairs


def flexion_series(records: Sequence[dict], min_angles: int = 3) -> List[List[str]]:
    """
    The flexion series of the cohort: the scans of one subject and load, across angles.

    :param records: The selected records.
    :param min_angles: Smallest number of angles a series must still have (a series loses a
        scan when it is QC-flagged or failed).
    :return: A list of scan-name lists, ordered by angle inside each series.
    """
    by_key: Dict[tuple, List[Tuple[int, str]]] = {}
    for record in records:
        if record['subject'] is None:
            continue
        by_key.setdefault((record['subject'], record['load']), []).append(
            (record['angle'], record['scan']))
    series = []
    for key in sorted(by_key, key=lambda k: (_natural_subject_key(k[0]), k[1])):
        scans = sorted(by_key[key])
        if len(scans) >= min_angles:
            series.append([scan for _angle, scan in scans])
    return series


def reproducibility_tables(records: Sequence[dict], min_angles: int = 3) -> Tuple[List[dict], dict]:
    """
    Loaded-vs-unloaded differences and across-flexion spreads, per region and pooled.

    **Spread, not precision.** A loaded and an unloaded scan of one knee, and the four flexion
    angles of one knee, are different acquisitions in different positions; the numbers below
    bound how much the pipeline's output moves when the knee is repositioned and reloaded, and
    they are not a measurement error. The paper's 19-84 um are repeat scans of an unchanged
    knee.

    :param records: The selected records.
    :param min_angles: Smallest number of angles a flexion series must have.
    :return: ``(rows, counts)`` -- summary rows and ``{'load_pairs': n, 'flexion_series': n}``.
    """
    pairs = load_pairs(records)
    series = flexion_series(records, min_angles=min_angles)
    aggregates = set(TIBIA_AGGREGATES) | set(FEMUR_AGGREGATES)
    rows: List[dict] = []
    for quantity, unit, relative, entries in _quantity_maps(records):
        # Pooled over the subregions only, overall and per bone: the aggregate regions are
        # sums of the subregions and would double-count them (and the documented cohort
        # figures -- e.g. the 900 loaded-vs-unloaded aMe pairs = 45 pairs x 20 subregions --
        # are pooled this way).
        pooled_load = {'pooled': [], 'pooled_tibia': [], 'pooled_femur': []}
        pooled_flexion = {'pooled': [], 'pooled_tibia': [], 'pooled_femur': []}
        pooled_cv = {'pooled': [], 'pooled_tibia': [], 'pooled_femur': []}
        for bone, region, label, values in entries:
            deltas = []
            for loaded, unloaded in pairs:
                a, b = values.get(loaded, float('nan')), values.get(unloaded, float('nan'))
                if not (math.isfinite(a) and math.isfinite(b)):
                    continue
                delta = abs(a - b)
                if relative:
                    # A percentage of the pair's mean; undefined (not "0 mm") if both vanish,
                    # so that a '%' column never silently carries an absolute value.
                    deltas.append(200 * delta / (a + b) if (a + b) > 0 else float('nan'))
                else:
                    deltas.append(delta)
            rows.append(_summary_row(f'load_spread_{quantity}', bone, label,
                                     unit if not relative else '%', deltas))
            if region not in aggregates:
                pooled_load['pooled'].extend(deltas)
                pooled_load[f'pooled_{bone}'].extend(deltas)

            spreads, cvs = [], []
            for scans in series:
                sample = [values[s] for s in scans
                          if math.isfinite(values.get(s, float('nan')))]
                if len(sample) < min_angles:
                    continue
                sd = float(np.std(sample, ddof=1))
                mean = float(np.mean(sample))
                spreads.append((100 * sd / mean if mean > 0 else float('nan'))
                               if relative else sd)
                if mean > 0:
                    cvs.append(100 * sd / mean)
            rows.append(_summary_row(f'flexion_spread_{quantity}', bone, label,
                                     unit if not relative else '%', spreads))
            # For a quantity whose spread is already relative, the CV *is* the spread: emitting
            # it again would duplicate the row. `flexion_cv(...)` falls back to the spread row.
            if not relative:
                rows.append(_summary_row(f'flexion_cv_{quantity}', bone, label, '%', cvs))
            if region not in aggregates:
                pooled_flexion['pooled'].extend(spreads)
                pooled_flexion[f'pooled_{bone}'].extend(spreads)
                pooled_cv['pooled'].extend(cvs)
                pooled_cv[f'pooled_{bone}'].extend(cvs)
        for scope in ('pooled', 'pooled_tibia', 'pooled_femur'):
            rows.append(_summary_row(f'load_spread_{quantity}', scope, '__all__',
                                     unit if not relative else '%', pooled_load[scope]))
            rows.append(_summary_row(f'flexion_spread_{quantity}', scope, '__all__',
                                     unit if not relative else '%', pooled_flexion[scope]))
            if not relative:
                rows.append(_summary_row(f'flexion_cv_{quantity}', scope, '__all__', '%',
                                         pooled_cv[scope]))
    return rows, {'load_pairs': len(pairs), 'flexion_series': len(series)}


# ----------------------------------------------------------------------------------------
# Output
# ----------------------------------------------------------------------------------------

def _format(value) -> str:
    """
    Format a CSV cell: plain strings, ``''`` for None, repr for floats (NaN included).

    :param value: The cell value.
    :return: Its string form.
    """
    if value is None:
        return ''
    if isinstance(value, float):
        return 'nan' if math.isnan(value) else repr(value)
    return str(value)


def write_scan_csv(records: Sequence[dict], path: Path):
    """
    Write the per-scan, per-region long table.

    :param records: All measured records (failures included, as a header-only row).
    :param path: Destination CSV path.
    """
    with open(path, 'w', newline='') as fh:
        writer = csv.writer(fh)
        writer.writerow(SCAN_COLUMNS)
        for record in records:
            head = [record['scan'], record['subject'], record['angle'], record['load'],
                    record['status'], record['qc_flags']]
            if not record['regions']:
                writer.writerow([_format(v) for v in head]
                                + [''] * (len(SCAN_COLUMNS) - len(head)))
                continue
            for row in record['regions']:
                writer.writerow([_format(v) for v in head]
                                + [_format(row[c]) for c in SCAN_COLUMNS[len(head):]])
    log.info('wrote %s', path)


def write_geometry_csv(records: Sequence[dict], path: Path):
    """
    Write the per-scan geometry / QC scalars as a long table.

    :param records: All measured records.
    :param path: Destination CSV path.
    """
    with open(path, 'w', newline='') as fh:
        writer = csv.writer(fh)
        writer.writerow(GEOMETRY_COLUMNS)
        for record in records:
            for metric in sorted(record['geometry']):
                writer.writerow([record['scan'], _format(record['subject']),
                                 _format(record['angle']), _format(record['load']),
                                 metric, _format(record['geometry'][metric])])
    log.info('wrote %s', path)


def write_summary_csv(rows: Sequence[dict], path: Path):
    """
    Write the cohort summary table (one row per statistic).

    :param rows: The summary rows.
    :param path: Destination CSV path.
    """
    with open(path, 'w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=SUMMARY_COLUMNS, extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _format(row.get(k)) for k in SUMMARY_COLUMNS})
    log.info('wrote %s', path)


def _lookup(rows: Sequence[dict], table: str, scope: str, region: str) -> Optional[dict]:
    """
    Find one summary row.

    :param rows: The summary rows.
    :param table: The table name.
    :param scope: The scope.
    :param region: The region.
    :return: The row, or ``None``.
    """
    for row in rows:
        if row['table'] == table and row['scope'] == scope and row['region'] == region:
            return row
    return None


def _cell(row: Optional[dict], key: str, digits: int = 2) -> str:
    """
    Format one statistic of a summary row for the markdown report.

    :param row: The summary row, or ``None``.
    :param key: Which statistic to show.
    :param digits: Decimal digits.
    :return: The formatted value, or ``-``.
    """
    if row is None or row.get(key) is None:
        return '-'
    value = row[key]
    if isinstance(value, str):
        return value or '-'
    if isinstance(value, float) and math.isnan(value):
        return '-'
    return f'{value:.{digits}f}'


def _mean_sd(row: Optional[dict], digits: int = 2) -> str:
    """
    ``mean (SD)`` of a summary row.

    :param row: The summary row.
    :param digits: Decimal digits.
    :return: e.g. ``20.00 (0.00)``, or ``-``.
    """
    if row is None:
        return '-'
    return f'{_cell(row, "mean", digits)} ({_cell(row, "sd", digits)})'


def _range(row: Optional[dict], digits: int = 1) -> str:
    """
    ``min-max`` of a summary row.

    :param row: The summary row.
    :param digits: Decimal digits.
    :return: e.g. ``4266-6445``, or ``-``.
    """
    if row is None:
        return '-'
    return f'{_cell(row, "min", digits)}-{_cell(row, "max", digits)}'


def _flexion_cv(rows: Sequence[dict], quantity: str, scope: str, region: str) -> Optional[dict]:
    """
    The flexion CV row of a quantity, falling back to its spread row.

    A quantity whose spread is already expressed in per cent of the mean (an area) has no
    separate CV row -- the two would be the same numbers (see
    :func:`reproducibility_tables`).

    :param rows: The summary rows.
    :param quantity: The quantity name (``'aMe'``, ``'tab_area'``, ...).
    :param scope: The scope.
    :param region: The region.
    :return: The row, or ``None``.
    """
    return (_lookup(rows, f'flexion_cv_{quantity}', scope, region)
            or _lookup(rows, f'flexion_spread_{quantity}', scope, region))


def _git(*args: str) -> str:
    """
    Run a git command in the repository this script belongs to.

    :param args: The git arguments.
    :return: Its stripped stdout, or ``'unknown'`` if git is unavailable or fails.
    """
    try:
        return subprocess.run(['git', '-C', str(REPO_ROOT), *args], capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return 'unknown'


def write_meta(path: Path, arguments: argparse.Namespace, paths: Sequence[str],
               records: Sequence[dict], selected: Sequence[dict], counts: dict):
    """
    Record which code and which inputs produced this report (as ``process_nako_large.py`` does).

    A report whose point is to be re-derivable has to say what it was derived from, including
    whether the checkout was dirty at the time.

    :param path: The meta JSON file to write.
    :param arguments: The parsed command line.
    :param paths: The resolved input scan paths.
    :param records: All measured records.
    :param selected: The records the statistics were computed over.
    :param counts: ``{'load_pairs', 'flexion_series'}`` from :func:`reproducibility_tables`.
    """
    meta = {
        'commit': _git('rev-parse', 'HEAD'),
        'dirty': _git('status', '--porcelain', '--untracked-files=no') != '',
        'branch': _git('rev-parse', '--abbrev-ref', 'HEAD'),
        'date': datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'),
        'host': socket.gethostname(),
        'args': {k: (str(v) if isinstance(v, Path) else v)
                 for k, v in sorted(vars(arguments).items())},
        'inputs': [str(p) for p in arguments.input],
        'n_scans_found': len(paths),
        'n_measured': len(records),
        'n_aggregated': len(selected),
        'load_pairs': counts['load_pairs'],
        'flexion_series': counts['flexion_series'],
        'excluded_by_flag': sorted(r['scan'] for r in records
                                   if r['qc_flags']
                                   and r['scan'] not in {k['scan'] for k in selected}),
        'excluded_by_name': sorted(arguments.exclude),
        'failed': sorted(r['scan'] for r in records if r['status'] != 'ok'),
        'seconds_per_bone': timing_summary(records),
    }
    path.write_text(json.dumps(meta, indent=2) + '\n')
    log.info('wrote %s', path)


def _write_qc_section(out: List[str], rows: Sequence[dict], records: Sequence[dict],
                      selected: Sequence[dict]):
    """
    Append the report's QC section: counts, the flagged / failed scans, geometry checks.

    :param out: The markdown lines, appended to in place.
    :param rows: The summary rows.
    :param records: All measured records.
    :param selected: The records the statistics were computed over.
    """
    out.append('## QC')
    out.append('')
    flagged = [r for r in records if r['qc_flags']]
    failed = [r for r in records if r['status'] != 'ok']
    out.append(f'* measured: {len(records)}; fully successful: '
               f'{sum(1 for r in records if r["status"] == "ok")}; '
               f'with a failed bone or load error: {len(failed)}; QC-flagged: {len(flagged)}.')
    larger = _lookup(rows, 'geometry', '', 'tibia_right_plate_larger')
    if larger is not None and larger['n']:
        out.append(f'* the patient-right tibial plate is the larger one in '
                   f'{round(larger["mean"] * larger["n"])} of {larger["n"]} aggregated scans '
                   f'(a 0/1 flag per scan, so only this share is meaningful).')
    out.append('')
    if failed or flagged:
        out.append('| scan | status | qc_flags | error |')
        out.append('|---|---|---|---|')
        for name in sorted(set(r['scan'] for r in failed) | set(r['scan'] for r in flagged)):
            record = next(r for r in records if r['scan'] == name)
            out.append(f'| {record["scan"]} | {record["status"]} | '
                       f'{record["qc_flags"] or "-"} | {record["error"] or "-"} |')
        out.append('')
    out.append('Geometry checks over the aggregated scans (median / min / max):')
    out.append('')
    out.append('| metric | median | min | max |')
    out.append('|---|---|---|---|')
    for metric in ('tibia_plate_area_ratio', 'tibia_plate_cab_ratio',
                   'tibia_dropped_area_mm2',
                   'tibia_filled_area_left_mm2', 'tibia_filled_area_right_mm2',
                   'tibia_anterior_cos_left', 'tibia_anterior_cos_right',
                   'tibia_transfer_max_mm', 'femur_transfer_max_mm',
                   'femur_ap_extent_left_mm', 'femur_ap_extent_right_mm',
                   'femur_shaft_residual_mm', 'femur_shaft_bins'):
        row = _lookup(rows, 'geometry', '', metric)
        out.append(f'| {metric} | {_cell(row, "p50", 3)} | {_cell(row, "min", 3)} | '
                   f'{_cell(row, "max", 3)} |')
    out.append('')
    timings = timing_summary(selected)
    if timings:
        out.append('Wall-clock seconds per bone (median [min-max]): '
                   + '; '.join(f'{name.replace("_seconds", "")} '
                               f'{t["median"]:.1f} [{t["min"]:.1f}-{t["max"]:.1f}]'
                               for name, t in timings.items())
                   + '. These inflate with `--workers > 1`, so they are kept out of '
                     '`cohort_summary.csv` (which stays comparable between runs) and recorded '
                     'per scan in `scan_geometry.csv` and in the run\'s `.meta.json`.')
        out.append('')


def write_report(rows: Sequence[dict], records: Sequence[dict], selected: Sequence[dict],
                 counts: dict, path: Path, arguments: argparse.Namespace):
    """
    Write the human-readable markdown report.

    :param rows: The summary rows (every number printed here is also in them).
    :param records: All measured records (for the QC section).
    :param selected: The records the statistics were computed over.
    :param counts: ``{'load_pairs', 'flexion_series'}`` from
        :func:`reproducibility_tables`.
    :param path: Destination markdown path.
    :param arguments: The parsed command line, echoed into the report's header.
    """
    out: List[str] = []
    out.append('# Knee cartilage cohort report (Wirth & Eckstein 2008 subregions)')
    out.append('')
    out.append(f'Scans measured: **{len(records)}**; aggregated: **{len(selected)}** '
               f'(flagged scans {"kept" if arguments.include_flagged else "dropped"}'
               + (f', excluded by name: {", ".join(arguments.exclude)}'
                  if arguments.exclude else '') + ').')
    out.append(f'Load pairs: {counts["load_pairs"]}; flexion series '
               f'(>= {arguments.min_angles} angles): {counts["flexion_series"]}.')
    out.append('')
    out.append('Region keys are **patient-side** (`clt` = central patient-left tibial plate, '
               '`iclf` = internal third of the patient-left femoral condyle). The knees\' '
               'laterality is not recorded, so **our plates and condyles cannot be matched to '
               'the paper\'s medial and lateral ones**; the paper\'s values are therefore '
               'printed in a table of their own, below ours, and no row pairs the two.')
    out.append('')

    # --- Table I analogue
    out.append('## Table I analogue: tibial region shares (% of that plate\'s tAB)')
    out.append('')
    out.append('| region | patient-left mean (SD) | min-max | patient-right mean (SD) | '
               'min-max |')
    out.append('|---|---|---|---|---|')
    for letter in ('c', 'i', 'e', 'a', 'p'):
        name = TIBIA_REGION_NAMES[letter]
        left = _lookup(rows, 'tibia_region_shares', 'plate_left', name)
        right = _lookup(rows, 'tibia_region_shares', 'plate_right', name)
        out.append(f'| {name} ({letter}) | {_mean_sd(left)} | {_range(left, 2)} | '
                   f'{_mean_sd(right)} | {_range(right, 2)} |')
    out.append('')

    # --- Table II analogue
    out.append('## Table II analogue: femoral thirds (% of that condyle\'s cF tAB)')
    out.append('')
    out.append('| third | patient-left mean (SD) | min-max | patient-right mean (SD) | '
               'min-max |')
    out.append('|---|---|---|---|---|')
    for letter in ('i', 'c', 'e'):
        name = FEMUR_REGION_NAMES[letter]
        left = _lookup(rows, 'femur_region_shares', 'condyle_left', name)
        right = _lookup(rows, 'femur_region_shares', 'condyle_right', name)
        out.append(f'| {name} ({letter}) | {_mean_sd(left, 2)} | {_range(left, 2)} | '
                   f'{_mean_sd(right, 2)} | {_range(right, 2)} |')
    out.append('')

    # --- The paper's own values, unpaired with ours on purpose (see above).
    out.append('### Wirth & Eckstein 2008, Tables I and II (unpaired reference values)')
    out.append('')
    out.append('Medial / lateral, as published. Which of our two plates (or condyles) is the '
               'medial one is unknown for this cohort, so these columns must not be read '
               'against the two above row by row.')
    out.append('')
    out.append('| region | paper MT | paper LT | third | paper cMF | paper cLF |')
    out.append('|---|---|---|---|---|---|')
    for index, letter in enumerate(('c', 'i', 'e', 'a', 'p')):
        name = TIBIA_REGION_NAMES[letter]
        third = ('i', 'c', 'e')[index] if index < 3 else None
        third_cells = (f'{FEMUR_REGION_NAMES[third]} ({third}) | '
                       f'{PAPER_FEMUR_SHARES["cMF"][third]:.1f} | '
                       f'{PAPER_FEMUR_SHARES["cLF"][third]:.1f}') if third else '- | - | -'
        out.append(f'| {name} ({letter}) | {PAPER_TIBIA_SHARES["MT"][letter]:.1f} | '
                   f'{PAPER_TIBIA_SHARES["LT"][letter]:.1f} | {third_cells} |')
    out.append('')

    # --- Areas
    out.append('## Subregion areas and denuded fraction')
    out.append('')
    out.append('| bone | region | tAB mm2 mean (SD) | tAB min-max | cAB mm2 mean | '
               'denuded fraction mean | denuded max |')
    out.append('|---|---|---|---|---|---|---|')
    for bone, keys, aggregates in (('tibia', TIBIA_REGION_KEYS, TIBIA_AGGREGATES),
                                   ('femur', FEMUR_REGION_KEYS, FEMUR_AGGREGATES)):
        for region in list(keys) + list(aggregates):
            tab = _lookup(rows, 'tab_area', bone, region)
            cab = _lookup(rows, 'cab_area', bone, region)
            den = _lookup(rows, 'denuded_fraction', bone, region)
            out.append(f'| {bone} | {region} | {_mean_sd(tab, 1)} | {_range(tab, 1)} | '
                       f'{_cell(cab, "mean", 1)} | {_cell(den, "mean", 4)} | '
                       f'{_cell(den, "max", 4)} |')
    out.append('')

    # --- Thickness
    out.append('## Thickness per subregion (ThC.cAB unless marked tAB)')
    out.append('')
    out.append('The `plate_*` / `cf_*` / `whole` rows are unions of subregions. Their `aMe` '
               '(and `cMe` / `bMe` / `tab_aMe`) is **exact** -- each direction is weighted by '
               'its own area -- and so are `aMin` / `aMax` / `tab_aMin` (min / max over the '
               'parts). Their `aSD`, `aMav`, `aMiv` and `tab_aSD` are **area-weighted means '
               'of the subregion values**, which is not the union\'s statistic: an aggregate '
               '`aSD` averages the within-subregion spreads and so drops the (large) '
               'between-subregion variance, and an aggregate `aMav` / `aMiv` averages the '
               'subregions\' own 1% tails rather than the union\'s. A `tab_aMin` of exactly '
               '0.0 means some triangle in the region is denuded, not that the cartilage is '
               'thin there.')
    out.append('')
    out.append('| bone | region | aMe mean (SD) | aMe min-max | aSD mean | aMin mean | '
               'aMax mean | tab_aMe mean | tab_aMin mean |')
    out.append('|---|---|---|---|---|---|---|---|---|')
    for bone, keys, aggregates in (('tibia', TIBIA_REGION_KEYS, TIBIA_AGGREGATES),
                                   ('femur', FEMUR_REGION_KEYS, FEMUR_AGGREGATES)):
        for region in list(keys) + list(aggregates):
            ame = _lookup(rows, 'thickness_aMe', bone, region)
            out.append(f'| {bone} | {region} | {_mean_sd(ame, 3)} | {_range(ame, 3)} | '
                       f'{_cell(_lookup(rows, "thickness_aSD", bone, region), "mean", 3)} | '
                       f'{_cell(_lookup(rows, "thickness_aMin", bone, region), "mean", 3)} | '
                       f'{_cell(_lookup(rows, "thickness_aMax", bone, region), "mean", 3)} | '
                       f'{_cell(_lookup(rows, "thickness_tab_aMe", bone, region), "mean", 3)} | '
                       f'{_cell(_lookup(rows, "thickness_tab_aMin", bone, region), "mean", 3)} |')
    out.append('')

    out.append('### Whole-bone aMe per flexion angle')
    out.append('')
    angles = sorted({r['angle'] for r in selected if r['angle'] is not None})
    if not angles:
        out.append('No scan name parsed as `K<subject>_<angle>_<load>`, so no flexion angle '
                   'is known for this cohort.')
    else:
        out.append('| bone | ' + ' | '.join(f'{a} deg' for a in angles) + ' |')
        out.append('|---' * (len(angles) + 1) + '|')
        for bone in ('tibia', 'femur'):
            cells = [_cell(_lookup(rows, 'whole_bone_aMe_by_angle', bone, f'{a}deg'), 'mean', 3)
                     for a in angles]
            out.append(f'| {bone} | ' + ' | '.join(cells) + ' |')
    out.append('')

    # --- Reproducibility
    out.append('## Reproducibility: spread, not precision')
    out.append('')
    out.append('These are **not** repeat scans. "Load" compares the loaded and unloaded scan '
               'of one subject at one flexion angle, "flexion" the spread across the flexion '
               'angles of one subject and load; the knee is repositioned and loaded '
               'differently between them, and the segmentation itself changes with it. The '
               'paper\'s 19-84 um test-retest rms SD is a different quantity and must not be '
               'compared with these.')
    out.append('')
    if not counts['load_pairs'] and not counts['flexion_series']:
        # Both need K<subject>_<angle>_<load> names; without them there is nothing to pair.
        out.append('*Skipped: no scan name parsed as `K<subject>_<angle>_<load>`, so neither '
                   'the loaded/unloaded pairs nor the flexion series could be formed.*')
        out.append('')
        _write_qc_section(out, rows, records, selected)
        path.write_text('\n'.join(out) + '\n')
        log.info('wrote %s', path)
        return
    out.append('| quantity | load abs-difference median (p90, max) | '
               'flexion SD median (p90, max) | flexion CV% median (p90, max) |')
    out.append('|---|---|---|---|')
    pooled_scopes = ('pooled', 'pooled_tibia', 'pooled_femur')
    for (quantity, unit), scope in itertools.product(
            (('region_share', 'pp'), ('tab_area', '%'), ('aMe', 'mm')), pooled_scopes):
        load = _lookup(rows, f'load_spread_{quantity}', scope, '__all__')
        flex = _lookup(rows, f'flexion_spread_{quantity}', scope, '__all__')
        cv = _flexion_cv(rows, quantity, scope, '__all__')
        label = scope.replace('pooled_', '') if scope != 'pooled' else 'both bones'
        out.append(f'| {quantity} ({unit}), {label} | '
                   f'{_cell(load, "p50", 3)} ({_cell(load, "p90", 3)}, {_cell(load, "max", 3)}) '
                   f'n={_cell(load, "n", 0)} | '
                   f'{_cell(flex, "p50", 3)} ({_cell(flex, "p90", 3)}, {_cell(flex, "max", 3)}) | '
                   f'{_cell(cv, "p50", 2)} ({_cell(cv, "p90", 2)}, {_cell(cv, "max", 2)}) |')
    out.append('')
    out.append('Per region (flexion spread, and the loaded-vs-unloaded difference):')
    out.append('')
    out.append('| bone | region | share flexion SD (pp) | tAB flexion CV (%) | '
               'aMe flexion SD (mm) | aMe load abs-difference (mm) |')
    out.append('|---|---|---|---|---|---|')
    for bone, keys, aggregates in (('tibia', TIBIA_REGION_KEYS, TIBIA_AGGREGATES),
                                   ('femur', FEMUR_REGION_KEYS, FEMUR_AGGREGATES)):
        for region in list(keys) + list(aggregates):
            out.append(
                f'| {bone} | {region} | '
                f'{_cell(_lookup(rows, "flexion_spread_region_share", bone, region), "p50", 3)} | '
                f'{_cell(_flexion_cv(rows, "tab_area", bone, region), "p50", 2)} | '
                f'{_cell(_lookup(rows, "flexion_spread_aMe", bone, region), "p50", 3)} | '
                f'{_cell(_lookup(rows, "load_spread_aMe", bone, region), "p50", 3)} |')
    out.append('')

    _write_qc_section(out, rows, records, selected)

    path.write_text('\n'.join(out) + '\n')
    log.info('wrote %s', path)


def main(argv=None):
    """
    Parse the command line, measure the cohort and write the report.

    :param argv: Argument list, defaulting to ``sys.argv[1:]``.
    :return: None.
    """
    parser = argparse.ArgumentParser(
        description='Regenerate the dfg_knees knee cartilage cohort tables.',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-i', '--input', required=True, nargs='+',
                        help='knee segmentation NIfTI files and/or directories (searched '
                             'recursively for .nii/.nii.gz)')
    parser.add_argument('-o', '--output', required=True, type=Path,
                        help='output directory for the CSVs and the markdown report')
    parser.add_argument('--workers', type=int, default=None,
                        help='worker processes (default: all visible CPUs; 1 runs in-process, '
                             'which makes the per-scan timings meaningful)')
    parser.add_argument('--femur-label', type=int, default=3,
                        help='label of the femoral cartilage (default: 3)')
    parser.add_argument('--tibia-label', type=int, default=4,
                        help='label of the tibial cartilage (default: 4)')
    parser.add_argument('--tibia-bone-label', type=int, default=2,
                        help='label of the tibial bone (default: 2)')
    parser.add_argument('--femur-bone-label', type=int, default=1,
                        help='label of the femoral bone (default: 1)')
    parser.add_argument('--include-flagged', action='store_true',
                        help='keep QC-flagged scans in the cohort statistics (default: drop '
                             'them; they are always listed in the QC section)')
    parser.add_argument('--exclude', nargs='+', default=[], metavar='SCAN',
                        help='scan names to drop from the statistics (e.g. K11_90_belastet)')
    parser.add_argument('--min-angles', type=int, default=3,
                        help='smallest number of flexion angles a series must still have to '
                             'enter the flexion spread (default: 3)')
    parser.add_argument('-v', '--verbose', action='store_true', help='debug logging')
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s', stream=sys.stdout)
    warnings.simplefilter('ignore')

    paths = resolve_inputs(args.input)
    log.info('%d scans to measure', len(paths))
    args.output.mkdir(parents=True, exist_ok=True)

    records = measure_cohort(paths, femur_label=args.femur_label, tibia_label=args.tibia_label,
                             tibia_bone_label=args.tibia_bone_label,
                             femur_bone_label=args.femur_bone_label, workers=args.workers)
    selected = select_scans(records, include_flagged=args.include_flagged,
                            exclude=args.exclude)
    log.info('%d of %d scans enter the cohort statistics', len(selected), len(records))

    rows = region_tables(selected) + paper_comparison_rows(selected)
    reproducibility, counts = reproducibility_tables(selected, min_angles=args.min_angles)
    rows += reproducibility + geometry_tables(selected)

    write_scan_csv(records, args.output / DEFAULT_SCAN_CSV)
    write_geometry_csv(records, args.output / DEFAULT_GEOMETRY_CSV)
    write_summary_csv(rows, args.output / DEFAULT_SUMMARY_CSV)
    write_report(rows, records, selected, counts, args.output / DEFAULT_REPORT, args)
    write_meta(args.output / DEFAULT_META, args, paths, records, selected, counts)


if __name__ == '__main__':
    main()
