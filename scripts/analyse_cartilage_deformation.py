"""Knee cartilage deformation: thickness change of every configuration against unloaded 30 deg.

The ``dfg_knees`` cohort has, per knee, up to eight segmentations ``K<n>_<angle>_<load>``:
flexion 0/30/60/90 degrees x unloaded (``unbelastet``) / loaded (``belastet``). This script
compares each knee's configurations against its **unloaded 30 degree** scan (the reference,
:data:`~morphometry.cartilage.deformation.REFERENCE`), in the seven comparisons of
:data:`~morphometry.cartilage.deformation.COMPARISONS`, and writes per-knee tables and
figures. The design is ``docs/superpowers/specs/2026-11-02-knee-cartilage-delta-maps-design.md``.

Stages
------
1. **Measure** every scan once (:func:`measure_scan`): both bones' Wirth & Eckstein 2008
   subdivision and thickness, detached from the image as a picklable
   :class:`~morphometry.cartilage.deformation.BoneMeasurement`, plus the scan's
   :func:`~morphometry.cartilage.knee.scan_qc_flags`. Loaded like the other knee scripts:
   LPI, **no** ``remove_outliers()``. Scans run in a ``ProcessPoolExecutor(--workers)`` and are
   cached under ``<out>/cache/<scan>.pkl`` (owner-only permissions), reused only when the
   cached entry was produced by the same git commit from a clean checkout, from a file with
   the same size and mtime, with the same labels (``--no-cache`` disables the cache).
2. **Per knee** (:func:`process_knee`, a second process pool): per bone, every configuration's
   bone surface is rigidly registered onto the reference's
   (:func:`~morphometry.cartilage.deformation.knee_delta_maps`), giving the Δ maps
   (configuration - reference, mm, negative = thinner) on the reference's map grid; the
   largest-thinning hotspot per compartment is located on the smoothed Δ map, at least
   ``--edge-margin-mm`` (2.0) from each scan's own map footprint edge
   (:func:`~morphometry.cartilage.deformation.find_largest_thinning`); the per-subregion Δ is
   ``aMe(config) - aMe(reference)`` with **each scan's own** subdivision.
3. **Cohort** (:func:`cohort_stage`): the per-knee rows are selected (:func:`select_rows`:
   status ``ok``, no QC flag unless ``--include-flagged``, neither the configuration nor the
   knee's reference scan in ``--exclude``), mapped to medial/lateral names when a
   ``--laterality-table`` is given (:func:`to_anatomical`; knees missing from it are dropped
   and listed), and pooled: per comparison x bone x subregion the n / mean / SD / median /
   min / max of Δ in mm and % with a two-sided Wilcoxon signed-rank p-value against 0,
   **uncorrected** (:func:`subregion_summary`, NaN-aware), and per comparison x bone x
   compartment how often the hotspot lands in each subregion (:func:`hotspot_frequency`).
   Without a laterality table everything is pooled by patient-side key, i.e. under the
   assumption that all knees are the same side, which ``report.md`` states.

Statuses
--------
Every knee gets exactly one row per comparison x bone x subregion in ``subregion_delta.csv``
and per comparison x bone x compartment in ``hotspots.csv``, so absences stay visible:

* ``missing`` -- there is no file for that configuration (seven knees have no 90 degree
  loaded scan). A missing configuration stays ``missing`` even when the reference failed.
* ``failed`` -- the file exists but that bone's measurement failed, its registration raised
  (the configuration is in ``BoneDeltaMaps.errors``), or the **reference** failed for that
  bone (missing reference file, failed measurement, or a reference that cannot be mapped):
  then every present comparison of that knee and bone is ``failed``. A hotspot row is also
  ``failed`` when its compartment has no measured Δ cell. Failed rows keep whatever ``aMe``
  values were measured but no Δ. The reasons are in ``failures.csv``.
* ``ok`` -- otherwise. An ``ok`` row may still carry QC flags.

QC flags
--------
``qc_flags`` is the ``;``-join of the reference scan's and the configuration's scan flags
(``non_lpi_orientation``, ``tibia_transfer_far``, ``femur_transfer_far``) plus, per bone,
``registration_rms:<mm>`` when the registration residual exceeds ``--max-registration-rms``
and ``registration_rotation:<deg>`` when the ICP rotated more than
:data:`MAX_ROTATION_CHANGE_DEG` away from the frame-derived start (a symmetric bone that slid
into a flipped minimum, which the trimmed residual cannot see). Flagged rows are written as
usual; the cohort stage leaves them out of its statistics unless ``--include-flagged``.

Outputs (into ``-o``)
---------------------
* ``subregion_delta.csv`` -- :data:`SUBREGION_COLUMNS`; ``subregion`` is the patient-side key,
  ``anatomical_subregion`` its medial/lateral name from ``--laterality-table`` (empty without
  a table or for a knee missing from it).
* ``hotspots.csv`` -- :data:`HOTSPOT_COLUMNS` (``anatomical_subregion`` as above); map
  coordinates in the reference's map frame
  (tibia: plateau plane, patient right / anterior; femur: medial-lateral / unrolled arc), world
  coordinates = the reference scan's RAS mm position of the hotspot cell
  (``BoneDeltaMaps.reference_positions``; NaN where the reference has no covered triangle);
  ``edge_distance_mm`` = the hotspot cell's distance to the nearer of the two scans' footprint
  edges (a boundary cell reads one grid step).
* ``registration.csv`` -- :data:`REGISTRATION_COLUMNS`, one row per registered
  (comparison, bone).
* ``failures.csv`` -- :data:`FAILURE_COLUMNS`, one row per failed measurement, registration
  or hotspot search, an unexpected crash of one bone's delta maps (stage ``delta_maps``,
  that bone only) or a figure that could not be drawn (stage ``figure``, rows unaffected).
* ``<knee>_tibia_delta.png`` / ``<knee>_femur_delta.png`` -- the 2 x 4 grid of
  :func:`~morphometry.cartilage.deformation_plot.plot_knee_deltas` (not drawn for a bone whose
  reference failed).
* ``cohort_summary.csv`` -- :data:`SUMMARY_COLUMNS`: the ``subregion_delta`` table (units
  ``mm`` and ``%``) and the ``hotspot_frequency`` table (``compartment`` filled, ``count`` of
  ``n``).
* ``tibia_hotspots_cohort.png`` / ``femur_hotspots_cohort.png`` -- every selected knee's
  hotspot per comparison (:func:`~morphometry.cartilage.deformation_plot.plot_cohort_hotspots`;
  medial positive with a laterality table).
* ``report.md`` -- the per-comparison tables, hotspot counts, registration residual summary,
  the lists of missing / failed / flagged / excluded comparisons and the caveats (uncorrected
  p-values, acquisition-to-acquisition spread, laterality assumption).
* ``report.meta.json`` -- commit, dirty flag, branch, host, date, args, inputs, counts.

Example
-------
::

    python scripts/analyse_cartilage_deformation.py \\
        -i /home/simon/Downloads/dfg_knees_segmentations/dfg_knees/segmentations \\
        -o out/deformation --workers 16 --knees K1 K7
"""
import argparse
import csv
import datetime
import json
import logging
import math
import os
import pickle
import re
import socket
import subprocess
import sys
import tempfile
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# Headless figures; must precede any pyplot import (deformation_plot imports it).
os.environ.setdefault('MPLBACKEND', 'Agg')

# Allow running the script standalone (repo root on the path), matching the other
# scripts in this directory.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from morphometry.cartilage.deformation_plot import plot_knee_deltas
from morphometry.cartilage.deformation import (
    COMPARISONS, COMPARTMENTS, DEFAULT_HOTSPOT_EDGE_MARGIN_MM, REFERENCE, bone_measurement,
    find_largest_thinning,
    knee_delta_maps,
)
from morphometry.cartilage.deformation_plot import plot_cohort_hotspots
from morphometry.cartilage.femur_regions import (
    FEMUR_REGION_KEYS, ZONE_THICKNESS_KEYS, anatomical_femur_names,
)
from morphometry.cartilage.registration import REGISTRATION_MAX_RMS_MM
from morphometry.cartilage.tibia_regions import TIBIA_REGION_KEYS, anatomical_tibia_names

log = logging.getLogger('analyse_cartilage_deformation')

NIFTI_SUFFIXES = ('.nii', '.nii.gz')

#: ``K<subject>_<flexion angle>_<load>`` (the dfg_knees naming), on the suffix-stripped name.
_NAME_RE = re.compile(r'^(K\d+)_(\d+)_(belastet|unbelastet)$')

#: The two bones, in output order, with their region keys (the ``subregion`` column).
BONES = ('tibia', 'femur')
REGION_KEYS = {'tibia': TIBIA_REGION_KEYS, 'femur': FEMUR_REGION_KEYS}

#: A comparison whose ICP rotation differs from the frame-derived initial rotation by more than
#: this (degrees) is flagged ``registration_rotation:<deg>``: the initial frames agree to a few
#: degrees on a sound scan, so a much larger change means the ICP slid away from the start
#: (e.g. a flip of a near-symmetric bone, which the trimmed residual cannot detect).
MAX_ROTATION_CHANGE_DEG = 15.0

#: Failures caught per measurement / registration; anything else in a worker is recorded too
#: (with its type name) but marks the scan as crashed (never cached).
CAUGHT = (RuntimeError, AssertionError, ValueError)

SUBREGION_COLUMNS = ['knee', 'angle', 'load', 'bone', 'subregion', 'anatomical_subregion',
                     'aMe_ref_mm', 'aMe_mm', 'delta_mm', 'delta_pct', 'status', 'qc_flags']
HOTSPOT_COLUMNS = ['knee', 'angle', 'load', 'bone', 'compartment', 'map_x_mm', 'map_y_mm',
                   'world_x', 'world_y', 'world_z', 'subregion', 'anatomical_subregion',
                   'smoothed_delta_mm',
                   'delta_mm', 'kernel_size', 'edge_distance_mm', 'fallback', 'status',
                   'qc_flags']
REGISTRATION_COLUMNS = ['knee', 'angle', 'load', 'bone', 'rms_mm', 'median_mm', 'p95_mm',
                        'n_points', 'iterations', 'converged', 'rotation_change_deg',
                        'dropped_fraction', 'flagged']
FAILURE_COLUMNS = ['knee', 'angle', 'load', 'bone', 'stage', 'error']
#: ``cohort_summary.csv``. ``compartment`` is filled for the ``hotspot_frequency`` table only
#: (the ``subregion_delta`` table is per subregion); ``count`` likewise.
SUMMARY_COLUMNS = ['table', 'angle', 'load', 'bone', 'compartment', 'region', 'unit', 'n',
                   'mean', 'sd', 'median', 'min', 'max', 'wilcoxon_p_uncorrected', 'count']

DEFAULT_SUBREGION_CSV = 'subregion_delta.csv'
DEFAULT_HOTSPOT_CSV = 'hotspots.csv'
DEFAULT_REGISTRATION_CSV = 'registration.csv'
DEFAULT_FAILURE_CSV = 'failures.csv'
DEFAULT_SUMMARY_CSV = 'cohort_summary.csv'
DEFAULT_REPORT = 'report.md'
DEFAULT_META = 'report.meta.json'

REPO_ROOT = Path(__file__).resolve().parents[1]
NAN = float('nan')


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

    :param name: The scan id or filename (a ``.nii`` / ``.nii.gz`` suffix is stripped first).
    :return: ``(knee, angle, load)``, e.g. ``('K1', 60, 'belastet')``; all three ``None`` for a
        name that does not match.
    """
    match = _NAME_RE.match(_strip_nifti_suffix(os.path.basename(name)))
    if match is None:
        return None, None, None
    knee, angle, load = match.groups()
    return knee, int(angle), load


def _natural_knee_key(knee: str) -> tuple:
    """
    Sort key so K2 precedes K10.

    :param knee: A ``K<n>`` knee id.
    :return: ``(n, knee)``.
    """
    return (int(knee[1:]), knee)


def discover_knees(paths: Sequence[str]) -> Dict[str, Dict[tuple, str]]:
    """
    Group scan files by knee and configuration.

    :param paths: Scan file paths. Names that do not parse as ``K<n>_<angle>_<load>`` are
        skipped with a WARNING.
    :return: ``{knee: {(angle, load): path}}`` with the knees in natural order (K2 before K10)
        and each knee's configurations sorted by load then angle.
    :raises ValueError: If one configuration of a knee is given twice (e.g. the same scan in
        two directories) -- all configurations of a knee must come from one label set.
    """
    knees: Dict[str, Dict[tuple, str]] = {}
    for path in paths:
        knee, angle, load = parse_scan_name(path)
        if knee is None:
            log.warning('skipping %s: not a K<n>_<angle>_<load> scan name', path)
            continue
        configs = knees.setdefault(knee, {})
        if (angle, load) in configs:
            raise ValueError(f'{knee}_{angle}_{load} is given twice: {configs[(angle, load)]} '
                             f'and {path}')
        configs[(angle, load)] = path
    return {knee: dict(sorted(knees[knee].items(), key=lambda kv: (kv[0][1], kv[0][0])))
            for knee in sorted(knees, key=_natural_knee_key)}


# ----------------------------------------------------------------------------------------
# Per-scan measurement (cached)
# ----------------------------------------------------------------------------------------

def load_scan(path: str):
    """
    Load one knee segmentation and reorient it to LPI (no ``remove_outliers()``).

    :param path: Path to the segmentation NIfTI.
    :return: The reoriented ``Segmentation``.
    """
    from morphometry.image_io import Segmentation
    seg = Segmentation('nibabel')
    seg.read_image(path)
    seg.transform_coordinate_system()
    return seg


def measure_scan(path: str, femur_label: int = 3, tibia_label: int = 4,
                 femur_bone_label: int = 1, tibia_bone_label: int = 2) -> dict:
    """
    Measure one scan: both bones' subdivision and thickness, as picklable measurements.

    Each bone is wrapped separately (a tibia that cannot be split into two plates does not cost
    the femur); a scan that cannot be loaded records the load error for both bones. The femur's
    thickness is measured with the tibia when the tibia succeeded (for the shared in-plane map
    frame); the deformation pipeline reads only its per-triangle thickness and statistics,
    which do not depend on that frame.

    :param path: Path to the segmentation NIfTI.
    :param femur_label: Label of the femoral cartilage.
    :param tibia_label: Label of the tibial cartilage.
    :param femur_bone_label: Label of the femoral bone.
    :param tibia_bone_label: Label of the tibial bone.
    :return: ``{'scan', 'tibia', 'femur', 'qc_flags', 'errors', 'seconds'}``: ``tibia`` /
        ``femur`` a :class:`~morphometry.cartilage.deformation.BoneMeasurement` or ``None``
        when that bone failed, ``qc_flags`` the scan's ``scan_qc_flags`` string, ``errors``
        ``{bone: '<ExceptionType>: <message>'}`` for the failed bones.
    """
    from morphometry.cartilage.knee import Femur, Tibia, scan_qc_flags

    started = time.time()
    scan = _strip_nifti_suffix(os.path.basename(path))
    record = {'scan': scan, 'tibia': None, 'femur': None, 'qc_flags': '', 'errors': {},
              'seconds': NAN}
    try:
        seg = load_scan(path)
    except CAUGHT + (OSError,) as error:
        message = f'load: {type(error).__name__}: {error}'
        record['errors'] = {bone: message for bone in BONES}
        return record

    tibia = femur = None
    try:
        tibia = Tibia(seg, cartilage_label=tibia_label, bone_label=tibia_bone_label)
        thickness = tibia.calculate_thickness()
        record['tibia'] = bone_measurement('tibia', tibia.subdivision, thickness)
    except CAUGHT as error:
        record['errors']['tibia'] = f'{type(error).__name__}: {error}'
        tibia = None
    try:
        femur = Femur(seg, cartilage_label=femur_label, bone_label=femur_bone_label)
        thickness = femur.calculate_thickness(tibia)
        record['femur'] = bone_measurement('femur', femur.subdivision, thickness)
    except CAUGHT as error:
        record['errors']['femur'] = f'{type(error).__name__}: {error}'
        femur = None
    try:
        record['qc_flags'] = scan_qc_flags(seg, tibia, femur)
    except CAUGHT as error:
        log.warning('%s: scan_qc_flags failed: %s', scan, error)
    record['seconds'] = time.time() - started
    return record


def crashed_record(path: str, error: BaseException) -> dict:
    """
    The record of a scan whose worker raised something :func:`measure_scan` does not catch.

    :param path: The scan path.
    :param error: The exception (or a ``BrokenProcessPool`` for a worker that died).
    :return: A :func:`measure_scan`-shaped record with both bones failed and ``crashed=True``
        (such a record is never cached).
    """
    message = f'{type(error).__name__}: {error}'
    return {'scan': _strip_nifti_suffix(os.path.basename(path)), 'tibia': None, 'femur': None,
            'qc_flags': '', 'errors': {bone: message for bone in BONES}, 'seconds': NAN,
            'crashed': True}


def _measure_job(arguments: tuple) -> dict:
    """
    Process-pool entry point: :func:`measure_scan`, recording any uncaught exception.

    :param arguments: ``(path, labels)`` with ``labels`` the keyword labels of
        :func:`measure_scan`.
    :return: The scan's record.
    """
    warnings.simplefilter('ignore')
    path, labels = arguments
    try:
        return measure_scan(path, **labels)
    except Exception as error:  # a worker crash must not take the pool down
        return crashed_record(path, error)


def _cache_path(out_dir: Path, path: str) -> Path:
    """
    Where a scan's measurement is cached.

    :param out_dir: The output directory (``-o``).
    :param path: The scan path.
    :return: ``<out_dir>/cache/<scan>.pkl``.
    """
    return Path(out_dir) / 'cache' / f'{_strip_nifti_suffix(os.path.basename(path))}.pkl'


def _file_signature(path: str) -> dict:
    """
    The scan file's size and modification time, to detect a changed input.

    :param path: The scan path.
    :return: ``{'size', 'mtime_ns'}``.
    """
    stat = os.stat(path)
    return {'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns}


def _read_cache(out_dir: Path, path: str, provenance: dict, labels: dict) -> Optional[dict]:
    """
    A scan's cached record, if it is valid for this run.

    Valid = produced by the same commit from a clean checkout, this run's checkout is clean
    too, the scan file has the same size and mtime, and the labels are the same.

    :param out_dir: The output directory.
    :param path: The scan path.
    :param provenance: ``{'commit', 'dirty'}`` of this run.
    :param labels: The labels passed to :func:`measure_scan`.
    :return: The record, or ``None`` (no entry, stale, or unreadable).
    """
    cache = _cache_path(out_dir, path)
    if provenance.get('dirty', True) or provenance.get('commit') in (None, 'unknown'):
        return None
    try:
        with open(cache, 'rb') as handle:
            entry = pickle.load(handle)
    except (OSError, pickle.UnpicklingError, EOFError, AttributeError, ImportError):
        return None
    if (entry.get('commit') != provenance['commit'] or entry.get('dirty', True)
            or entry.get('file') != _file_signature(path) or entry.get('labels') != labels):
        return None
    return entry['record']


def _write_cache(out_dir: Path, path: str, record: dict, provenance: dict, labels: dict):
    """
    Cache a scan's record, owner-readable only, written atomically.

    :param out_dir: The output directory.
    :param path: The scan path.
    :param record: Its :func:`measure_scan` record.
    :param provenance: ``{'commit', 'dirty'}`` of this run (a dirty entry is never reused).
    :param labels: The labels the record was measured with.
    """
    cache = _cache_path(out_dir, path)
    cache.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    entry = {'record': record, 'commit': provenance.get('commit'),
             'dirty': bool(provenance.get('dirty', True)), 'file': _file_signature(path),
             'labels': dict(labels)}
    fd, tmp = tempfile.mkstemp(dir=cache.parent, prefix='.tmp_', suffix='.pkl')  # mode 0600
    try:
        with os.fdopen(fd, 'wb') as handle:
            pickle.dump(entry, handle, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, cache)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def measure_scans(paths: Sequence[str], out_dir: Path, workers: Optional[int],
                  use_cache: bool, provenance: dict, labels: dict) -> Dict[str, dict]:
    """
    Measure every scan once, from the cache where valid, else in a process pool.

    :param paths: The scan paths.
    :param out_dir: The output directory (the cache lives in ``<out_dir>/cache``).
    :param workers: Worker processes; ``None`` = all CPUs, ``1`` = in-process.
    :param use_cache: ``False`` (``--no-cache``) neither reads nor writes the cache.
    :param provenance: ``{'commit', 'dirty'}`` of this checkout.
    :param labels: Keyword labels for :func:`measure_scan` (``femur_label``, ``tibia_label``,
        ``femur_bone_label``, ``tibia_bone_label``).
    :return: ``{path: record}``. A worker that crashes or dies yields a
        :func:`crashed_record`, not an exception.
    """
    records, todo = {}, []
    for path in paths:
        cached = _read_cache(out_dir, path, provenance, labels) if use_cache else None
        if cached is not None:
            records[path] = cached
        else:
            todo.append(path)
    log.info('%d scans cached, %d to measure', len(records), len(todo))

    def finish(path, record):
        records[path] = record
        if use_cache and not record.get('crashed'):
            _write_cache(out_dir, path, record, provenance, labels)
        status = 'ok' if not record['errors'] else '; '.join(
            f'{bone} failed' for bone in record['errors'])
        log.info('[%d/%d] %s: %s', len(records), len(paths), record['scan'], status)

    if workers == 1:
        for path in todo:
            finish(path, _measure_job((path, labels)))
    elif todo:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_measure_job, (path, labels)): path for path in todo}
            for future in as_completed(futures):
                path = futures[future]
                try:
                    record = future.result()
                except BrokenProcessPool as error:  # the worker died hard
                    record = crashed_record(path, error)
                finish(path, record)
    return records


# ----------------------------------------------------------------------------------------
# Per-knee processing
# ----------------------------------------------------------------------------------------

def _join_flags(*flags: str) -> str:
    """
    ``;``-join non-empty flag strings.

    :param flags: Flag strings (each possibly already ``;``-joined, possibly empty).
    :return: The joined string.
    """
    return ';'.join(f for f in flags if f)


def _ame(measurement, region: str) -> float:
    """
    A subregion's ThC.cAB.aMe from a bone measurement.

    The femoral statistics key the extension zones by their thickness-map names
    (``ZONE_THICKNESS_KEYS``, e.g. ``alf`` -> ``left_anterior_zone``); the region key is
    tried as a fallback.

    :param measurement: A ``BoneMeasurement`` (or ``None``).
    :param region: A region key of ``TIBIA_REGION_KEYS`` / ``FEMUR_REGION_KEYS``.
    :return: The ``aMe`` in mm, ``NaN`` if unavailable.
    """
    if measurement is None:
        return NAN
    statistics = measurement.statistics or {}
    entry = statistics.get(ZONE_THICKNESS_KEYS.get(region, region), statistics.get(region))
    try:
        return float(entry['aMe'])
    except (TypeError, KeyError, ValueError):
        return NAN


def _registration_flags(maps, key: tuple, max_rms_mm: float) -> str:
    """
    Registration QC flags of one registered comparison.

    :param maps: The bone's ``BoneDeltaMaps``.
    :param key: The ``(angle, load)`` of the comparison (must be in ``maps.registrations``).
    :param max_rms_mm: Residual threshold (mm).
    :return: ``registration_rms:<mm>`` and/or ``registration_rotation:<deg>``, ``;``-joined.
    """
    flags = []
    rms = maps.registrations[key].rms_mm
    if rms > max_rms_mm:
        flags.append(f'registration_rms:{rms:.3f}')
    rotation = maps.rotation_change_deg.get(key, NAN)
    if rotation > MAX_ROTATION_CHANGE_DEG:
        flags.append(f'registration_rotation:{rotation:.1f}')
    return ';'.join(flags)


def _hotspot_values(maps, hotspot: Optional[dict]) -> dict:
    """
    The value columns of one hotspot row.

    :param maps: The bone's ``BoneDeltaMaps`` (for the reference world positions).
    :param hotspot: A :func:`find_largest_thinning` result or ``None``.
    :return: Dict of the ``map_*``, ``world_*``, ``subregion``, delta, ``kernel_size``,
        ``edge_distance_mm`` and ``fallback`` columns (NaN / empty when ``hotspot`` is
        ``None``).
    """
    if hotspot is None:
        return {'map_x_mm': NAN, 'map_y_mm': NAN, 'world_x': NAN, 'world_y': NAN,
                'world_z': NAN, 'subregion': '', 'anatomical_subregion': '',
                'smoothed_delta_mm': NAN, 'delta_mm': NAN,
                'kernel_size': NAN, 'edge_distance_mm': NAN, 'fallback': ''}
    i, j = hotspot['cell']
    world = np.asarray(maps.reference_positions[i, j], dtype=float)
    return {'map_x_mm': hotspot['map_mm'][0], 'map_y_mm': hotspot['map_mm'][1],
            'world_x': float(world[0]), 'world_y': float(world[1]), 'world_z': float(world[2]),
            'subregion': hotspot['subregion'], 'anatomical_subregion': '',
            'smoothed_delta_mm': hotspot['smoothed_delta_mm'],
            'delta_mm': hotspot['delta_mm'], 'kernel_size': hotspot['kernel_size'],
            'edge_distance_mm': hotspot['edge_distance_mm'],
            'fallback': bool(hotspot['fallback'])}


def process_knee(knee: str, measurements: Dict[tuple, dict], grid_mm: float = 1.0,
                 neighbourhood_mm: float = 3.0,
                 max_rms_mm: float = REGISTRATION_MAX_RMS_MM,
                 edge_margin_mm: float = DEFAULT_HOTSPOT_EDGE_MARGIN_MM) -> dict:
    """
    Compare one knee's configurations against its reference, per bone.

    :param knee: Knee id (``K<n>``).
    :param measurements: ``{(angle, load): record}`` with the :func:`measure_scan` records of
        the knee's scans; a configuration without a file is simply absent.
    :param grid_mm: Δ-map cell size (mm).
    :param neighbourhood_mm: Smoothing neighbourhood of the hotspot search (mm).
    :param max_rms_mm: Registration residual above which a comparison is flagged (mm).
    :param edge_margin_mm: Minimum distance (mm) of a hotspot cell from each scan's map
        footprint edge (:func:`~morphometry.cartilage.deformation.find_largest_thinning`).
    :return: ``{'subregion_rows', 'hotspot_rows', 'registration_rows', 'failures', 'maps',
        'hotspots', 'panel_titles'}``: the rows of the three CSVs (column sets
        :data:`SUBREGION_COLUMNS` / :data:`HOTSPOT_COLUMNS` / :data:`REGISTRATION_COLUMNS`) and
        of ``failures.csv``; ``maps`` ``{bone: BoneDeltaMaps or None}`` (``None`` when the
        reference failed for that bone); ``hotspots`` ``{bone: {(angle, load): whole-bone
        hotspot dict or None}}`` and ``panel_titles`` ``{bone: {(angle, load): 'no scan' |
        'failed'}}`` for :func:`~morphometry.cartilage.deformation_plot.plot_knee_deltas`.
    """
    out = {'subregion_rows': [], 'hotspot_rows': [], 'registration_rows': [], 'failures': [],
           'maps': {}, 'hotspots': {}, 'panel_titles': {}}
    reference = measurements.get(REFERENCE)
    ref_flags = reference['qc_flags'] if reference is not None else ''

    def failure(key, bone, stage, error):
        out['failures'].append({'knee': knee, 'angle': key[0], 'load': key[1], 'bone': bone,
                                'stage': stage, 'error': error})

    for bone in BONES:
        ref_m = reference[bone] if reference is not None else None
        maps, ref_error = None, ''
        if reference is None:
            ref_error = 'reference: no scan'
        elif ref_m is None:
            ref_error = f"reference: {reference['errors'].get(bone, 'measurement failed')}"
        for key in COMPARISONS:
            record = measurements.get(key)
            if record is not None and record[bone] is None:
                failure(key, bone, 'measurement', record['errors'].get(bone, 'failed'))
        configurations = {key: measurements[key][bone] for key in COMPARISONS
                          if key in measurements and measurements[key][bone] is not None}
        ref_stage = 'reference'
        if ref_m is not None:
            try:
                maps = knee_delta_maps(bone, ref_m, configurations, grid_mm=grid_mm)
            except CAUGHT as error:
                ref_error = f'reference: {type(error).__name__}: {error}'
            except Exception as error:  # an unexpected crash costs this bone only
                ref_stage = 'delta_maps'
                ref_error = f'delta maps crashed: {type(error).__name__}: {error}'
        if ref_error:
            failure(REFERENCE, bone, ref_stage, ref_error)
            log.warning('%s %s: %s; every comparison of this bone fails', knee, bone, ref_error)
        if maps is not None:
            for key, error in maps.errors.items():
                failure(key, bone, 'registration', error)

        out['maps'][bone] = maps
        hotspots, titles = {}, {}
        for key in COMPARISONS:
            record = measurements.get(key)
            if record is None:
                status = 'missing'
            elif maps is None or record[bone] is None or key in maps.errors \
                    or key not in maps.deltas:
                status = 'failed'
            else:
                status = 'ok'
            if status != 'ok':
                titles[key] = 'no scan' if status == 'missing' else 'failed'
            flags = _join_flags(ref_flags, record['qc_flags'] if record is not None else '')
            if status == 'ok':
                flags = _join_flags(flags, _registration_flags(maps, key, max_rms_mm))
            base = {'knee': knee, 'angle': key[0], 'load': key[1], 'bone': bone}

            config_m = record[bone] if record is not None else None
            for region in REGION_KEYS[bone]:
                ref_ame, ame = _ame(ref_m, region), _ame(config_m, region)
                delta = ame - ref_ame if status == 'ok' else NAN
                pct = 100.0 * delta / ref_ame if (status == 'ok' and ref_ame > 0) else NAN
                out['subregion_rows'].append(dict(
                    base, subregion=region, anatomical_subregion='', aMe_ref_mm=ref_ame,
                    aMe_mm=ame, delta_mm=delta, delta_pct=pct, status=status, qc_flags=flags))

            for compartment in COMPARTMENTS:
                hotspot, row_status = None, status
                if status == 'ok':
                    hotspot = find_largest_thinning(maps, key, compartment,
                                                    neighbourhood_mm=neighbourhood_mm,
                                                    edge_margin_mm=edge_margin_mm)
                    if hotspot is None:
                        row_status = 'failed'
                        failure(key, bone, f'hotspot_{compartment}',
                                'no measured delta cell in the compartment')
                if compartment == 'both':
                    hotspots[key] = hotspot
                out['hotspot_rows'].append(dict(
                    base, compartment=compartment, **_hotspot_values(maps, hotspot),
                    status=row_status, qc_flags=flags))

            if status == 'ok':
                result = maps.registrations[key]
                rotation = float(maps.rotation_change_deg.get(key, NAN))
                out['registration_rows'].append(dict(
                    base, rms_mm=result.rms_mm, median_mm=result.median_mm,
                    p95_mm=result.p95_mm, n_points=result.n_points,
                    iterations=result.iterations, converged=bool(result.converged),
                    rotation_change_deg=rotation,
                    dropped_fraction=float(maps.dropped_fraction.get(key, NAN)),
                    flagged=bool(result.rms_mm > max_rms_mm
                                 or rotation > MAX_ROTATION_CHANGE_DEG)))
        out['hotspots'][bone] = hotspots
        out['panel_titles'][bone] = titles
    return out


def write_knee_figures(knee: str, result: dict, out_dir: Path) -> List[str]:
    """
    Draw ``<knee>_tibia_delta.png`` and ``<knee>_femur_delta.png``.

    :param knee: Knee id.
    :param result: Its :func:`process_knee` result.
    :param out_dir: Output directory.
    :return: The paths written (a bone whose reference failed has no map and no figure). A
        figure that cannot be drawn or saved is logged and appended to ``result['failures']``
        with stage ``figure``; it never touches the knee's rows.
    """
    import matplotlib.pyplot as plt

    written = []
    for bone in BONES:
        maps = result['maps'].get(bone)
        if maps is None:
            log.warning('%s %s: no delta maps (reference failed); no figure', knee, bone)
            continue
        path = Path(out_dir) / f'{knee}_{bone}_delta.png'
        try:
            figure = plot_knee_deltas(maps, knee, hotspots=result['hotspots'][bone],
                                      panel_titles=result['panel_titles'][bone],
                                      path=str(path))
            plt.close(figure)
        except Exception as error:
            message = f'{type(error).__name__}: {error}'
            log.error('%s %s: figure failed: %s', knee, bone, message)
            result['failures'].append({'knee': knee, 'angle': REFERENCE[0],
                                       'load': REFERENCE[1], 'bone': bone, 'stage': 'figure',
                                       'error': message})
            plt.close('all')
            continue
        written.append(str(path))
    return written


def failed_record(record: dict, error: str) -> dict:
    """
    A copy of a scan record with both bones failed (used when a knee job crashes).

    :param record: A :func:`measure_scan` record.
    :param error: The error to record for both bones.
    :return: The failed copy (scan name and QC flags kept).
    """
    return dict(record, tibia=None, femur=None, errors={bone: error for bone in BONES})


def _knee_job(arguments: tuple) -> Tuple[str, dict]:
    """
    Process-pool entry point: :func:`process_knee` plus the knee's figures.

    Figures are drawn after the rows, each in its own ``try`` (:func:`write_knee_figures`), so
    a figure failing never fails a row. An exception :func:`process_knee` itself does not
    catch (it already isolates each bone's delta maps) does not lose the knee's rows: the knee
    is re-processed with every scan failed, so every row is written with
    status ``failed`` (``missing`` for absent files) and the error in ``failures.csv``.

    :param arguments: ``(knee, measurements, grid_mm, neighbourhood_mm, max_rms_mm,
        edge_margin_mm, out_dir)``.
    :return: ``(knee, result)``.
    """
    warnings.simplefilter('ignore')
    knee, measurements, grid_mm, neighbourhood_mm, max_rms_mm, edge_margin_mm, out_dir = \
        arguments
    try:
        result = process_knee(knee, measurements, grid_mm, neighbourhood_mm, max_rms_mm,
                              edge_margin_mm)
    except Exception as error:
        message = f'knee job: {type(error).__name__}: {error}'
        log.error('%s: %s', knee, message)
        result = process_knee(knee, {k: failed_record(r, message)
                                     for k, r in measurements.items()},
                              grid_mm, neighbourhood_mm, max_rms_mm, edge_margin_mm)
    # Figures after, and isolated: a PNG that cannot be drawn never fails the knee's rows.
    write_knee_figures(knee, result, out_dir)
    return knee, result


def process_knees(knees: Dict[str, Dict[tuple, dict]], out_dir: Path, workers: Optional[int],
                  grid_mm: float, neighbourhood_mm: float,
                  max_rms_mm: float,
                  edge_margin_mm: float = DEFAULT_HOTSPOT_EDGE_MARGIN_MM) -> Dict[str, dict]:
    """
    Run :func:`_knee_job` for every knee, in a process pool.

    :param knees: ``{knee: {(angle, load): record}}``.
    :param out_dir: Output directory (figures).
    :param workers: Worker processes; ``None`` = all CPUs, ``1`` = in-process.
    :param grid_mm: Δ-map cell size (mm).
    :param neighbourhood_mm: Hotspot smoothing neighbourhood (mm).
    :param max_rms_mm: Registration residual threshold (mm).
    :param edge_margin_mm: Hotspot edge margin (mm).
    :return: ``{knee: process_knee result}`` in the order of ``knees``.
    """
    jobs = [(knee, measurements, grid_mm, neighbourhood_mm, max_rms_mm, edge_margin_mm, out_dir)
            for knee, measurements in knees.items()]
    results = {}
    started = time.time()
    if workers == 1:
        for job in jobs:
            knee, result = _knee_job(job)
            results[knee] = result
            log.info('[%d/%d] knee %s done', len(results), len(jobs), knee)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_knee_job, job): job for job in jobs}
            for future in as_completed(futures):
                job = futures[future]
                try:
                    knee, result = future.result()
                except BrokenProcessPool as error:
                    knee = job[0]
                    message = f'knee job: {type(error).__name__}: {error}'
                    result = process_knee(knee, {k: failed_record(r, message)
                                                 for k, r in job[1].items()},
                                          grid_mm, neighbourhood_mm, max_rms_mm,
                                          edge_margin_mm)
                results[knee] = result
                log.info('[%d/%d] knee %s done (%.0fs elapsed)', len(results), len(jobs), knee,
                         time.time() - started)
    return {knee: results[knee] for knee in knees}


# ----------------------------------------------------------------------------------------
# Output
# ----------------------------------------------------------------------------------------

def _format(value) -> str:
    """
    Format one CSV cell: NaN/None empty, floats to 6 significant digits.

    :param value: The value.
    :return: Its text.
    """
    if value is None:
        return ''
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value))
    if isinstance(value, (float, np.floating)):
        return '' if math.isnan(value) else f'{float(value):.6g}'
    return str(value)


def write_csv(rows: Sequence[dict], columns: Sequence[str], path: Path):
    """
    Write rows to a CSV with exactly ``columns``.

    :param rows: Dicts holding (at least) the columns.
    :param columns: Column order.
    :param path: Output path.
    """
    with open(path, 'w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for row in rows:
            writer.writerow([_format(row.get(column)) for column in columns])


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


def provenance() -> dict:
    """
    The checkout's commit and whether it is dirty (tracked files modified).

    :return: ``{'commit', 'dirty'}``; an unknown commit counts as dirty.
    """
    commit = _git('rev-parse', 'HEAD')
    status = _git('status', '--porcelain', '--untracked-files=no')
    return {'commit': commit, 'dirty': commit == 'unknown' or status != ''}


# ----------------------------------------------------------------------------------------
# Cohort stage: selection, laterality, statistics, report
# ----------------------------------------------------------------------------------------

#: Display names of the two loads in the report.
LOAD_NAMES = {'unbelastet': 'unloaded', 'belastet': 'loaded'}


def _comparison_name(angle, load) -> str:
    """
    A comparison's display name, e.g. ``60° loaded``.

    :param angle: Flexion angle (degrees).
    :param load: ``'unbelastet'`` or ``'belastet'``.
    :return: The name.
    """
    return f'{angle}° {LOAD_NAMES.get(load, load)}'


def load_laterality_table(path) -> Dict[str, str]:
    """
    Read a per-knee laterality table.

    Same format and errors as ``scripts/compute_knee_cartilage_thickness.py``'s table (the
    scripts do not import each other): a CSV with the columns ``subject`` and ``laterality``,
    ``laterality`` ``left`` or ``right`` (case-insensitive; an empty cell means unknown, i.e.
    the knee is treated as absent from the table). Here ``subject`` is the **knee id**
    ``K<n>``, since all configurations of a knee share one laterality.

    :param path: Path to the CSV, or ``None``.
    :return: ``{knee: 'left' | 'right'}`` (empty without a path).
    :raises ValueError: On a missing column or an invalid laterality value.
    """
    if path is None:
        return {}
    table: Dict[str, str] = {}
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        if not {'subject', 'laterality'} <= set(reader.fieldnames or ()):
            raise ValueError(f'{path}: needs the columns subject,laterality')
        for row in reader:
            value = (row['laterality'] or '').strip().lower()
            if not value:
                continue
            if value not in ('left', 'right'):
                raise ValueError(f"{path}: laterality of {row['subject']!r} must be left or "
                                 f"right, got {value!r}")
            table[row['subject'].strip()] = value
    return table


def _scan_name(knee: str, angle, load) -> str:
    """
    The scan id of one configuration of a knee.

    :param knee: Knee id.
    :param angle: Flexion angle.
    :param load: Load.
    :return: ``K<n>_<angle>_<load>``.
    """
    return f'{knee}_{angle}_{load}'


def select_rows(rows: Sequence[dict], include_flagged: bool,
                exclude: Sequence[str]) -> List[dict]:
    """
    The rows that enter the cohort statistics.

    Applies to subregion and hotspot rows alike. Kept: ``status == 'ok'``; dropped: a row with
    non-empty ``qc_flags`` (unless ``include_flagged``) and a row whose configuration scan or
    whose knee's reference scan (``K<n>_30_unbelastet``) is named in ``exclude``.

    :param rows: Per-knee rows (``subregion_delta`` or ``hotspots``).
    :param include_flagged: Keep QC-flagged rows (``--include-flagged``).
    :param exclude: Scan names to leave out (``--exclude``, e.g. ``K7_0_belastet``).
    :return: The kept rows, in input order (the same dict objects).
    """
    excluded = set(exclude or ())
    kept = []
    for row in rows:
        if row.get('status') != 'ok':
            continue
        if row.get('qc_flags') and not include_flagged:
            continue
        if (_scan_name(row['knee'], row['angle'], row['load']) in excluded
                or _scan_name(row['knee'], *REFERENCE) in excluded):
            continue
        kept.append(row)
    return kept


def _names(bone: str, laterality: str) -> Dict[str, str]:
    """
    The patient-side -> anatomical region-name map of one bone.

    :param bone: ``'tibia'`` or ``'femur'``.
    :param laterality: ``'left'`` or ``'right'``.
    :return: ``{patient_side_key: anatomical_name}``
        (:func:`~morphometry.cartilage.tibia_regions.anatomical_tibia_names` /
        :func:`~morphometry.cartilage.femur_regions.anatomical_femur_names`).
    """
    return (anatomical_tibia_names if bone == 'tibia' else anatomical_femur_names)(laterality)


def missing_laterality(rows: Sequence[dict], laterality: Dict[str, str]) -> List[str]:
    """
    The knees of ``rows`` that a supplied laterality table does not cover.

    :param rows: Per-knee rows (any kind with a ``knee`` column).
    :param laterality: ``{knee: 'left' | 'right'}``; empty = no table.
    :return: The uncovered knee ids in natural order (K2 before K10); ``[]`` without a table.
    """
    if not laterality:
        return []
    return sorted({row['knee'] for row in rows if row['knee'] not in laterality},
                  key=_natural_knee_key)


def fill_anatomical_subregion(rows: Sequence[dict], laterality: Dict[str, str]) -> None:
    """
    Fill the ``anatomical_subregion`` column of per-knee rows, in place.

    The ``subregion`` column keeps its patient-side key (the per-knee CSVs stay patient-side);
    only knees in the table get a name, every other row (and a row without a subregion) an
    empty string.

    :param rows: Subregion or hotspot rows (with ``knee``, ``bone``, ``subregion``).
    :param laterality: ``{knee: 'left' | 'right'}``; empty = no table (nothing filled).
    """
    for row in rows:
        side = laterality.get(row['knee'])
        region = row.get('subregion') or ''
        row['anatomical_subregion'] = (_names(row['bone'], side).get(region, '')
                                       if side and region else '')


def to_anatomical(rows: Sequence[dict], laterality: Dict[str, str]) -> List[dict]:
    """
    Map rows to one anatomical convention for pooling, given a laterality table.

    Per row (a copy; the input rows are not modified): ``subregion`` (and
    ``anatomical_subregion``) becomes the medial/lateral name of
    :func:`~morphometry.cartilage.tibia_regions.anatomical_tibia_names` /
    :func:`~morphometry.cartilage.femur_regions.anatomical_femur_names` (``cMT``, ``icLF``,
    ``aMF`` ...); a hotspot row's ``compartment`` ``left`` / ``right`` becomes ``medial`` /
    ``lateral`` (a right knee's patient-left side is medial, a left knee's patient-right side)
    and its ``map_x_mm`` is mirrored so that medial is positive: both map frames have x towards
    the patient's right, so a right knee's x is negated and a left knee's kept. ``both`` stays
    ``both``. Knees missing from the table are dropped (list them with
    :func:`missing_laterality`).

    :param rows: Subregion and/or hotspot rows.
    :param laterality: ``{knee: 'left' | 'right'}``; empty = no table: ``rows`` are returned
        unchanged (patient side, the same-side assumption).
    :return: The mapped rows.
    """
    if not laterality:
        return list(rows)
    out = []
    for row in rows:
        side = laterality.get(row['knee'])
        if side is None:
            continue
        new = dict(row)
        region = row.get('subregion') or ''
        if region:
            new['subregion'] = new['anatomical_subregion'] = _names(row['bone'], side)[region]
        medial = 'left' if side == 'right' else 'right'
        if row.get('compartment') in ('left', 'right'):
            new['compartment'] = 'medial' if row['compartment'] == medial else 'lateral'
        if 'map_x_mm' in row and side == 'right':
            new['map_x_mm'] = -float(row['map_x_mm'])
        out.append(new)
    log.info('laterality table applied: %d of %d rows kept', len(out), len(rows))
    return out


def _finite(values) -> np.ndarray:
    """
    The finite values of a sequence, as floats.

    :param values: Numbers (NaN / None / non-numeric allowed).
    :return: 1-D float array of the finite ones.
    """
    array = np.array([v if isinstance(v, (int, float, np.number)) else NAN for v in values],
                     dtype=float)
    return array[np.isfinite(array)]


def _wilcoxon_p(values: np.ndarray) -> float:
    """
    Two-sided Wilcoxon signed-rank p-value of ``values`` against 0, **uncorrected**.

    :param values: Finite values.
    :return: ``scipy.stats.wilcoxon(values).pvalue``; NaN for fewer than two values, all
        values zero, or a test scipy refuses.
    """
    if len(values) < 2 or not np.any(values != 0):
        return NAN
    from scipy.stats import wilcoxon
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        try:
            return float(wilcoxon(values).pvalue)
        except ValueError:
            return NAN


def _describe(values: np.ndarray) -> dict:
    """
    Descriptive statistics of finite values.

    :param values: Finite values.
    :return: ``{'n', 'mean', 'sd', 'median', 'min', 'max'}``; ``sd`` is the sample SD
        (ddof 1, NaN for n < 2), every other statistic NaN for n = 0.
    """
    n = len(values)
    if n == 0:
        return {'n': 0, 'mean': NAN, 'sd': NAN, 'median': NAN, 'min': NAN, 'max': NAN}
    return {'n': n, 'mean': float(np.mean(values)),
            'sd': float(np.std(values, ddof=1)) if n > 1 else NAN,
            'median': float(np.median(values)), 'min': float(np.min(values)),
            'max': float(np.max(values))}


def _comparison_order(rows: Sequence[dict]) -> List[tuple]:
    """
    The comparisons present in ``rows``, in reporting order (:data:`COMPARISONS`).

    :param rows: Rows with ``angle`` / ``load``.
    :return: ``[(angle, load), ...]``; a comparison outside :data:`COMPARISONS` goes last.
    """
    present = {(row['angle'], row['load']) for row in rows}
    order = [key for key in COMPARISONS if key in present]
    return order + sorted(present - set(order), key=str)


def _first_seen(values) -> list:
    """
    Unique values in order of first appearance.

    :param values: Hashables.
    :return: The de-duplicated list.
    """
    return list(dict.fromkeys(values))


def subregion_summary(rows: Sequence[dict]) -> List[dict]:
    """
    Per comparison x bone x subregion x unit: descriptive statistics and Wilcoxon of Δ.

    NaN-aware: only finite Δ values count (an ``ok`` row whose subregion has no ``aMe`` carries
    a NaN Δ and is not counted in ``n``). Units: ``mm`` (``delta_mm``) and ``%``
    (``delta_pct``). The p-value is the two-sided Wilcoxon signed-rank test of Δ against 0,
    **uncorrected for multiple comparisons** (20 subregions x 7 comparisons per bone).

    :param rows: Selected subregion rows (see :func:`select_rows`), patient-side or mapped by
        :func:`to_anatomical`; ``subregion`` is the pooling key.
    :return: Rows with :data:`SUMMARY_COLUMNS` (``table='subregion_delta'``), ordered by
        comparison (:data:`COMPARISONS`), bone (:data:`BONES`), subregion (first appearance),
        unit.
    """
    groups: Dict[tuple, list] = {}
    for row in rows:
        groups.setdefault((row['angle'], row['load'], row['bone'], row['subregion']),
                          []).append(row)
    out = []
    for angle, load in _comparison_order(rows):
        for bone in BONES:
            regions = _first_seen(row['subregion'] for row in rows
                                  if (row['angle'], row['load'], row['bone'])
                                  == (angle, load, bone))
            for region in regions:
                members = groups[(angle, load, bone, region)]
                for unit, column in (('mm', 'delta_mm'), ('%', 'delta_pct')):
                    values = _finite(row.get(column) for row in members)
                    out.append(dict(table='subregion_delta', angle=angle, load=load,
                                    bone=bone, compartment='', region=region, unit=unit,
                                    **_describe(values),
                                    wilcoxon_p_uncorrected=_wilcoxon_p(values), count=NAN))
    return out


def hotspot_frequency(rows: Sequence[dict]) -> List[dict]:
    """
    How often the largest-thinning hotspot lands in each subregion.

    Per comparison x bone x compartment: ``n`` = the number of hotspot rows (knees), and per
    subregion one row with ``count`` = how many of them sit in it. A hotspot cell without a
    reference region (``subregion`` empty) is counted under ``none``.

    :param rows: Selected hotspot rows (see :func:`select_rows`), patient-side or mapped by
        :func:`to_anatomical`.
    :return: Rows with :data:`SUMMARY_COLUMNS` (``table='hotspot_frequency'``, ``unit``
        ``'count'``, the statistics columns NaN), ordered by comparison, bone, compartment
        (first appearance), subregion (first appearance).
    """
    out = []
    for angle, load in _comparison_order(rows):
        for bone in BONES:
            members = [row for row in rows
                       if (row['angle'], row['load'], row['bone']) == (angle, load, bone)]
            for compartment in _first_seen(row['compartment'] for row in members):
                within = [row for row in members if row['compartment'] == compartment]
                regions = [row.get('subregion') or 'none' for row in within]
                for region in _first_seen(regions):
                    out.append(dict(table='hotspot_frequency', angle=angle, load=load,
                                    bone=bone, compartment=compartment, region=region,
                                    unit='count', n=len(within), mean=NAN, sd=NAN,
                                    median=NAN, min=NAN, max=NAN,
                                    wilcoxon_p_uncorrected=NAN,
                                    count=regions.count(region)))
    return out


def registration_summary(rows: Sequence[dict]) -> List[dict]:
    """
    The registration residual per bone, over every registered comparison.

    :param rows: ``registration.csv`` rows (:data:`REGISTRATION_COLUMNS`).
    :return: Per bone ``{'bone', 'n', 'median_rms', 'p90_rms', 'max_rms', 'n_flagged',
        'n_not_converged'}`` (rms in mm, NaN when the bone has no registration).
    """
    out = []
    for bone in BONES:
        members = [row for row in rows if row['bone'] == bone]
        rms = _finite(row['rms_mm'] for row in members)
        out.append({'bone': bone, 'n': len(members),
                    'median_rms': float(np.median(rms)) if len(rms) else NAN,
                    'p90_rms': float(np.percentile(rms, 90)) if len(rms) else NAN,
                    'max_rms': float(np.max(rms)) if len(rms) else NAN,
                    'n_flagged': sum(bool(row['flagged']) for row in members),
                    'n_not_converged': sum(not bool(row['converged']) for row in members)})
    return out


def _md(value, digits: int = 3) -> str:
    """
    Format a number for a markdown table (NaN as an em dash).

    :param value: The number.
    :param digits: Decimals.
    :return: Its text.
    """
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return '—'
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return f'{float(value):.{digits}f}'


def _grouped(rows: Sequence[dict], keep_flags: bool = False) -> List[str]:
    """
    Group per-bone rows into one line per knee x comparison (bones joined).

    :param rows: Rows with ``knee``, ``angle``, ``load``, ``bone`` (and ``qc_flags``).
    :param keep_flags: Append the rows' ``qc_flags`` (grouped by the flag string too).
    :return: Lines like ``K2 90° loaded (tibia, femur)`` or ``K2 60° unloaded (tibia):
        <flags>``, in input order.
    """
    groups: Dict[tuple, list] = {}
    for row in rows:
        key = (row['knee'], row['angle'], row['load'],
               row.get('qc_flags', '') if keep_flags else '')
        bones = groups.setdefault(key, [])
        if row['bone'] not in bones:
            bones.append(row['bone'])
    lines = []
    for (knee, angle, load, flags), bones in groups.items():
        line = f'{knee} {_comparison_name(angle, load)} ({", ".join(bones)})'
        lines.append(f'{line}: `{flags}`' if flags else line)
    return lines


def write_report(path: Path, args: argparse.Namespace, cohort: dict):
    """
    Write the human-readable ``report.md``.

    :param path: Destination.
    :param args: The parsed command line (inputs, selection options).
    :param cohort: The :func:`cohort_stage` dict (summary rows, selected / all rows, lists).
    """
    anatomical = bool(cohort['laterality'])
    summary = cohort['summary']
    out: List[str] = ['# Knee cartilage deformation against unloaded 30°', '']
    out.append(f'Knees: **{cohort["n_knees"]}**; scans found: **{cohort["n_scans"]}**. '
               f'Label set: {", ".join(f"`{p}`" for p in args.input)}.')
    out.append('')
    out.append('Δ = configuration − reference (unloaded 30° of the same knee), in mm and in % of '
               'the reference aMe (ThC.cAB.aMe of each scan\'s own Wirth & Eckstein 2008 '
               'subdivision); **negative = thinner**. The configurations are **separate '
               'acquisitions of a repositioned knee**, so every Δ contains the '
               'acquisition-to-acquisition (segmentation and repositioning) spread as well as '
               'any deformation; it is not a measure of measurement precision.')
    out.append('')
    out.append('**p-values are uncorrected** for multiple comparisons: a two-sided Wilcoxon '
               'signed-rank test of Δ against 0 per subregion and comparison, i.e. 20 '
               'subregions × 7 comparisons = 140 tests per bone. Read them as a screen, not '
               'as evidence; with n ≤ 15 the smallest attainable p is about 6e-5, with n = 8 '
               'about 0.008.')
    out.append('')
    if anatomical:
        out.append(f'Laterality: from `{args.laterality_table}`; subregions and compartments '
                   'are pooled by **anatomical** name (medial/lateral: `cMT` = central medial '
                   'tibia, `icLF` = internal third of the lateral condyle; `aMF`/`pMF` are our '
                   'names for the trochlear and posterior zones, not the paper\'s), and the '
                   'hotspot scatter is mirrored so that **medial is positive**.')
        if cohort['missing_laterality']:
            out.append('')
            out.append('Knees missing from the laterality table (left out of every cohort '
                       f'statistic): {", ".join(cohort["missing_laterality"])}.')
    else:
        out.append('Laterality: **no laterality table was given, so every knee is assumed to '
                   'be the same side** and subregions are pooled by their **patient-side** key '
                   '(`clt` = central patient-left tibial plate, `iclf` = internal third of the '
                   'patient-left condyle). Patient-left is medial in a right knee and lateral in '
                   'a left knee; if the cohort mixes sides these pooled values mix medial with '
                   'lateral. Pass `--laterality-table` to pool by anatomical name.')
    out.append('')
    selection = ('QC-flagged comparisons are **included**' if args.include_flagged
                 else 'QC-flagged comparisons are **excluded**')
    if args.exclude:
        selection += f'; scans excluded by name: {", ".join(args.exclude)}'
    out.append(f'Selection: only comparisons with status `ok`; {selection}.')
    out.append('')

    # --- n per comparison
    out.append('## Knees per comparison')
    out.append('')
    out.append('| comparison | ' + ' | '.join(BONES) + ' |')
    out.append('|---|' + '---|' * len(BONES))
    for angle, load in COMPARISONS:
        cells = [str(len({r['knee'] for r in cohort['subregion_rows']
                          if (r['angle'], r['load'], r['bone']) == (angle, load, bone)}))
                 for bone in BONES]
        out.append(f'| {_comparison_name(angle, load)} | ' + ' | '.join(cells) + ' |')
    out.append('')
    out.append('A subregion\'s own n (below) can be smaller when its aMe is undefined in a scan.')
    out.append('')

    # --- per comparison tables
    out.append('## Δ per subregion')
    for bone in BONES:
        for angle, load in COMPARISONS:
            rows = [r for r in summary if r['table'] == 'subregion_delta'
                    and (r['angle'], r['load'], r['bone']) == (angle, load, bone)]
            if not rows:
                continue
            out.append('')
            out.append(f'### {bone}, {_comparison_name(angle, load)}')
            out.append('')
            out.append('| subregion | n | mean Δ (mm) | SD | median | min | max | mean Δ (%) '
                       '| p (uncorrected) |')
            out.append('|---|---|---|---|---|---|---|---|---|')
            for mm in (r for r in rows if r['unit'] == 'mm'):
                pct = next(r for r in rows if r['unit'] == '%' and r['region'] == mm['region'])
                out.append(f'| {mm["region"]} | {mm["n"]} | {_md(mm["mean"])} | '
                           f'{_md(mm["sd"])} | {_md(mm["median"])} | {_md(mm["min"])} | '
                           f'{_md(mm["max"])} | {_md(pct["mean"], 1)} | '
                           f'{_md(mm["wilcoxon_p_uncorrected"], 4)} |')
    out.append('')

    # --- hotspot frequency
    out.append('## Where the largest thinning sits (hotspot subregion counts)')
    out.append('')
    out.append('Per comparison and compartment: in how many knees the hotspot (minimum of the '
               'smoothed Δ map) lies in each subregion (`none` = a cell with no reference '
               'region). Scatter plots: `tibia_hotspots_cohort.png`, '
               '`femur_hotspots_cohort.png`.')
    for bone in BONES:
        rows = [r for r in summary if r['table'] == 'hotspot_frequency' and r['bone'] == bone]
        if not rows:
            continue
        out.append('')
        out.append(f'### {bone}')
        out.append('')
        out.append('| comparison | compartment | n | subregion (count) |')
        out.append('|---|---|---|---|')
        groups: Dict[tuple, list] = {}
        for r in rows:
            groups.setdefault((r['angle'], r['load'], r['compartment'], r['n']), []).append(r)
        for (angle, load, compartment, n), members in groups.items():
            ranked = sorted(members, key=lambda r: -r['count'])
            cells = ', '.join(f'{r["region"]} {r["count"]}' for r in ranked)
            out.append(f'| {_comparison_name(angle, load)} | {compartment} | {n} | {cells} |')
    out.append('')

    # --- registration
    out.append('## Registration residual (bone surface, after rigid ICP)')
    out.append('')
    out.append('Over every registered comparison (flagged ones included).')
    out.append('')
    out.append('| bone | n | median RMS (mm) | p90 | max | flagged | not converged |')
    out.append('|---|---|---|---|---|---|---|')
    for r in cohort['registration']:
        out.append(f'| {r["bone"]} | {r["n"]} | {_md(r["median_rms"])} | {_md(r["p90_rms"])} '
                   f'| {_md(r["max_rms"])} | {r["n_flagged"]} | {r["n_not_converged"]} |')
    out.append('')

    # --- absences
    out.append('## Comparisons left out')
    sections = [('Missing (no scan for the configuration)', cohort['missing'], False),
                ('Failed (see `failures.csv`)', cohort['failed'], False),
                ('QC-flagged' + (' (kept, `--include-flagged`)' if args.include_flagged
                                 else ' (excluded)'), cohort['flagged'], True),
                ('Excluded by `--exclude`', cohort['excluded'], False)]
    for title, rows, flags in sections:
        out.append('')
        out.append(f'### {title}')
        out.append('')
        lines = _grouped(rows, keep_flags=flags)
        out.extend(f'- {line}' for line in lines) if lines else out.append('None.')
    if cohort['failures']:
        out.append('')
        out.append('### Failure records')
        out.append('')
        for f in cohort['failures']:
            out.append(f'- {f["knee"]} {_comparison_name(f["angle"], f["load"])} {f["bone"]} '
                       f'[{f["stage"]}]: {f["error"]}')
    out.append('')
    path.write_text('\n'.join(out))
    log.info('wrote %s', path)


def write_meta(path: Path, args: argparse.Namespace, context: dict, cohort: dict):
    """
    Record which code and which inputs produced the cohort outputs.

    :param path: The meta JSON file to write.
    :param args: The parsed command line.
    :param context: The :func:`per_knee_stage` context (paths, knees, records).
    :param cohort: The :func:`cohort_stage` dict (counts and lists).
    """
    meta = {
        'commit': _git('rev-parse', 'HEAD'),
        'dirty': _git('status', '--porcelain', '--untracked-files=no') != '',
        'branch': _git('rev-parse', '--abbrev-ref', 'HEAD'),
        'date': datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'),
        'host': socket.gethostname(),
        'args': {k: (str(v) if isinstance(v, Path) else v) for k, v in sorted(vars(args).items())},
        'inputs': [str(p) for p in args.input],
        'n_scans_found': len(context['paths']),
        'n_scans_measured': len(context['records']),
        'n_knees': cohort['n_knees'],
        'knees': list(context['knees']),
        'n_subregion_rows_selected': len(cohort['subregion_rows']),
        'n_hotspot_rows_selected': len(cohort['hotspot_rows']),
        'knees_per_comparison': {
            f'{a}_{l}': {bone: len({r['knee'] for r in cohort['subregion_rows']
                                    if (r['angle'], r['load'], r['bone']) == (a, l, bone)})
                         for bone in BONES}
            for a, l in COMPARISONS},
        'missing': _grouped(cohort['missing']),
        'failed': _grouped(cohort['failed']),
        'flagged': _grouped(cohort['flagged'], keep_flags=True),
        'hotspot_edge_margin_mm': args.edge_margin_mm,
        'excluded_by_name': sorted(args.exclude),
        'laterality_table': str(args.laterality_table) if args.laterality_table else None,
        'missing_laterality': cohort['missing_laterality'],
    }
    path.write_text(json.dumps(meta, indent=2) + '\n')
    log.info('wrote %s', path)


def cohort_stage(args: argparse.Namespace, context: dict,
                 laterality: Dict[str, str]) -> dict:
    """
    Pool the per-knee rows into cohort statistics and write the cohort outputs.

    Selection (:func:`select_rows`), then laterality (:func:`to_anatomical`), applied to the
    subregion and hotspot rows alike; then :func:`subregion_summary` and
    :func:`hotspot_frequency` into ``cohort_summary.csv``, the hotspot scatter per bone
    (``<bone>_hotspots_cohort.png``), ``report.md`` and ``report.meta.json``.

    :param args: The parsed command line (``output``, ``include_flagged``, ``exclude``,
        ``laterality_table``, ``input``).
    :param context: The :func:`per_knee_stage` context.
    :param laterality: ``{knee: 'left' | 'right'}`` (:func:`load_laterality_table`); empty =
        no table, pool by patient-side key.
    :return: ``{'summary', 'subregion_rows', 'hotspot_rows', 'registration', 'missing',
        'failed', 'flagged', 'excluded', 'failures', 'missing_laterality', 'laterality',
        'n_knees', 'n_scans'}``: the summary rows, the selected (and mapped) rows, the
        registration summary, the per-bone rows of the absent comparisons (subregion rows
        de-duplicated per knee x comparison x bone), the failure records and the knees missing
        from the laterality table.
    """
    results = context['results']

    def collect(kind):
        return [row for result in results.values() for row in result.get(kind, [])]

    subregion_rows, hotspot_rows = collect('subregion_rows'), collect('hotspot_rows')
    # One representative row per knee x comparison x bone, for the absence lists.
    per_bone = list({(r['knee'], r['angle'], r['load'], r['bone']): r
                     for r in subregion_rows}.values())
    excluded_names = set(args.exclude or ())
    cohort = {
        'missing': [r for r in per_bone if r['status'] == 'missing'],
        'failed': [r for r in per_bone if r['status'] == 'failed'],
        'flagged': [r for r in per_bone if r['status'] == 'ok' and r['qc_flags']],
        'excluded': [r for r in per_bone if r['status'] == 'ok' and (
            _scan_name(r['knee'], r['angle'], r['load']) in excluded_names
            or _scan_name(r['knee'], *REFERENCE) in excluded_names)],
        'failures': collect('failures'),
        'missing_laterality': missing_laterality(subregion_rows, laterality),
        'laterality': dict(laterality),
        'n_knees': len(results),
        'n_scans': len(context['paths']),
    }
    if cohort['missing_laterality']:
        log.warning('knees missing from the laterality table (left out of the statistics): %s',
                    ' '.join(cohort['missing_laterality']))

    cohort['subregion_rows'] = to_anatomical(
        select_rows(subregion_rows, args.include_flagged, args.exclude), laterality)
    cohort['hotspot_rows'] = to_anatomical(
        select_rows(hotspot_rows, args.include_flagged, args.exclude), laterality)
    cohort['summary'] = (subregion_summary(cohort['subregion_rows'])
                         + hotspot_frequency(cohort['hotspot_rows']))
    cohort['registration'] = registration_summary(collect('registration_rows'))

    out_dir = Path(args.output)
    write_csv(cohort['summary'], SUMMARY_COLUMNS, out_dir / DEFAULT_SUMMARY_CSV)
    import matplotlib.pyplot as plt
    for bone in BONES:
        try:
            figure = plot_cohort_hotspots(cohort['hotspot_rows'], bone,
                                          path=str(out_dir / f'{bone}_hotspots_cohort.png'),
                                          anatomical=bool(laterality))
            plt.close(figure)
        except Exception as error:  # a figure never costs the tables or the report
            log.error('%s cohort hotspot figure failed: %s: %s', bone, type(error).__name__,
                      error)
            plt.close('all')
    write_report(out_dir / DEFAULT_REPORT, args, cohort)
    write_meta(out_dir / DEFAULT_META, args, context, cohort)
    return cohort


# ----------------------------------------------------------------------------------------
# Command line
# ----------------------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """
    The command-line parser.

    :return: An ``argparse.ArgumentParser``.
    """
    parser = argparse.ArgumentParser(
        description='Knee cartilage deformation against unloaded 30 degrees (dfg_knees).',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-i', '--input', required=True, nargs='+',
                        help='segmentation NIfTI files and/or directories (searched '
                             'recursively); all configurations of a knee must come from one '
                             'label set')
    parser.add_argument('-o', '--output', required=True, type=Path,
                        help='output directory (CSVs, figures, cache/)')
    parser.add_argument('--workers', type=int, default=None,
                        help='worker processes (default: all CPUs; 1 = in-process)')
    parser.add_argument('--knees', nargs='+', default=None, metavar='KNEE',
                        help='only these knees (e.g. K1 K3)')
    parser.add_argument('--include-flagged', action='store_true',
                        help='cohort stage: keep QC-flagged comparisons in the statistics')
    parser.add_argument('--exclude', nargs='+', default=[], metavar='SCAN',
                        help='cohort stage: scans whose comparisons leave the statistics')
    parser.add_argument('--laterality-table', type=Path, default=None, metavar='CSV',
                        help='cohort stage: CSV(subject,laterality) for medial/lateral names')
    parser.add_argument('--grid-mm', type=float, default=1.0,
                        help='delta-map cell size in mm (default: 1.0)')
    parser.add_argument('--neighbourhood-mm', type=float, default=3.0,
                        help='hotspot smoothing neighbourhood in mm (default: 3.0)')
    parser.add_argument('--edge-margin-mm', type=float, default=DEFAULT_HOTSPOT_EDGE_MARGIN_MM,
                        help='minimum distance (mm) of a hotspot cell from each scan\'s own map '
                             'footprint edge, against the near-zero rim band and the edge '
                             'mismatch between acquisitions; 0 disables (default: %(default)s)')
    parser.add_argument('--max-registration-rms', type=float, default=REGISTRATION_MAX_RMS_MM,
                        help='registration residual (mm) above which a comparison is flagged '
                             f'(default: {REGISTRATION_MAX_RMS_MM})')
    parser.add_argument('--no-cache', action='store_true',
                        help='neither read nor write <output>/cache')
    parser.add_argument('--femur-label', type=int, default=3)
    parser.add_argument('--tibia-label', type=int, default=4)
    parser.add_argument('--femur-bone-label', type=int, default=1)
    parser.add_argument('--tibia-bone-label', type=int, default=2)
    parser.add_argument('-v', '--verbose', action='store_true', help='debug logging')
    return parser


def per_knee_stage(args: argparse.Namespace,
                   laterality: Optional[Dict[str, str]] = None) -> dict:
    """
    Measure the scans, process every knee and write the per-knee outputs.

    :param args: Parsed command line (:func:`build_parser`).
    :param laterality: ``{knee: 'left' | 'right'}`` from ``--laterality-table``; fills the
        ``anatomical_subregion`` column of ``subregion_delta.csv`` / ``hotspots.csv`` (the
        ``subregion`` column stays patient-side). ``None`` / empty leaves it empty.
    :return: ``{'paths', 'knees', 'records', 'results', 'provenance'}`` -- the selected knees'
        ``{knee: {(angle, load): path}}``, the scan records by path, the
        :func:`process_knee` results by knee and the checkout provenance, for the cohort
        stage.
    """
    paths = resolve_inputs(args.input)
    knees = discover_knees(paths)
    if args.knees:
        unknown = sorted(set(args.knees) - set(knees))
        if unknown:
            log.warning('knees not found in the input: %s', ' '.join(unknown))
        knees = {knee: configs for knee, configs in knees.items() if knee in args.knees}
    if not knees:
        log.error('no knees to process')
        sys.exit(1)
    for knee, configs in knees.items():
        absent = [k for k in (REFERENCE,) + tuple(COMPARISONS) if k not in configs]
        if absent:
            log.info('%s: no scan for %s', knee, ', '.join(f'{a} {l}' for a, l in absent))
    args.output.mkdir(parents=True, exist_ok=True)

    prov = provenance()
    if prov['dirty']:
        log.warning('dirty checkout (%s): cached measurements are neither reused nor trusted',
                    prov['commit'])
    labels = {'femur_label': args.femur_label, 'tibia_label': args.tibia_label,
              'femur_bone_label': args.femur_bone_label,
              'tibia_bone_label': args.tibia_bone_label}
    scan_paths = [p for configs in knees.values() for p in configs.values()]
    records = measure_scans(scan_paths, args.output, args.workers, not args.no_cache, prov,
                            labels)
    knee_measurements = {knee: {key: records[path] for key, path in configs.items()}
                         for knee, configs in knees.items()}
    results = process_knees(knee_measurements, args.output, args.workers, args.grid_mm,
                            args.neighbourhood_mm, args.max_registration_rms,
                            args.edge_margin_mm)

    def collect(kind):
        return [row for result in results.values() for row in result[kind]]

    fill_anatomical_subregion(collect('subregion_rows'), laterality or {})
    fill_anatomical_subregion(collect('hotspot_rows'), laterality or {})
    write_csv(collect('subregion_rows'), SUBREGION_COLUMNS, args.output / DEFAULT_SUBREGION_CSV)
    write_csv(collect('hotspot_rows'), HOTSPOT_COLUMNS, args.output / DEFAULT_HOTSPOT_CSV)
    write_csv(collect('registration_rows'), REGISTRATION_COLUMNS,
              args.output / DEFAULT_REGISTRATION_CSV)
    write_csv(collect('failures'), FAILURE_COLUMNS, args.output / DEFAULT_FAILURE_CSV)
    log.info('wrote the per-knee outputs of %d knees to %s (%d failures)', len(results),
             args.output, len(collect('failures')))
    return {'paths': paths, 'knees': knees, 'records': records, 'results': results,
            'provenance': prov}


def main(argv=None):
    """
    Run the per-knee stage, then the cohort stage.

    The laterality table is read first, so a malformed one fails before any scan is measured.

    :param argv: Argument list, defaulting to ``sys.argv[1:]``.
    :return: The per-knee stage's context dict (see :func:`per_knee_stage`) with the
        :func:`cohort_stage` dict under ``'cohort'``.
    """
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s', stream=sys.stdout)
    warnings.simplefilter('ignore')
    laterality = load_laterality_table(args.laterality_table)
    context = per_knee_stage(args, laterality)
    context['cohort'] = cohort_stage(args, context, laterality)
    return context


if __name__ == '__main__':
    main()
