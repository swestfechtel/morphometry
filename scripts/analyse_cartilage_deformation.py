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
   largest-thinning hotspot per compartment is located on the smoothed Δ map
   (:func:`~morphometry.cartilage.deformation.find_largest_thinning`); the per-subregion Δ is
   ``aMe(config) - aMe(reference)`` with **each scan's own** subdivision.
3. *(Cohort statistics, laterality and the report are appended to* :func:`main` *after the
   per-knee stage; the per-knee stage returns everything that needs.)*

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
* ``subregion_delta.csv`` -- :data:`SUBREGION_COLUMNS`; ``anatomical_subregion`` is filled by
  the cohort stage from ``--laterality-table`` and is empty here.
* ``hotspots.csv`` -- :data:`HOTSPOT_COLUMNS`; map coordinates in the reference's map frame
  (tibia: plateau plane, patient right / anterior; femur: medial-lateral / unrolled arc), world
  coordinates = the reference scan's RAS mm position of the hotspot cell
  (``BoneDeltaMaps.reference_positions``; NaN where the reference has no covered triangle).
* ``registration.csv`` -- :data:`REGISTRATION_COLUMNS`, one row per registered
  (comparison, bone).
* ``failures.csv`` -- :data:`FAILURE_COLUMNS`, one row per failed measurement, registration
  or hotspot search.
* ``<knee>_tibia_delta.png`` / ``<knee>_femur_delta.png`` -- the 2 x 4 grid of
  :func:`~morphometry.cartilage.deformation_plot.plot_knee_deltas` (not drawn for a bone whose
  reference failed).

Example
-------
::

    python scripts/analyse_cartilage_deformation.py \\
        -i /home/simon/Downloads/dfg_knees_segmentations/dfg_knees/segmentations \\
        -o out/deformation --workers 16 --knees K1 K7
"""
import argparse
import csv
import logging
import math
import os
import pickle
import re
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

from morphometry.cartilage.deformation import (
    COMPARISONS, COMPARTMENTS, REFERENCE, bone_measurement, find_largest_thinning,
    knee_delta_maps,
)
from morphometry.cartilage.femur_regions import FEMUR_REGION_KEYS, ZONE_THICKNESS_KEYS
from morphometry.cartilage.registration import REGISTRATION_MAX_RMS_MM
from morphometry.cartilage.tibia_regions import TIBIA_REGION_KEYS

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
                   'world_x', 'world_y', 'world_z', 'subregion', 'smoothed_delta_mm',
                   'delta_mm', 'kernel_size', 'fallback', 'status', 'qc_flags']
REGISTRATION_COLUMNS = ['knee', 'angle', 'load', 'bone', 'rms_mm', 'median_mm', 'p95_mm',
                        'n_points', 'iterations', 'converged', 'rotation_change_deg',
                        'dropped_fraction', 'flagged']
FAILURE_COLUMNS = ['knee', 'angle', 'load', 'bone', 'stage', 'error']

DEFAULT_SUBREGION_CSV = 'subregion_delta.csv'
DEFAULT_HOTSPOT_CSV = 'hotspots.csv'
DEFAULT_REGISTRATION_CSV = 'registration.csv'
DEFAULT_FAILURE_CSV = 'failures.csv'

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
    :return: Dict of the ``map_*``, ``world_*``, ``subregion``, delta, ``kernel_size`` and
        ``fallback`` columns (NaN / empty when ``hotspot`` is ``None``).
    """
    if hotspot is None:
        return {'map_x_mm': NAN, 'map_y_mm': NAN, 'world_x': NAN, 'world_y': NAN,
                'world_z': NAN, 'subregion': '', 'smoothed_delta_mm': NAN, 'delta_mm': NAN,
                'kernel_size': NAN, 'fallback': ''}
    i, j = hotspot['cell']
    world = np.asarray(maps.reference_positions[i, j], dtype=float)
    return {'map_x_mm': hotspot['map_mm'][0], 'map_y_mm': hotspot['map_mm'][1],
            'world_x': float(world[0]), 'world_y': float(world[1]), 'world_z': float(world[2]),
            'subregion': hotspot['subregion'], 'smoothed_delta_mm': hotspot['smoothed_delta_mm'],
            'delta_mm': hotspot['delta_mm'], 'kernel_size': hotspot['kernel_size'],
            'fallback': bool(hotspot['fallback'])}


def process_knee(knee: str, measurements: Dict[tuple, dict], grid_mm: float = 1.0,
                 neighbourhood_mm: float = 3.0,
                 max_rms_mm: float = REGISTRATION_MAX_RMS_MM) -> dict:
    """
    Compare one knee's configurations against its reference, per bone.

    :param knee: Knee id (``K<n>``).
    :param measurements: ``{(angle, load): record}`` with the :func:`measure_scan` records of
        the knee's scans; a configuration without a file is simply absent.
    :param grid_mm: Δ-map cell size (mm).
    :param neighbourhood_mm: Smoothing neighbourhood of the hotspot search (mm).
    :param max_rms_mm: Registration residual above which a comparison is flagged (mm).
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
        if ref_m is not None:
            try:
                maps = knee_delta_maps(bone, ref_m, configurations, grid_mm=grid_mm)
            except CAUGHT as error:
                ref_error = f'reference: {type(error).__name__}: {error}'
        if ref_error:
            failure(REFERENCE, bone, 'reference', ref_error)
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
                                                    neighbourhood_mm=neighbourhood_mm)
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
    :return: The paths written (a bone whose reference failed has no map and no figure).
    """
    import matplotlib.pyplot as plt
    from morphometry.cartilage.deformation_plot import plot_knee_deltas

    written = []
    for bone in BONES:
        maps = result['maps'].get(bone)
        if maps is None:
            log.warning('%s %s: no delta maps (reference failed); no figure', knee, bone)
            continue
        path = Path(out_dir) / f'{knee}_{bone}_delta.png'
        figure = plot_knee_deltas(maps, knee, hotspots=result['hotspots'][bone],
                                  panel_titles=result['panel_titles'][bone], path=str(path))
        plt.close(figure)
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

    An exception :func:`process_knee` does not catch (or a figure failing) does not lose the
    knee's rows: the knee is re-processed with every scan failed, so every row is written with
    status ``failed`` (``missing`` for absent files) and the error in ``failures.csv``.

    :param arguments: ``(knee, measurements, grid_mm, neighbourhood_mm, max_rms_mm, out_dir)``.
    :return: ``(knee, result)``.
    """
    warnings.simplefilter('ignore')
    knee, measurements, grid_mm, neighbourhood_mm, max_rms_mm, out_dir = arguments
    try:
        result = process_knee(knee, measurements, grid_mm, neighbourhood_mm, max_rms_mm)
        write_knee_figures(knee, result, out_dir)
    except Exception as error:
        message = f'knee job: {type(error).__name__}: {error}'
        log.error('%s: %s', knee, message)
        result = process_knee(knee, {k: failed_record(r, message)
                                     for k, r in measurements.items()},
                              grid_mm, neighbourhood_mm, max_rms_mm)
    return knee, result


def process_knees(knees: Dict[str, Dict[tuple, dict]], out_dir: Path, workers: Optional[int],
                  grid_mm: float, neighbourhood_mm: float,
                  max_rms_mm: float) -> Dict[str, dict]:
    """
    Run :func:`_knee_job` for every knee, in a process pool.

    :param knees: ``{knee: {(angle, load): record}}``.
    :param out_dir: Output directory (figures).
    :param workers: Worker processes; ``None`` = all CPUs, ``1`` = in-process.
    :param grid_mm: Δ-map cell size (mm).
    :param neighbourhood_mm: Hotspot smoothing neighbourhood (mm).
    :param max_rms_mm: Registration residual threshold (mm).
    :return: ``{knee: process_knee result}`` in the order of ``knees``.
    """
    jobs = [(knee, measurements, grid_mm, neighbourhood_mm, max_rms_mm, out_dir)
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
                                          grid_mm, neighbourhood_mm, max_rms_mm)
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


def per_knee_stage(args: argparse.Namespace) -> dict:
    """
    Measure the scans, process every knee and write the per-knee outputs.

    :param args: Parsed command line (:func:`build_parser`).
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
                            args.neighbourhood_mm, args.max_registration_rms)

    def collect(kind):
        return [row for result in results.values() for row in result[kind]]

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
    Run the per-knee stage (and, once added, the cohort stage).

    :param argv: Argument list, defaulting to ``sys.argv[1:]``.
    :return: The per-knee stage's context dict (see :func:`per_knee_stage`).
    """
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s', stream=sys.stdout)
    warnings.simplefilter('ignore')
    context = per_knee_stage(args)
    # The cohort stage (statistics, laterality, report) is appended here.
    return context


if __name__ == '__main__':
    main()
