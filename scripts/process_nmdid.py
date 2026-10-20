"""Batch morphometry over the NMDID whole-leg CT cohort.

Reads whole-body CT bone segmentations (femur=1, tibia=2, fibula=3, talus=4,
patella=5, hip/pelvis=7) and writes one CSV row per (case, image side) with:

    femoral torsion (Murphy), tibial torsion, CCD angle, femur length,
    tibia length, femoral offset, mechanical axis deviation,
    hip-knee-ankle angle, center-edge angle, acetabular version

"left" / "right" refer to the *image* side, not the patient side -- the mask is
split at ``array.shape[0] // 2`` as everywhere else in this package.

Designed for a SLURM cluster: ``--workers`` fans a case out over the CPUs of one
node, and ``--shard i/n`` (or ``SLURM_ARRAY_TASK_ID`` / ``SLURM_ARRAY_TASK_COUNT``)
splits the cohort across the tasks of a job array. Every shard writes its own CSV;
``--merge`` concatenates them afterwards. Runs are resumable: a case whose row is
already in the shard CSV is skipped unless ``--overwrite`` is given.

Examples
--------
    # single node, all cores
    python scripts/process_nmdid.py -i /hpcwork/.../nmdid/segmentations -o results/

    # one task of a 16-way job array
    python scripts/process_nmdid.py -i .../segmentations -o results/ --shard 3/16

    # stitch the array's shards into one table
    python scripts/process_nmdid.py -o results/ --merge
"""
import argparse
import csv
import logging
import os
import signal
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import label as connected_components

sys.path.append(str(Path(__file__).resolve().parents[1]))

from morphometry.femur import get_murphy_neck_qc_ct
from morphometry.image_io import Segmentation, split_ct_image, split_image_left_right
from morphometry.measurements.femur import calculate_femoral_torsion_ct
from morphometry.measurements.hip import (calculate_acetabular_anteversion, calculate_ccd_ct,
                                          calculate_center_edge_angle, calculate_femoral_offset_projected)
from morphometry.measurements.tibia import calculate_tibial_torsion
from morphometry.measurements.whole_leg import (calculate_bone_length_ct, calculate_hip_knee_ankle_angle,
                                                calculate_mechanical_axis_deviation)

# The NMDID label map.
FEMUR_LABEL = 1
TIBIA_LABEL = 2
FIBULA_LABEL = 3
PATELLA_LABEL = 5
PELVIS_LABEL = 7

# The pelvis label covers the whole hip bone, not the acetabulum alone, so the acetabular
# articular surface is extracted by casting rays out from the femoral head centre and
# looking this many head radii outward for the first bone they meet (see
# calculate_center_edge_angle's acetabulum_radius_factor). Results are insensitive to the
# exact value between 1.2 and 1.6; the rim sits at roughly one radius by construction.
ACETABULUM_RADIUS_FACTOR = 1.4

# ``qc_flags`` carries both the segmentation-integrity flags of ``qc_flags()`` (a label in several
# pieces, a missing bone) and the Murphy femoral-neck flags of ``murphy_qc()``; a flagged case is
# measured and written as usual.
FIELDNAMES = ['case', 'side', 'femoral_torsion', 'tibial_torsion', 'ccd', 'ccd_projected',
              'femur_length', 'tibia_length', 'femoral_offset', 'mechanical_axis_deviation',
              'hip_knee_ankle_angle', 'center_edge_angle', 'acetabular_version',
              'status', 'qc_flags', 'failed', 'seconds']

# Measurement code raises these on anatomically implausible masks; one bad case
# must not abort a cohort run, so they are caught per measurement and stored as NaN.
MEASUREMENT_ERRORS = (RuntimeError, AssertionError, ValueError, IndexError, TypeError,
                      ZeroDivisionError, np.linalg.LinAlgError)

log = logging.getLogger('process_nmdid')


def _measure(results: dict, failures: list, key: str, fn):
    """Run one measurement, storing NaN and recording the failure if it raises.

    :param results: The per-side result dict to write into.
    :param failures: A list collecting the names of the measurements that failed.
    :param key: Result key; a tuple of keys when the measurement returns several values.
    :param fn: A zero-argument callable performing the measurement.
    :return: None.
    """
    keys = (key,) if isinstance(key, str) else key
    try:
        value = fn()
    except MEASUREMENT_ERRORS as exc:
        log.debug('%s failed: %s: %s', keys[0], type(exc).__name__, exc)
        for k in keys:
            results[k] = np.nan
            failures.append(k)
        return

    values = (value,) if len(keys) == 1 else value
    for k, v in zip(keys, values):
        results[k] = float(v)


def load_case(path: Path) -> Segmentation:
    """Load a segmentation into the LPI orientation with disconnected fragments removed.

    :param path: Path to the ``.nii.gz`` segmentation mask.
    :return: The prepared Segmentation.
    """
    seg = Segmentation('nibabel')
    seg.read_image(str(path))
    seg.transform_coordinate_system()
    seg.remove_outliers()
    return seg


LABEL_NAMES = {FEMUR_LABEL: 'femur', TIBIA_LABEL: 'tibia', FIBULA_LABEL: 'fibula',
               PATELLA_LABEL: 'patella', PELVIS_LABEL: 'pelvis'}


def _components(mask: np.ndarray) -> list:
    """List the 26-connected components of a binary mask, largest first.

    :param mask: A binary 3D mask.
    :return: A list of ``(size, z_min, z_max, component_index)`` tuples, largest first.
    """
    components, count = connected_components(mask, np.ones((3, 3, 3), dtype=bool))
    out = []
    for index in range(1, count + 1):
        voxels = np.argwhere(components == index)
        out.append((len(voxels), int(voxels[:, 2].min()), int(voxels[:, 2].max()), index))
    out.sort(reverse=True)
    return out, components


def clean_leg(array: np.ndarray) -> np.ndarray:
    """Drop the anatomically impossible label components from one image half.

    These are whole-*body* scans, so the arm bones are in the field of view and are
    regularly mislabelled as femur or tibia; the spine and ribs likewise leak into the
    pelvis label. Two rules remove them without touching real bone:

    * the pelvis keeps only its largest component -- one hip bone per image half;
    * every other label drops the components lying entirely superior to that hip bone,
      since no leg bone can sit above the pelvis.

    Deliberately conservative: a femur that the segmentation broke into a proximal and a
    distal fragment keeps both fragments (they are not above the pelvis), so a genuinely
    damaged segmentation stays visible in the numbers and in ``qc_flags`` rather than
    being silently truncated to one fragment.
    :param array: A single-leg label array.
    :return: The cleaned label array.
    """
    cleaned = array.copy()

    pelvis_sizes, pelvis_components = _components(array == PELVIS_LABEL)
    if not pelvis_sizes:
        return cleaned
    _, pelvis_top, _, largest = pelvis_sizes[0]
    cleaned[(array == PELVIS_LABEL) & (pelvis_components != largest)] = 0

    for value in np.unique(array):
        if value in (0, PELVIS_LABEL):
            continue
        sizes, components = _components(array == value)
        for _, _, z_max, index in sizes:
            if z_max < pelvis_top:  # LPI: smaller z is more superior
                cleaned[components == index] = 0
    return cleaned


def qc_flags(array: np.ndarray) -> str:
    """Describe what still looks wrong with a cleaned single-leg mask.

    A flagged leg is not necessarily unusable, but its measurements should not be
    trusted without looking at the mask: a label in several pieces means the bone was
    broken up or, as happens in this cohort, that part of one bone carries another
    bone's label.
    :param array: A cleaned single-leg label array.
    :return: A semicolon-separated flag string, empty when the leg looks intact.
    """
    flags = []
    for value, name in sorted(LABEL_NAMES.items()):
        mask = array == value
        if not mask.any():
            flags.append(f'{name}:missing')
            continue
        sizes, _ = _components(mask)
        # ignore specks: only fragments worth a tenth of the bone indicate a real break
        significant = [s for s, *_ in sizes if s >= 0.1 * sizes[0][0]]
        if len(significant) > 1:
            flags.append(f'{name}:{len(significant)}pieces')
    return ';'.join(flags)


def murphy_qc(leg: Segmentation, side: str) -> str:
    """QC-flag the Murphy femoral-neck landmarks of one leg, never raising.

    Appended to this row's ``qc_flags``. A flag marks the femoral torsion of that row as
    untrustworthy (the Murphy neck base is degenerate); it never changes the measured value, and
    an unflagged row is bit-identical to one produced before the flags existed.
    ``get_murphy_neck_qc_ct`` already reports a failed landmark search as a flag rather than
    raising, so reaching the handler here means the QC itself is broken. That is logged at WARNING,
    not debug: an empty ``qc_flags`` column must not be able to mean "the QC never ran".
    :param leg: One cleaned single-leg Segmentation.
    :param side: The image side, 'left' or 'right'.
    :return: The ``;``-separated flag string, empty when the neck looks sound.
    """
    try:
        return get_murphy_neck_qc_ct(leg, side=side, segmentation_label=FEMUR_LABEL)
    except Exception as exc:  # QC must never cost a row
        log.warning('Murphy neck QC itself failed (%s side): %s: %s -- qc_flags for this row is '
                    'empty because the QC did not run, not because the neck is sound',
                    side, type(exc).__name__, exc)
        return ''


def split_legs(seg: Segmentation) -> tuple:
    """Split a both-legs mask into cleaned left / right image halves and their union.

    The mask is split at ``array.shape[0] // 2`` (the package-wide convention; verified
    on this cohort to fall in the gap between the two legs), each half is cleaned by
    ``clean_leg``, and the halves are re-joined so the both-sides hip measurements see
    the same voxels as the per-leg ones.
    :param seg: The prepared whole-body segmentation.
    :return: ``(halves, whole, flags)`` -- per-side Segmentations, the re-joined
    Segmentation, and the per-side QC flag strings.
    """
    # split with per-half geometry: the right half's index 0 is the whole mask's index ``half``
    raw = dict(zip(('left', 'right'), split_image_left_right(seg)))
    arrays = {side: clean_leg(half.array) for side, half in raw.items()}

    def _as_segmentation(array, geometry):
        return Segmentation.from_nibabel(nib.Nifti1Image(array, geometry.affine, geometry.header))

    halves = {side: _as_segmentation(arrays[side], raw[side]) for side in arrays}
    whole = _as_segmentation(np.concatenate([arrays['left'], arrays['right']], axis=0), seg)
    flags = {side: qc_flags(array) for side, array in arrays.items()}
    return halves, whole, flags


def measure_case(path: Path) -> list:
    """Compute every measurement for one case.

    :param path: Path to the case's segmentation mask.
    :return: Two row dicts (left and right image side) ready for the CSV writer.
    """
    case = path.name.split('.')[0].replace('_seg', '')
    started = time.time()

    seg = load_case(path)
    legs, seg, flags = split_legs(seg)

    # Both-sides measurements: the acetabular landmarks need both femoral head
    # centres, so these are computed once on the whole mask and split per side.
    both = {}
    both_failures = []
    _measure(both, both_failures, ('center_edge_angle_left', 'center_edge_angle_right'),
             lambda: calculate_center_edge_angle(seg, femur_label=FEMUR_LABEL, acetabulum_label=PELVIS_LABEL,
                                                 ct=True, project=True,
                                                 acetabulum_radius_factor=ACETABULUM_RADIUS_FACTOR))
    _measure(both, both_failures, ('acetabular_version_left', 'acetabular_version_right'),
             lambda: calculate_acetabular_anteversion(seg, femur_label=FEMUR_LABEL, acetabulum_label=PELVIS_LABEL,
                                                      ct=True,
                                                      acetabulum_radius_factor=ACETABULUM_RADIUS_FACTOR))

    rows = []
    for side in ('left', 'right'):
        leg = legs[side]
        row = {'case': case, 'side': side}
        failures = list(both_failures)

        # The hip / knee / ankle crops are shared by every measurement below that
        # needs them; splitting is expensive, so it is done once per leg.
        try:
            _, masks = split_ct_image(leg, None, femur_label=FEMUR_LABEL, hip_label=PELVIS_LABEL,
                                      patella_label=PATELLA_LABEL, fibula_label=FIBULA_LABEL)
        except MEASUREMENT_ERRORS as exc:
            log.warning('%s %s: cannot split into hip/knee/ankle: %s: %s', case, side, type(exc).__name__, exc)
            masks = None

        _measure(row, failures, ('ccd', 'ccd_projected'),
                 lambda: calculate_ccd_ct(leg, side=side, segmentation_label=FEMUR_LABEL))
        _measure(row, failures, 'femur_length', lambda: calculate_bone_length_ct(leg, FEMUR_LABEL))
        _measure(row, failures, 'tibia_length', lambda: calculate_bone_length_ct(leg, TIBIA_LABEL))
        _measure(row, failures, 'femoral_offset',
                 lambda: calculate_femoral_offset_projected(leg, side=side, femur_label=FEMUR_LABEL, ct=True))

        if masks is None:
            for key in ('femoral_torsion', 'tibial_torsion', 'mechanical_axis_deviation', 'hip_knee_ankle_angle'):
                row[key] = np.nan
                failures.append(key)
        else:
            knee, ankle = masks[1], masks[2]
            _measure(row, failures, 'femoral_torsion',
                     lambda: calculate_femoral_torsion_ct(leg, knee, side=side, segmentation_label=FEMUR_LABEL))
            _measure(row, failures, 'tibial_torsion',
                     lambda: calculate_tibial_torsion(knee.array, ankle.array, TIBIA_LABEL, TIBIA_LABEL,
                                                      FIBULA_LABEL, side=side))
            _measure(row, failures, 'mechanical_axis_deviation',
                     lambda: calculate_mechanical_axis_deviation(leg, femur_label=FEMUR_LABEL,
                                                                 tibia_label=TIBIA_LABEL, fibula_label=FIBULA_LABEL,
                                                                 patella_label=PATELLA_LABEL, hip_label=PELVIS_LABEL,
                                                                 side=side, masks=masks))
            # calculate_hip_knee_ankle_angle returns the deviation from a straight leg axis;
            # 180 - deviation is the hip-knee-ankle angle in the usual clinical convention
            # (~180 degrees straight, < 180 varus).
            _measure(row, failures, 'hip_knee_ankle_angle',
                     lambda: 180. - calculate_hip_knee_ankle_angle(leg, femur_label=FEMUR_LABEL,
                                                                   tibia_label=TIBIA_LABEL, fibula_label=FIBULA_LABEL,
                                                                   patella_label=PATELLA_LABEL, hip_label=PELVIS_LABEL,
                                                                   side=side, masks=masks))

        row['center_edge_angle'] = both.get(f'center_edge_angle_{side}', np.nan)
        row['acetabular_version'] = both.get(f'acetabular_version_{side}', np.nan)

        # Murphy's femoral-torsion axis is short, so a neck-base landmark a few millimetres out of
        # place is a large angular error that looks perfectly plausible in the number. The flags say
        # the femoral torsion of this row should not be trusted without looking at the mask; the
        # value is still measured and written (see morphometry.femur.murphy_neck_base_qc).
        row['qc_flags'] = ';'.join(f for f in (flags[side], murphy_qc(leg, side)) if f)
        row['failed'] = ';'.join(sorted(set(failures)))
        row['status'] = 'ok' if not failures else 'partial'
        row['seconds'] = round(time.time() - started, 1)
        rows.append(row)

    return rows


def _worker(path_str: str) -> list:
    """Process-pool entry point: measure one case, never raising.

    :param path_str: The case path as a string (picklable).
    :return: The case's rows, or two failure rows if loading the case blew up.
    """
    path = Path(path_str)
    warnings.simplefilter('ignore')
    try:
        return measure_case(path)
    except Exception as exc:  # a broken file must not take the pool down
        case = path.name.split('.')[0].replace('_seg', '')
        log.error('%s: %s: %s', case, type(exc).__name__, exc)
        return [{'case': case, 'side': side, 'status': 'error', 'failed': f'{type(exc).__name__}: {exc}'}
                for side in ('left', 'right')]


def discover_cases(inputs: list, cohort_status: Path = None, verdicts: set = None) -> list:
    """Collect the segmentation files to process, sorted for a deterministic sharding.

    :param inputs: Files and/or directories (searched recursively for ``*.nii`` / ``*.nii.gz``).
    :param cohort_status: Optional NMDID ``cohort_status.csv`` used to filter cases.
    :param verdicts: The verdicts to keep from that CSV (e.g. ``{'usable'}``).
    :return: A sorted list of case paths.
    """
    paths = []
    for item in inputs:
        item = Path(item)
        if item.is_dir():
            paths.extend(p for p in item.rglob('*.nii*') if p.name.endswith(('.nii', '.nii.gz')))
        elif item.exists():
            paths.append(item)
        else:
            log.warning('input not found, skipping: %s', item)

    if cohort_status is not None and verdicts:
        with open(cohort_status, newline='') as fh:
            keep = {r['case'] for r in csv.DictReader(fh) if r['verdict'] in verdicts}
        before = len(paths)
        paths = [p for p in paths if p.name.split('.')[0].replace('_seg', '') in keep]
        log.info('cohort filter %s kept %d of %d cases', sorted(verdicts), len(paths), before)

    return sorted(set(paths))


def _read_done(csv_path: Path) -> set:
    """Read the case ids already present in a shard CSV, for resuming a run.

    :param csv_path: The shard's output CSV.
    :return: The set of case ids already written.
    """
    if not csv_path.exists():
        return set()
    with open(csv_path, newline='') as fh:
        return {row['case'] for row in csv.DictReader(fh) if row.get('case')}


def _parse_shard(spec: str) -> tuple:
    """Parse a ``i/n`` shard specification into zero-based ``(index, count)``.

    :param spec: The shard specification, e.g. ``'3/16'``.
    :return: The shard index and the shard count.
    """
    index, _, count = spec.partition('/')
    index, count = int(index), int(count)
    if not 0 <= index < count:
        raise argparse.ArgumentTypeError(f'shard index {index} out of range for {count} shards')
    return index, count


def _shard_from_environment():
    """Derive the shard from the SLURM job-array variables, if this is an array task.

    :return: ``(index, count)`` or None when not running as a job array.
    """
    index = os.environ.get('SLURM_ARRAY_TASK_ID')
    count = os.environ.get('SLURM_ARRAY_TASK_COUNT')
    if index is None or count is None:
        return None
    minimum = int(os.environ.get('SLURM_ARRAY_TASK_MIN', 0))
    return int(index) - minimum, int(count)


def _default_workers() -> int:
    """The number of worker processes to use, honouring the SLURM CPU allocation.

    :return: A sensible worker count for this node.
    """
    allocated = os.environ.get('SLURM_CPUS_PER_TASK')
    if allocated:
        return max(1, int(allocated))
    return len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else (os.cpu_count() or 1)


def merge_shards(out_dir: Path, output: Path):
    """Concatenate every shard CSV in a directory into one table.

    :param out_dir: The directory holding the ``shard_*.csv`` files.
    :param output: The combined CSV to write.
    :return: The number of rows written.
    """
    shards = sorted(out_dir.glob('shard_*.csv'))
    if not shards:
        raise SystemExit(f'no shard_*.csv found in {out_dir}')

    rows = []
    for shard in shards:
        with open(shard, newline='') as fh:
            rows.extend(csv.DictReader(fh))
    rows.sort(key=lambda r: (r['case'], r['side']))

    with open(output, 'w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    log.info('merged %d shards into %s (%d rows)', len(shards), output, len(rows))
    return len(rows)


def run(paths: list, csv_path: Path, workers: int, overwrite: bool):
    """Measure every case in ``paths``, appending rows to ``csv_path`` as they finish.

    Results are flushed per case so an interrupted or pre-empted run keeps everything
    it had already computed and can be resumed by re-running the same command.
    :param paths: The case paths this shard is responsible for.
    :param csv_path: The shard's output CSV.
    :param workers: The number of worker processes.
    :param overwrite: Recompute cases already present in the CSV.
    :return: None.
    """
    done = set() if overwrite else _read_done(csv_path)
    todo = [p for p in paths if p.name.split('.')[0].replace('_seg', '') not in done]
    if done:
        log.info('resuming: %d of %d cases already done', len(done), len(paths))
    if not todo:
        log.info('nothing to do')
        return

    write_header = overwrite or not csv_path.exists()
    mode = 'w' if overwrite else 'a'
    started = time.time()

    with open(csv_path, mode, newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES, extrasaction='ignore')
        if write_header:
            writer.writeheader()
            fh.flush()

        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_worker, str(p)): p for p in todo}
            for n, future in enumerate(as_completed(futures), start=1):
                rows = future.result()
                writer.writerows(rows)
                fh.flush()
                elapsed = time.time() - started
                log.info('[%d/%d] %s %s (%.0fs elapsed, %.1f cases/min)', n, len(todo), rows[0]['case'],
                         '/'.join(r['status'] for r in rows), elapsed, 60 * n / max(elapsed, 1e-9))


def main(argv=None):
    """Parse the command line and run the batch, a merge, or a listing.

    :param argv: Argument list, defaulting to ``sys.argv[1:]``.
    :return: None.
    """
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-i', '--input', nargs='+', default=[],
                        help='segmentation files and/or directories (searched recursively)')
    parser.add_argument('-o', '--output', type=Path, required=True,
                        help='output directory for the shard CSVs (and the merged table)')
    parser.add_argument('--workers', type=int, default=None,
                        help='worker processes (default: SLURM_CPUS_PER_TASK, else all visible CPUs)')
    parser.add_argument('--shard', type=_parse_shard, default=None, metavar='I/N',
                        help='process shard I of N (default: taken from SLURM_ARRAY_TASK_ID, else all cases)')
    parser.add_argument('--cohort-status', type=Path, default=None,
                        help="NMDID cohort_status.csv to filter cases by verdict")
    parser.add_argument('--verdict', nargs='+', default=['usable', 'review'],
                        help='verdicts to keep from --cohort-status (default: usable review)')
    parser.add_argument('--limit', type=int, default=None, help='process at most this many cases (for a test run)')
    parser.add_argument('--overwrite', action='store_true', help='recompute cases already present in the shard CSV')
    parser.add_argument('--merge', action='store_true', help='concatenate the shard CSVs and exit')
    parser.add_argument('--merged-name', default='morphometry.csv', help='file name for the merged table')
    parser.add_argument('-v', '--verbose', action='store_true', help='log per-measurement failures')
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s', stream=sys.stdout)
    warnings.simplefilter('ignore')

    args.output.mkdir(parents=True, exist_ok=True)

    if args.merge:
        merge_shards(args.output, args.output / args.merged_name)
        return

    if not args.input:
        parser.error('-i/--input is required unless --merge is given')

    paths = discover_cases(args.input, args.cohort_status, set(args.verdict))
    if args.limit:
        paths = paths[:args.limit]
    if not paths:
        raise SystemExit('no segmentations found')

    shard = args.shard or _shard_from_environment()
    if shard is None:
        index, count = 0, 1
    else:
        index, count = shard
        paths = paths[index::count]  # stride, so every shard gets a comparable mix

    workers = args.workers or _default_workers()
    csv_path = args.output / (f'shard_{index:03d}.csv' if count > 1 else 'shard_000.csv')
    log.info('shard %d/%d: %d cases, %d workers -> %s', index, count, len(paths), workers, csv_path)

    # A pre-emption or a time limit arrives as SIGTERM; the rows already flushed
    # stay on disk and re-running the same command resumes from them.
    signal.signal(signal.SIGTERM, lambda *_: sys.exit('received SIGTERM, stopping'))

    run(paths, csv_path, workers, args.overwrite)
    log.info('shard %d/%d done', index, count)


if __name__ == '__main__':
    main()
