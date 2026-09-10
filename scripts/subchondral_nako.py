"""Per-ray subchondral femoral-head-to-acetabulum distances over the NaKo hip MRI cohort.

Unlike the cohort scripts that reduce each hip to summary statistics, this one keeps
every traced ray, because the distribution over the acetabular dome is the point: a
mean of 4 mm taken over 190 rays and the same mean taken over 60 surviving rays mean
different things. Two tables are written per shard:

``summary_shard_XXX.csv``
    One row per (case, image side): the femoral head centre, the surviving-ray count,
    and mean / std / min / max.
``rays_shard_XXX.csv.gz``
    One row per surviving ray: its distance and the femoral-head exit and acetabular
    hit points. Gzipped -- the full cohort is roughly 11 million rays.

Coordinates
-----------
Every ``*_i/_j/_k`` column is a **voxel index into the LPI-reoriented full-image
array** -- the frame the measurement runs in, obtained by loading the mask and calling
``transform_coordinate_system()``. It is *not* an index into the file as stored on
disk (this cohort is stored L/I/P), so reorient before overlaying. Every ``*_x/_y/_z``
column is the physical (world, mm) coordinate of the same point, which is unambiguous
and needs no reorientation.

Both are given in the frame of the **whole** image, not the image half the measurement
runs on. The halves are cut at ``array.shape[0] // 2`` and the right half is given a
correspondingly shifted affine, without which every right-hip coordinate would be off
by half the sagittal extent (192 mm on this cohort). Distances are unaffected either
way, being differences.

Designed for a SLURM cluster: ``--workers`` fans a shard out over the CPUs of one node,
and ``--shard i/n`` (or ``SLURM_ARRAY_TASK_ID`` / ``SLURM_ARRAY_TASK_COUNT``) splits the
cohort across the tasks of a job array. Runs are resumable: a case already in the shard
summary is skipped unless ``--overwrite`` is given.

Examples
--------
    # single node, all cores
    python scripts/subchondral_nako.py -i /hpcwork/.../nako -o results/

    # one task of an 8-way job array
    python scripts/subchondral_nako.py -i /hpcwork/.../nako -o results/ --shard 3/8

    # stitch the array's shards into one pair of tables
    python scripts/subchondral_nako.py -o results/ --merge
"""
import argparse
import csv
import gzip
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

sys.path.append(str(Path(__file__).resolve().parents[1]))

from morphometry.hip import get_femoral_head_center
from morphometry.image_io import Segmentation
# The public `calculate_subchondral_distance_ray_tracing` resolves the femoral head centre
# internally and does not hand it back, but we have to report it -- and re-deriving it costs
# 0.55 s against a 0.74 s measurement, near-doubling a 61k-hip sweep. The public wrapper is a
# five-line passthrough to this core, so we resolve the landmark once and call the core with
# it. Same code path, same numbers (asserted in tests/unit/test_subchondral_nako_script.py).
from morphometry.measurements.hip import _subchondral_distance_ray_tracing

# The NaKo hip MRI label map (isotropic 1 mm; both hips in one mask).
FEMUR_LABEL = 1
CARTILAGE_LABEL = 2
ACETABULUM_LABEL = 3

N_RAYS = 200
CONE_ANGLE = 45.0

SUMMARY_FIELDS = ['case', 'side', 'status',
                  'fhc_i', 'fhc_j', 'fhc_k', 'fhc_x', 'fhc_y', 'fhc_z', 'fhc_radius_vox',
                  'n_rays_cast', 'n_rays_used', 'ray_fraction',
                  'mean_mm', 'std_mm', 'min_mm', 'max_mm', 'error', 'seconds']

RAY_FIELDS = ['case', 'side', 'ray', 'distance_mm',
              'exit_i', 'exit_j', 'exit_k', 'hit_i', 'hit_j', 'hit_k',
              'exit_x', 'exit_y', 'exit_z', 'hit_x', 'hit_y', 'hit_z']

# The measurement raises these on hips it cannot resolve; one bad case must not abort a
# cohort run, so they are caught per hip and recorded as a failed row.
MEASUREMENT_ERRORS = (RuntimeError, AssertionError, ValueError, IndexError, TypeError,
                      ZeroDivisionError, np.linalg.LinAlgError)

log = logging.getLogger('subchondral_nako')


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


def image_half(seg: Segmentation, side: str) -> Segmentation:
    """Extract one image half, keeping its geometry tied to the whole image.

    The halves are cut at ``shape[0] // 2``. The right half starts at that offset, so its
    affine translation is advanced by the same amount: without it the half would inherit
    the parent's affine unchanged and every physical coordinate it reports would be off by
    half the sagittal extent. Distances are differences and so are identical either way --
    it is the reported points that need this.
    :param seg: The whole-image segmentation, already in LPI.
    :param side: The image side, 'left' or 'right'.
    :return: A Segmentation of that half in the parent's physical frame.
    """
    half = seg.array.shape[0] // 2
    affine = seg.affine.copy()
    if side == 'right':
        affine[:3, 3] = affine[:3, :3].dot(np.array([half, 0, 0])) + affine[:3, 3]
        array = seg.array[half:]
    else:
        array = seg.array[:half]
    return Segmentation.from_nibabel(nib.Nifti1Image(array, affine, seg.header))


def measure_case(path: Path) -> tuple:
    """Measure both hips of one case.

    :param path: Path to the case's segmentation mask.
    :return: ``(summary_rows, ray_rows)`` ready for the CSV writers.
    """
    case = path.name.split('.')[0]
    started = time.time()

    seg = load_case(path)
    half = seg.array.shape[0] // 2

    summaries, rays = [], []
    for side in ('left', 'right'):
        hip = image_half(seg, side)
        row = {'case': case, 'side': side, 'n_rays_cast': N_RAYS}
        try:
            radius, fhc = get_femoral_head_center(hip.array, side=side,
                                                  segmentation_label=FEMUR_LABEL, isotropic=True)
            # min_ray_fraction=0 disables the library's heavy-ray-loss WARNING: across
            # ~61k hips it would flood the job log, and n_rays_used / ray_fraction below
            # carry the same signal per row, exactly rather than as a threshold.
            mean, std, low, high, distances, exits, hits = _subchondral_distance_ray_tracing(
                hip, fhc, radius,
                np.where(hip.array == FEMUR_LABEL, 1, 0),
                np.where(hip.array == ACETABULUM_LABEL, 1, 0),
                n_rays=N_RAYS, cone_angle=CONE_ANGLE, min_ray_fraction=0.)
        except MEASUREMENT_ERRORS as exc:
            log.debug('%s %s: %s: %s', case, side, type(exc).__name__, exc)
            row.update({'status': 'error', 'error': f'{type(exc).__name__}: {exc}',
                        'n_rays_used': 0, 'seconds': round(time.time() - started, 1)})
            summaries.append(row)
            continue

        # The head centre is resolved on the half's array, so its sagittal index is relative
        # to the half; shift it into the whole-image frame.
        fhc_whole = np.asarray(fhc, dtype=float) + np.array([half if side == 'right' else 0, 0, 0])
        fhc_phys = hip.transform_index_to_physical_point(np.asarray(fhc, dtype=float))

        row.update({'status': 'ok',
                    'fhc_i': round(float(fhc_whole[0]), 3), 'fhc_j': round(float(fhc_whole[1]), 3),
                    'fhc_k': round(float(fhc_whole[2]), 3),
                    'fhc_x': round(float(fhc_phys[0]), 3), 'fhc_y': round(float(fhc_phys[1]), 3),
                    'fhc_z': round(float(fhc_phys[2]), 3),
                    'fhc_radius_vox': round(float(radius), 3),
                    'n_rays_used': len(distances),
                    'ray_fraction': round(len(distances) / N_RAYS, 4),
                    'mean_mm': round(mean, 4), 'std_mm': round(std, 4),
                    'min_mm': round(low, 4), 'max_mm': round(high, 4),
                    'seconds': round(time.time() - started, 1)})
        summaries.append(row)

        offset = np.array([half if side == 'right' else 0, 0, 0])
        for index, (distance, exit_phys, hit_phys) in enumerate(zip(distances, exits, hits)):
            exit_idx = np.asarray(hip.transform_physical_point_to_index(exit_phys)) + offset
            hit_idx = np.asarray(hip.transform_physical_point_to_index(hit_phys)) + offset
            rays.append({'case': case, 'side': side, 'ray': index,
                         'distance_mm': round(float(distance), 4),
                         'exit_i': round(float(exit_idx[0]), 3), 'exit_j': round(float(exit_idx[1]), 3),
                         'exit_k': round(float(exit_idx[2]), 3),
                         'hit_i': round(float(hit_idx[0]), 3), 'hit_j': round(float(hit_idx[1]), 3),
                         'hit_k': round(float(hit_idx[2]), 3),
                         'exit_x': round(float(exit_phys[0]), 3), 'exit_y': round(float(exit_phys[1]), 3),
                         'exit_z': round(float(exit_phys[2]), 3),
                         'hit_x': round(float(hit_phys[0]), 3), 'hit_y': round(float(hit_phys[1]), 3),
                         'hit_z': round(float(hit_phys[2]), 3)})

    return summaries, rays


def _worker(path_str: str) -> tuple:
    """Process-pool entry point: measure one case, never raising.

    :param path_str: The case path as a string (picklable).
    :return: ``(summary_rows, ray_rows)``, or two failure summaries if loading blew up.
    """
    path = Path(path_str)
    warnings.simplefilter('ignore')
    try:
        return measure_case(path)
    except Exception as exc:  # a broken file must not take the pool down
        case = path.name.split('.')[0]
        log.error('%s: %s: %s', case, type(exc).__name__, exc)
        return ([{'case': case, 'side': side, 'status': 'error', 'n_rays_cast': N_RAYS,
                  'n_rays_used': 0, 'error': f'{type(exc).__name__}: {exc}'}
                 for side in ('left', 'right')], [])


def discover_cases(inputs: list) -> list:
    """Collect the segmentation files to process, sorted for a deterministic sharding.

    :param inputs: Files and/or directories (searched recursively for ``*.nii`` / ``*.nii.gz``).
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
    return sorted(set(paths))


def _read_done(csv_path: Path) -> set:
    """Read the case ids already present in a shard summary, for resuming a run.

    :param csv_path: The shard's summary CSV.
    :return: The set of case ids already written.
    """
    if not csv_path.exists():
        return set()
    with open(csv_path, newline='') as fh:
        return {row['case'] for row in csv.DictReader(fh) if row.get('case')}


def _parse_shard(spec: str) -> tuple:
    """Parse a ``i/n`` shard specification into zero-based ``(index, count)``.

    :param spec: The shard specification, e.g. ``'3/8'``.
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


def merge_shards(out_dir: Path):
    """Concatenate every shard's summary and ray table into one pair of files.

    :param out_dir: The directory holding the shard files.
    :return: None.
    """
    summaries = sorted(out_dir.glob('summary_shard_*.csv'))
    if not summaries:
        raise SystemExit(f'no summary_shard_*.csv found in {out_dir}')

    rows = []
    for shard in summaries:
        with open(shard, newline='') as fh:
            rows.extend(csv.DictReader(fh))
    rows.sort(key=lambda r: (r['case'], r['side']))
    with open(out_dir / 'subchondral_summary.csv', 'w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    log.info('merged %d summary shards -> subchondral_summary.csv (%d rows)', len(summaries), len(rows))

    # The ray tables are far too large to sort in memory, so they are streamed in shard
    # order; rows of one (case, side) stay contiguous, which is what grouping needs.
    ray_shards = sorted(out_dir.glob('rays_shard_*.csv.gz'))
    written = 0
    with gzip.open(out_dir / 'subchondral_rays.csv.gz', 'wt', newline='') as out:
        writer = csv.DictWriter(out, fieldnames=RAY_FIELDS)
        writer.writeheader()
        for shard in ray_shards:
            with gzip.open(shard, 'rt', newline='') as fh:
                for row in csv.DictReader(fh):
                    writer.writerow(row)
                    written += 1
    log.info('merged %d ray shards -> subchondral_rays.csv.gz (%d rows)', len(ray_shards), written)


def run(paths: list, summary_path: Path, ray_path: Path, workers: int, overwrite: bool):
    """Measure every case in ``paths``, appending rows as they finish.

    Both tables are flushed per case, so an interrupted or pre-empted run keeps what it had
    and re-running the same command resumes from it.
    :param paths: The case paths this shard is responsible for.
    :param summary_path: The shard's summary CSV.
    :param ray_path: The shard's gzipped ray CSV.
    :param workers: The number of worker processes.
    :param overwrite: Recompute cases already present in the summary.
    :return: None.
    """
    done = set() if overwrite else _read_done(summary_path)
    todo = [p for p in paths if p.name.split('.')[0] not in done]
    if done:
        log.info('resuming: %d of %d cases already done', len(done), len(paths))
    if not todo:
        log.info('nothing to do')
        return

    write_summary_header = overwrite or not summary_path.exists()
    write_ray_header = overwrite or not ray_path.exists()
    mode = 'w' if overwrite else 'a'
    started = time.time()

    # Appending to a gzip file starts a new member; readers treat concatenated members as
    # one stream, so a resumed shard stays a single valid table.
    with open(summary_path, mode, newline='') as sfh, gzip.open(ray_path, mode + 't', newline='') as rfh:
        summary_writer = csv.DictWriter(sfh, fieldnames=SUMMARY_FIELDS, extrasaction='ignore')
        ray_writer = csv.DictWriter(rfh, fieldnames=RAY_FIELDS, extrasaction='ignore')
        if write_summary_header:
            summary_writer.writeheader()
            sfh.flush()
        if write_ray_header:
            ray_writer.writeheader()

        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_worker, str(p)): p for p in todo}
            for n, future in enumerate(as_completed(futures), start=1):
                summaries, rays = future.result()
                summary_writer.writerows(summaries)
                ray_writer.writerows(rays)
                sfh.flush()
                rfh.flush()
                if n % 50 == 0 or n == len(todo):
                    elapsed = time.time() - started
                    log.info('[%d/%d] %s (%.0fs elapsed, %.0f cases/min)', n, len(todo),
                             summaries[0]['case'], elapsed, 60 * n / max(elapsed, 1e-9))


def main(argv=None):
    """Parse the command line and run the batch or a merge.

    :param argv: Argument list, defaulting to ``sys.argv[1:]``.
    :return: None.
    """
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-i', '--input', nargs='+', default=[],
                        help='segmentation files and/or directories (searched recursively)')
    parser.add_argument('-o', '--output', type=Path, required=True,
                        help='output directory for the shard tables (and the merged pair)')
    parser.add_argument('--workers', type=int, default=None,
                        help='worker processes (default: SLURM_CPUS_PER_TASK, else all visible CPUs)')
    parser.add_argument('--shard', type=_parse_shard, default=None, metavar='I/N',
                        help='process shard I of N (default: taken from SLURM_ARRAY_TASK_ID, else all cases)')
    parser.add_argument('--limit', type=int, default=None, help='process at most this many cases (for a test run)')
    parser.add_argument('--overwrite', action='store_true', help='recompute cases already present in the shard')
    parser.add_argument('--merge', action='store_true', help='concatenate the shard tables and exit')
    parser.add_argument('-v', '--verbose', action='store_true', help='log per-hip failures')
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s', stream=sys.stdout)
    warnings.simplefilter('ignore')

    args.output.mkdir(parents=True, exist_ok=True)

    if args.merge:
        merge_shards(args.output)
        return

    if not args.input:
        parser.error('-i/--input is required unless --merge is given')

    paths = discover_cases(args.input)
    if args.limit:
        paths = paths[:args.limit]
    if not paths:
        raise SystemExit('no segmentations found')

    shard = args.shard or _shard_from_environment()
    index, count = shard if shard else (0, 1)
    if shard:
        paths = paths[index::count]  # stride, so every shard gets a comparable mix

    workers = args.workers or _default_workers()
    summary_path = args.output / f'summary_shard_{index:03d}.csv'
    ray_path = args.output / f'rays_shard_{index:03d}.csv.gz'
    log.info('shard %d/%d: %d cases, %d workers -> %s', index, count, len(paths), workers, summary_path)

    # A pre-emption or a time limit arrives as SIGTERM; the rows already flushed stay on
    # disk and re-running the same command resumes from them.
    signal.signal(signal.SIGTERM, lambda *_: sys.exit('received SIGTERM, stopping'))

    run(paths, summary_path, ray_path, workers, args.overwrite)
    log.info('shard %d/%d done', index, count)


if __name__ == '__main__':
    main()
