"""Batch the NaKo hip MRI measurements over one chunk of the cohort (one CSV per chunk).

The cohort (isotropic hip segmentations; femur=1, cartilage=2, acetabulum=3, both sides in one
mask) is stored in chunks ``<input-root>/chunk_<N>_segmentations``. For each case, both image
halves are measured (CCD, Murphy antetorsion, alpha angle, acetabular anteversion, centre-edge
angle, projected femoral offset, cartilage thickness) and written to
``<output-dir>/results_chunk_<N>.csv`` in the layout of the original cohort tables: one row per
(Patient, Side), where Side is the PATIENT side (the image-left half is the patient's right), plus
an appended ``qc_flags`` column holding the Murphy femoral-neck QC flags of that hip
(:func:`morphometry.femur.murphy_neck_base_qc`; empty when the neck looks sound).
A ``results_chunk_<N>.meta.json`` next to it records the git commit, date and arguments, so every
table is traceable to the code that produced it. An existing chunk CSV is never overwritten unless
``--overwrite`` is given. Failures are caught per measurement and written as NaN.

    python scripts/process_nako_large.py --chunk 0 --output-dir ~/Data/nako/rerun_<sha>
    python scripts/process_nako_large.py --output-dir ~/Data/nako/rerun_<sha> --merge   # -> results.csv

See ``scripts/process_nako_large.sbatch`` for the SLURM job array (one task per chunk).
"""
import os
import sys
import argparse
import datetime
import json
import re
import socket
import subprocess
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

import pandas as pd
import numpy as np

from morphometry.measurements.hip import calculate_ccd, calculate_anteversion, calculate_acetabular_anteversion, \
    calculate_alpha_angle, calculate_acetabular_depth, calculate_center_edge_angle, \
    calculate_cartilage_thickness_knn, calculate_femoral_offset, calculate_femoral_offset_projected
from morphometry.femur import get_murphy_neck_qc
from morphometry.image_io import Segmentation, split_image_left_right

DEFAULT_INPUT_ROOT = '/hpcwork/p0021834/workspace_simon/nako'
DEFAULT_OUTPUT_DIR = '/home/sw521914/Data/nako'
COLUMNS = ['CCD', 'AT_murphy', 'AA_anterior', 'AA_posterior', 'AAV', 'CE', 'Offset', 'Cartilage_thickness']


def _nan_result(patient_id: str, load_failed: bool = False) -> dict:
    """The result of a case that could not be measured at all: every value NaN.

    :param patient_id: The patient id (file name without extensions).
    :param load_failed: Mark the result as a load failure, which :func:`run_chunk` retries and accounts
        for rather than writing it out at once.
    :return: A result dict with the keys :func:`f` returns (plus ``load_failed`` if set).
    """
    nan2, nan4 = (np.nan, np.nan), (np.nan, np.nan, np.nan, np.nan)
    return {'patient': patient_id, 'load_failed': load_failed, 'ccd_left': np.nan, 'ccd_right': np.nan, 'fat_left': np.nan, 'fat_right': np.nan,
            'aa_left': nan2, 'aa_right': nan2, 'aav_left': np.nan, 'aav_right': np.nan,
            'cea_left': np.nan, 'cea_right': np.nan, 'offset_left': np.nan, 'offset_right': np.nan,
            'cartilage_thickness_left': nan4, 'cartilage_thickness_right': nan4,
            'murphy_qc_left': '', 'murphy_qc_right': ''}


def f(patient):
    # a case that cannot even be loaded becomes all-NaN rows, like a failed measurement, so one bad
    # file does not cost the whole chunk
    try:
        mask = Segmentation('nibabel')
        mask.read_image(patient)
        mask.transform_coordinate_system()
        mask.remove_outliers()

        mask_left, mask_right = split_image_left_right(mask)  # image-side halves with correct per-half geometry
    except Exception as e:
        print(f"Error loading patient {patient.name} (retried once at the end of the run): {e}", flush=True)
        return _nan_result(patient.name.split('.')[0], load_failed=True)

    try:
        # ccd_left, _ = calculate_ccd(mask_left, None, 'left', 1, isotropic=True, plot=p)
        _, ccd_left = calculate_ccd(mask_left, None, 'left', 1, isotropic=True, plot=False)  # use projected ccd
    except Exception as e:
        print(f"Error calculating CCD for left side of patient {patient.name}: {e}")
        ccd_left = np.nan

    try:
        # ccd_right, _ = calculate_ccd(mask_right, None, 'right', 1, isotropic=True, plot=p)
        _, ccd_right = calculate_ccd(mask_right, None, 'right', 1, isotropic=True, plot=False)
    except Exception as e:
        print(f"Error calculating CCD for right side of patient {patient.name}: {e}")
        ccd_right = np.nan

    try:
        fat_left = calculate_anteversion(mask_left, 'left', 1, isotropic=True, plot=False)
    except Exception as e:
        print(f"Error calculating FAT for left side of patient {patient.name}: {e}")
        fat_left = np.nan

    try:
        fat_right = calculate_anteversion(mask_right, 'right', 1, isotropic=True, plot=False)
    except Exception as e:
        print(f"Error calculating FAT for right side of patient {patient.name}: {e}")
        fat_right = np.nan

    try:
        aa_left = calculate_alpha_angle(mask_left.array, 'left', 1, isotropic=True, plot=False)
    except Exception as e:
        print(f"Error calculating AA for left side of patient {patient.name}: {e}")
        aa_left = (np.nan, np.nan)
    # calculate_alpha_angle(mask_left.array, 'left', 1, isotropic=True)

    try:
        aa_right = calculate_alpha_angle(mask_right.array, 'right', 1, isotropic=True, plot=False)
    except Exception as e:
        print(f"Error calculating AA for right side of patient {patient.name}: {e}")
        aa_right = (np.nan, np.nan)

    try:
        aav = calculate_acetabular_anteversion(mask, 1, 3, isotropic=True, plot=False)
    except Exception as e:
        print(f"Error calculating AAV for patient {patient.name}: {e}")
        aav = [np.nan, np.nan]

    try:
        cea = calculate_center_edge_angle(mask, 1, 3, project=True, isotropic=True, plot=False)
    except Exception as e:
        print(f"Error calculating CEA for left side of patient {patient.name}: {e}")
        cea = [np.nan, np.nan]

    try:
        offset_left = calculate_femoral_offset_projected(mask_left, None, 'left', 1, isotropic=True)
    except Exception as e:
        print(f"Error calculating femoral offset for left side of patient {patient.name}: {e}")
        offset_left = np.nan

    try:
        offset_right = calculate_femoral_offset_projected(mask_right, None, 'right', 1, isotropic=True)
    except Exception as e:
        print(f"Error calculating femoral offset for right side of patient {patient.name}: {e}")
        offset_right = np.nan

    try:
        cartilage_thickness_left = calculate_cartilage_thickness_knn(mask_left.array, 2)
    except Exception as e:
        print(f"Error calculating cartilage thickness for left side of patient {patient.name}: {e}")
        cartilage_thickness_left = (np.nan, np.nan, np.nan, np.nan)

    try:
        cartilage_thickness_right = calculate_cartilage_thickness_knn(mask_right.array, 2)
    except Exception as e:
        print(f"Error calculating cartilage thickness for right side of patient {patient.name}: {e}")
        cartilage_thickness_right = (np.nan, np.nan, np.nan, np.nan)

    # QC of the Murphy neck landmarks behind AT_murphy: a short or mis-placed neck axis is a large
    # angular error that the angle itself does not reveal. Flagged hips keep their measurement.
    murphy_qc = {}
    for side, half in (('left', mask_left), ('right', mask_right)):
        try:
            murphy_qc[f'murphy_qc_{side}'] = get_murphy_neck_qc(half, side, 1, isotropic=True)
        except Exception as e:
            print(f"Error QC-ing the Murphy neck for the {side} side of patient {patient.name}: {e}")
            murphy_qc[f'murphy_qc_{side}'] = ''

    return {'patient': patient.name.split('.')[0], 'ccd_left': ccd_left, 'ccd_right': ccd_right,'fat_left': fat_left, 'fat_right': fat_right,
            'aa_left': aa_left, 'aa_right': aa_right, 'aav_left': aav[0], 'aav_right': aav[1],
            'cea_left': cea[0], 'cea_right': cea[1], 'offset_left': offset_left, 'offset_right': offset_right,
            'cartilage_thickness_left': cartilage_thickness_left,'cartilage_thickness_right': cartilage_thickness_right,
            **murphy_qc}


def build_table(results: list, patients: list) -> pd.DataFrame:
    """Arrange the per-patient results in the layout of the original cohort tables.

    One row per (Patient, Side) with Side the PATIENT side: the image-left half (``*_left``) is the
    patient's right. Values are rounded to one decimal. The bare ``Cartilage_thickness`` column is
    kept (empty) because the original files have it.

    A ``qc_flags`` column is appended after those historical columns: the ``;``-separated
    ``name:value`` flags of :func:`morphometry.femur.murphy_neck_base_qc` for that hip's Murphy
    neck landmarks (empty when the neck looks sound). ``AT_murphy`` is written whether or not the
    hip is flagged -- the flag says the antetorsion of that row should not be trusted without
    looking at the mask, not that it was changed.
    :param results: Dicts as returned by :func:`f`.
    :param patients: Patient ids in row order.
    :return: The table, indexed by (Patient, Side).
    """
    index = pd.MultiIndex.from_product([patients, ['right', 'left']], names=['Patient', 'Side'])
    df = pd.DataFrame(columns=COLUMNS, index=index)

    for x in results:
        patient = x['patient']
        df.loc[(patient, 'right'), 'CCD'] = round(x['ccd_left'], 1)
        df.loc[(patient, 'left'), 'CCD'] = round(x['ccd_right'], 1)
        df.loc[(patient, 'right'), 'AT_murphy'] = round(x['fat_left'], 1)
        df.loc[(patient, 'left'), 'AT_murphy'] = round(x['fat_right'], 1)
        df.loc[(patient, 'right'), 'AA_anterior'] = round(x['aa_left'][0], 1)
        df.loc[(patient, 'left'), 'AA_anterior'] = round(x['aa_right'][0], 1)
        df.loc[(patient, 'left'), 'AA_posterior'] = round(x['aa_right'][1], 1)
        df.loc[(patient, 'right'), 'AA_posterior'] = round(x['aa_left'][1], 1)
        df.loc[(patient, 'right'), 'AAV'] = round(x['aav_left'], 1)
        df.loc[(patient, 'left'), 'AAV'] = round(x['aav_right'], 1)
        df.loc[(patient, 'right'), 'CE'] = round(x['cea_left'], 1)
        df.loc[(patient, 'left'), 'CE'] = round(x['cea_right'], 1)
        df.loc[(patient, 'right'), 'Offset'] = round(x['offset_left'], 1)
        df.loc[(patient, 'left'), 'Offset'] = round(x['offset_right'], 1)
        df.loc[(patient, 'right'), 'Cartilage_thickness (mean)'] = round(x['cartilage_thickness_left'][0], 1)
        df.loc[(patient, 'left'), 'Cartilage_thickness (mean)'] = round(x['cartilage_thickness_right'][0], 1)
        df.loc[(patient, 'right'), 'Cartilage_thickness (std)'] = round(x['cartilage_thickness_left'][1], 1)
        df.loc[(patient, 'left'), 'Cartilage_thickness (std)'] = round(x['cartilage_thickness_right'][1], 1)
        df.loc[(patient, 'right'), 'Cartilage_thickness (min)'] = round(x['cartilage_thickness_left'][2], 1)
        df.loc[(patient, 'left'), 'Cartilage_thickness (min)'] = round(x['cartilage_thickness_right'][2], 1)
        df.loc[(patient, 'right'), 'Cartilage_thickness (max)'] = round(x['cartilage_thickness_left'][3], 1)
        df.loc[(patient, 'left'), 'Cartilage_thickness (max)'] = round(x['cartilage_thickness_right'][3], 1)

    # appended after the original columns so the historical layout is untouched up to here
    df['qc_flags'] = ''
    for x in results:
        patient = x['patient']
        df.loc[(patient, 'right'), 'qc_flags'] = x.get('murphy_qc_left', '')
        df.loc[(patient, 'left'), 'qc_flags'] = x.get('murphy_qc_right', '')

    return df


def check_writable(path: Path, overwrite: bool):
    """Refuse to replace an existing result unless asked to.

    :param path: The output file about to be written.
    :param overwrite: Whether an existing file may be replaced.
    :raises FileExistsError: If ``path`` exists and ``overwrite`` is False.
    """
    if path.exists() and not overwrite:
        raise FileExistsError(f'{path} exists; pass --overwrite to replace it or choose another --output-dir')


def default_workers() -> int:
    """The number of worker processes, honouring the SLURM CPU allocation.

    :return: ``SLURM_CPUS_PER_TASK`` if set, else the CPUs this process may run on.
    """
    allocated = os.environ.get('SLURM_CPUS_PER_TASK')
    if allocated:
        return int(allocated)
    return len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else (os.cpu_count() or 1)


def _git(*args: str) -> str:
    """Run a git command in the repository this script belongs to.

    :param args: The git arguments.
    :return: Its stripped stdout, or ``'unknown'`` if git is unavailable or fails.
    """
    try:
        return subprocess.run(['git', '-C', str(REPO_ROOT), *args], capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return 'unknown'


def write_meta(path: Path, args: dict, n_cases: int, failures: dict = None):
    """Record which code produced a result table, and which of its cases are NaN for want of data.

    :param path: The meta JSON file to write.
    :param args: The command-line arguments (JSON-serialisable).
    :param n_cases: The number of cases processed.
    :param failures: ``{'load_failed': [ids], 'crashed': [ids]}``: cases written as all-NaN rows
        because they could not be loaded, or because they killed their worker even when run alone.
    """
    failures = failures or {'load_failed': [], 'crashed': []}
    meta = {'commit': _git('rev-parse', 'HEAD'),
            'dirty': _git('status', '--porcelain', '--untracked-files=no') != '',
            'branch': _git('rev-parse', '--abbrev-ref', 'HEAD'),
            'date': datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'),
            'host': socket.gethostname(),
            'slurm_job': os.environ.get('SLURM_ARRAY_JOB_ID') or os.environ.get('SLURM_JOB_ID'),
            'args': args, 'n_cases': n_cases,
            'n_load_failed': len(failures['load_failed']), 'load_failed': sorted(failures['load_failed']),
            'n_crashed': len(failures['crashed']), 'crashed': sorted(failures['crashed'])}
    path.write_text(json.dumps(meta, indent=2) + '\n')


def chunk_paths(out_dir: Path, chunk: int) -> dict:
    """The files one chunk writes.

    :param out_dir: The output directory.
    :param chunk: The chunk number.
    :return: ``{'final', 'partial', 'meta', 'suspects', 'failures'}`` paths. ``partial`` holds the rows
        flushed so far, ``suspects`` the cases in flight when a worker died, ``failures`` the cases
        written as NaN (load failures, isolated crashes) until the meta file takes them over.
    """
    stem = out_dir / f'results_chunk_{chunk}'
    return {'final': stem.with_suffix('.csv'), 'partial': stem.with_suffix('.partial.csv'),
            'meta': stem.with_suffix('.meta.json'), 'suspects': stem.with_suffix('.suspects.json'),
            'failures': stem.with_suffix('.failures.json')}


def _read_table(path: Path) -> pd.DataFrame:
    """Read a chunk table (final or partial) with the patient ids kept as strings.

    :param path: The CSV file.
    :return: The table, indexed by (Patient, Side).
    """
    return pd.read_csv(path, index_col=[0, 1], dtype={'Patient': str, 'Side': str})


def _load_partial(partial: Path) -> pd.DataFrame:
    """Read the partial chunk file, dropping a case torn by a kill mid-write.

    A case is complete when both of its rows (right and left) are present and the file ends with a
    newline; a torn trailing line is cut off and an incomplete case is dropped, so it is recomputed.
    The file is rewritten if anything was dropped, and deleted if not even its header is complete.
    :param partial: The partial CSV.
    :return: The complete rows, indexed by (Patient, Side) (empty if nothing usable was left).
    """
    text = partial.read_text()
    text = text[:text.rfind('\n') + 1]  # cut a torn trailing line ('' if no line is complete)
    if not text:
        partial.unlink()
        return build_table([], [])
    partial.write_text(text)
    table = _read_table(partial)
    sides = table.groupby(level=0).size()
    torn = sides.index[sides != 2]
    if len(torn):
        table = table.drop(index=torn, level=0)
        table.to_csv(partial)
    return table


def _flush(partial: Path, result: dict):
    """Append one case's two rows to the partial file in a single write.

    :param partial: The partial CSV (created with a header if absent).
    :param result: The case's result dict.
    """
    rows = build_table([result], [result['patient']]).to_csv(header=not partial.exists())
    with open(partial, 'a') as fh:  # one write per case, so a kill leaves at most one torn case
        fh.write(rows)


def _read_json(path: Path, default):
    """Read a JSON sidecar file.

    :param path: The file.
    :param default: Returned if the file does not exist.
    :return: Its content.
    """
    return json.loads(path.read_text()) if path.exists() else default


def _run_pool(cases: list, workers: int, worker, on_result, on_break):
    """Run ``worker`` over ``cases`` with at most ``workers`` cases in flight.

    Cases are submitted only as workers free up, so the in-flight set is exactly the cases being
    worked on; if a worker dies hard, ``on_break`` receives them (the suspects) before
    ``BrokenProcessPool`` propagates.
    :param cases: The case files.
    :param workers: Number of worker processes.
    :param worker: The per-case function.
    :param on_result: Called with each result dict as it completes.
    :param on_break: Called with the list of in-flight cases when the pool breaks.
    """
    queue = iter(cases)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        inflight = {}
        try:
            while True:
                for case in queue:
                    inflight[pool.submit(worker, case)] = case
                    if len(inflight) >= workers:
                        break
                if not inflight:
                    return
                done, _ = wait(inflight, return_when=FIRST_COMPLETED)
                for future in done:
                    result = future.result()  # raises BrokenProcessPool if a worker died
                    del inflight[future]
                    on_result(result)
        except BrokenProcessPool:
            on_break(list(inflight.values()))
            raise


def run_chunk(patients: list, out_dir: Path, chunk: int, workers: int, overwrite: bool, args: dict, worker=None,
              max_load_fail_frac: float = 0.01):
    """Measure one chunk, flushing each case to a partial CSV, then write the sorted chunk table.

    Each finished case is appended to ``results_chunk_<N>.partial.csv`` at once, so an interrupted
    task loses nothing: re-running it skips every patient already there (a case torn by a kill
    mid-write is recomputed). When all patients are in, the rows are sorted into the layout of
    :func:`build_table` and written as ``results_chunk_<N>.csv`` with the meta file, and the
    sidecar files are removed.

    Two kinds of case end up as all-NaN rows, and both are listed in the meta file:

    * a case that fails to load is not flushed but retried once at the end of the run; if it still
      fails it is written as NaN -- unless more than ``max_load_fail_frac`` of the chunk failed to
      load, which points at the storage rather than the files, and the run fails (``RuntimeError``)
      without finalising, so a resubmission retries them;
    * a worker that dies hard (OOM kill, segfault) breaks the pool, which raises
      ``BrokenProcessPool`` instead of hanging, after the cases in flight are saved as suspects. The
      next run measures each suspect alone in its own one-worker pool first; a suspect that breaks
      even alone is the poison case and is written as NaN with a log line.
    :param patients: The case files (``.nii.gz``).
    :param out_dir: The output directory (created if needed).
    :param chunk: The chunk number.
    :param workers: Number of worker processes.
    :param overwrite: Discard an existing chunk table and every sidecar file and start afresh. Without
        it a finished chunk is refused (``FileExistsError``) and a partial one resumed.
    :param args: The command-line arguments, recorded in the meta file.
    :param worker: The per-case function (default :func:`f`); must be picklable.
    :param max_load_fail_frac: Largest fraction of the chunk allowed to fail to load.
    """
    worker = worker or f
    paths = chunk_paths(out_dir, chunk)
    partial = paths['partial']
    check_writable(paths['final'], overwrite)  # fail before hours of work, not after
    if overwrite:
        for path in paths.values():
            path.unlink(missing_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    ids = {p.name.split('.')[0]: p for p in patients}
    done = set(_load_partial(partial).index.get_level_values(0)) if partial.exists() else set()
    failures = _read_json(paths['failures'], {'load_failed': [], 'crashed': []})
    suspects = [ids[i] for i in _read_json(paths['suspects'], []) if i in ids and i not in done]
    todo = [p for i, p in ids.items() if i not in done and p not in suspects]
    print(f'chunk {chunk}: {len(ids)} cases, {len(done)} already done, {len(suspects)} suspects, '
          f'{len(todo)} to go, {workers} workers', flush=True)

    load_failed = []

    def collect(result):
        """Flush a result, or hold it back for a retry if the case did not load."""
        if result.get('load_failed'):
            load_failed.append(ids[result['patient']])
        else:
            _flush(partial, result)

    def save_suspects(inflight):
        """Remember the cases in flight when the pool broke."""
        names = sorted(set(_read_json(paths['suspects'], [])) | {p.name.split('.')[0] for p in inflight})
        paths['suspects'].write_text(json.dumps(names))
        print(f'chunk {chunk}: a worker died; suspects saved for an isolated rerun: {names}', flush=True)

    # suspects first, each alone, so a poison case can only break its own pool
    for case in suspects:
        name = case.name.split('.')[0]
        try:
            _run_pool([case], 1, worker, collect, lambda inflight: None)
        except BrokenProcessPool:
            print(f'chunk {chunk}: {name} kills its worker even when run alone; written as NaN', flush=True)
            _flush(partial, _nan_result(name))
            failures['crashed'].append(name)
            paths['failures'].write_text(json.dumps(failures))
    paths['suspects'].unlink(missing_ok=True)

    _run_pool(todo, workers, worker, collect, save_suspects)

    if load_failed:  # one retry, e.g. for a transient I/O error
        retry, load_failed = load_failed, []
        _run_pool(retry, workers, worker, collect, save_suspects)
    if len(load_failed) > max_load_fail_frac * len(ids):
        raise RuntimeError(f'chunk {chunk}: {len(load_failed)} of {len(ids)} cases failed to load (limit '
                           f'{max_load_fail_frac:.1%}), e.g. {[p.name for p in load_failed[:5]]}; check the input '
                           f'mount and resubmit -- the cases that loaded are kept in {partial}')
    for case in load_failed:
        name = case.name.split('.')[0]
        _flush(partial, _nan_result(name))
        failures['load_failed'].append(name)
    paths['failures'].write_text(json.dumps(failures))

    table = _load_partial(partial) if partial.exists() else build_table([], [])
    order = pd.MultiIndex.from_product([sorted(ids), ['right', 'left']], names=['Patient', 'Side'])
    missing = sorted(set(ids) - set(table.index.get_level_values(0)))
    if missing:
        raise RuntimeError(f'chunk {chunk}: no result for {missing}; {partial} kept for a resume')
    table.reindex(order).to_csv(paths['final'])
    write_meta(paths['meta'], args, len(ids), failures)
    print(f'chunk {chunk}: done, {len(ids)} cases; all-NaN because they failed to load: '
          f'{sorted(failures["load_failed"])}, because they crashed their worker: {sorted(failures["crashed"])}',
          flush=True)
    for key in ('partial', 'suspects', 'failures'):
        paths[key].unlink(missing_ok=True)


def merge_chunks(out_dir: Path, output: Path, overwrite: bool, expected_chunks: int = 20):
    """Concatenate ``results_chunk_<N>.csv`` files, in numeric chunk order, into one table.

    Refuses an incomplete or inconsistent set: every chunk ``0 .. expected_chunks-1`` must be
    finished (a chunk with only a partial file is unfinished), and no (Patient, Side) may occur twice.
    :param out_dir: Directory holding the chunk CSVs.
    :param output: The merged CSV to write.
    :param overwrite: Whether an existing merged file may be replaced.
    :param expected_chunks: The number of chunks the cohort was split into.
    :raises FileNotFoundError: If a chunk is missing or unfinished.
    :raises ValueError: If a (Patient, Side) row occurs more than once.
    """
    check_writable(output, overwrite)
    chunk_number = lambda p: int(re.search(r'_(\d+)(\.partial)?\.csv$', p.name).group(1))
    finished = {chunk_number(p): p for p in out_dir.glob('results_chunk_*.csv') if not p.name.endswith('.partial.csv')}
    unfinished = sorted(chunk_number(p) for p in out_dir.glob('results_chunk_*.partial.csv'))
    extra = sorted((set(finished) | set(unfinished)) - set(range(expected_chunks)))
    if extra:
        raise ValueError(f'in {out_dir}: chunks {extra} lie outside 0..{expected_chunks - 1}; '
                         f'wrong --expected-chunks or a stray file')
    missing = sorted(set(range(expected_chunks)) - set(finished))
    if missing or unfinished:
        raise FileNotFoundError(f'in {out_dir}: missing chunks {missing}, unfinished chunks {unfinished} '
                                f'(expected {expected_chunks})')
    merged = pd.concat([_read_table(finished[c]) for c in sorted(finished)])
    duplicated = merged.index[merged.index.duplicated()].unique()
    if len(duplicated):
        raise ValueError(f'{len(duplicated)} duplicate (Patient, Side) rows, e.g. {list(duplicated[:5])}')
    merged.to_csv(output)
    print(f'merged {len(finished)} chunks into {output}: {len(merged)} rows')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process one chunk of the NaKo hip cohort')
    parser.add_argument('--chunk', type=int, default=0, help='Chunk number to process')
    parser.add_argument('--input-root', type=Path, default=Path(DEFAULT_INPUT_ROOT),
                        help='directory holding chunk_<N>_segmentations/ (default: %(default)s)')
    parser.add_argument('--output-dir', type=Path, default=Path(DEFAULT_OUTPUT_DIR),
                        help='directory for results_chunk_<N>.csv and its .meta.json (default: %(default)s)')
    parser.add_argument('--workers', type=int, default=None,
                        help='worker processes (default: SLURM_CPUS_PER_TASK, else all visible CPUs)')
    parser.add_argument('--overwrite', action='store_true',
                        help='discard an existing chunk table / partial file (or merged table) and start afresh')
    parser.add_argument('--merge', action='store_true',
                        help='concatenate the chunk CSVs in --output-dir into results.csv and exit')
    parser.add_argument('--max-load-fail-frac', type=float, default=0.01,
                        help='fail the chunk if more than this fraction of its cases cannot be loaded '
                             '(default: %(default)s)')
    parser.add_argument('--expected-chunks', type=int, default=20,
                        help='with --merge: chunks 0..N-1 must all be finished (default: %(default)s)')
    args = parser.parse_args()

    if args.merge:
        merge_chunks(args.output_dir, args.output_dir / 'results.csv', args.overwrite, args.expected_chunks)
        sys.exit(0)

    chunk_dir = args.input_root / f'chunk_{args.chunk}_segmentations'
    patients = [patient for patient in sorted(chunk_dir.iterdir()) if patient.name.endswith('.nii.gz')]
    if not patients:
        raise FileNotFoundError(f'no .nii.gz segmentations in {chunk_dir}')
    run_chunk(patients, args.output_dir, args.chunk, args.workers or default_workers(), args.overwrite,
              {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              max_load_fail_frac=args.max_load_fail_frac)
