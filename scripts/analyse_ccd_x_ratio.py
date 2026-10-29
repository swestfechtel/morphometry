#!/usr/bin/env python3
"""Measure what the MRI CCD's ``x_ratio`` slice-thickness factor is actually worth.

Backlog task #6. The torsion pipelines all pass

    x_ratio = abs(hip.spacing[2]) / 2 * abs(hip.spacing[0])

which is not a slice aspect ratio -- it is ``mm / 2 x mm``, dimensionally an area,
and it only happens to land on a plausible number for the Augsburg hip geometry.
The obvious correction is the true aspect ``spacing[2] / spacing[0]``. This script
exists because that correction makes the CCD measurably **worse**, and the reason
is worth keeping: the femoral-neck landmark samples a *sphere* shell in the scaled
index space (``hip.get_femoral_neck_center`` -> ``_femoral_neck_points_and_center``),
so a larger ``x_ratio`` shrinks the shell's reach in slices. On 10 mm Augsburg
slices the true aspect leaves the neck point cloud spanning two or three slices,
and the neck centroid's superior-inferior coordinate is quantised away.

What it reports, over the 50 Augsburg reader-study patients (both legs, so 100
image halves):

* a **sweep** of ``x_ratio``, with the CCD mean, the left/right agreement within a
  patient (which needs no reader at all and so is the one criterion free of the
  readers' own scatter), the bias and MAD against the two readers' mean, and the
  size and slice span of the neck point cloud the factor selects;
* the **torsion invariance** check at the two named factors -- the femoral torsion
  is expected to be bit-identical, because on the non-isotropic path ``x_ratio``
  reaches nothing but the femoral-head slab bound, which does not bite here. This
  is the claim that confines the whole item to the CCD, so it is measured, not
  assumed;
* a **conditioning** table: how many halves get a neck cloud thinner than a stated
  number of slices at each factor.

Writes ``sweep.csv`` (long: patient x side x factor), ``summary.csv`` (one row per
factor), ``torsion.csv`` (per half, both factors), ``report.md`` and
``report.meta.json`` (commit, dirty flag, date, args, cohort size) to ``-o``. It
never writes outside that directory.

    python scripts/analyse_ccd_x_ratio.py -o out/ccd_x_ratio [--workers 12]

**This script does not pick a factor, and its numbers must not be used to.** The
two criteria it reports disagree -- left/right agreement peaks near 3-4 and the
reader bias near 5.47 -- and both are computed on the same 50 patients that any
chosen value would then be quoted against, which is the calibration-on-the-
evaluation-set mistake ``CLAUDE.md`` already records for ``CEA_UPPER_MARGIN_FACTOR``.
The findings drawn from it are in ``docs/ccd_x_ratio.md``.

:return: Nothing; the findings are the written files.
"""
import argparse
import json
import subprocess
import sys
import warnings
from datetime import datetime, timezone
from importlib.util import module_from_spec, spec_from_file_location
from multiprocessing import Pool
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from morphometry.hip import get_femoral_head_center, get_femoral_neck_center
from morphometry.image_io import (Segmentation, mirror_image_left_right,
                                  split_image_left_right)
from morphometry.measurements.femur import calculate_femoral_torsion
from morphometry.measurements.hip import calculate_ccd

#: Reader-study segmentations, one directory per patient (as ``analyse_reader_bias.py``).
DEFAULT_ROOT = Path('/home/simon/Data/Augsburg_large/reader_study_augsburg')
#: The two readers' manual measurements.
DEFAULT_TABLES = Path('/home/simon/Data/Augsburg_large/tables')

#: The factors swept. 5.46875 is what the pipelines compute on this cohort
#: (``spacing_z / 2 * spacing_x`` at 10 mm slices and 1.09375 mm in plane) and
#: 9.142857 is the true slice aspect ``spacing_z / spacing_x``; the rest bracket them
#: so the behaviour either side of both is visible rather than inferred.
SWEEP_FACTORS = (1.0, 2.0, 3.0, 4.0, 5.46875, 7.0, 9.142857, 12.0)
#: The two factors the item is actually about, by name.
PIPELINE_FACTOR, ASPECT_FACTOR = 5.46875, 9.142857

#: The per-measurement exceptions the batch scripts already treat as "no value".
MEASUREMENT_ERRORS = (ValueError, IndexError, RuntimeError, AssertionError)

#: Neck clouds at or below this many slices are reported as degenerate: the centroid's
#: z is then pinned to one or two 10 mm sample positions rather than interpolated over a
#: band, which is the mechanism this script exists to show.
THIN_NECK_SLICES = 2


def _reader_helpers():
    """Borrow ``load_readers`` / ``reader_means`` from the sibling reader-bias script.

    The two scripts read the same two xlsx files with the same two trailing summary
    rows to drop, and duplicating that parsing is how the drop gets forgotten in one
    copy. ``scripts/`` is not a package, so the sibling is loaded by path.

    :return: The ``(load_readers, reader_means)`` callables of
        ``scripts/analyse_reader_bias.py``.
    """
    path = Path(__file__).resolve().parent / 'analyse_reader_bias.py'
    spec = spec_from_file_location('analyse_reader_bias', path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.load_readers, module.reader_means


def _load_halves(patient: str, root: Path):
    """Load one patient's hip and knee segmentations and split them into image halves.

    Mirrors the production preparation exactly -- LPI reorientation then
    ``remove_outliers()`` -- so the CCD values here are the pipeline's own.

    :param patient: The patient directory name, e.g. ``'PA000001'``.
    :param root: The cohort root holding one directory per patient.
    :return: A dict ``{image_side: (hip_half, knee_half)}`` plus the hip spacing, or
        ``None`` if the patient's files are absent.
    """
    segs = {}
    for part in ('hip', 'knee'):
        seg = Segmentation('nibabel')
        try:
            seg.read_image(str(root / patient / f'{part}_seg_test.nii.gz'))
        except FileNotFoundError:
            return None
        seg.transform_coordinate_system()
        seg.remove_outliers()
        segs[part] = seg
    hips = split_image_left_right(segs['hip'])
    knees = split_image_left_right(segs['knee'])
    halves = {side: (hips[i], knees[i]) for i, side in enumerate(('left', 'right'))}
    return halves, np.abs(np.asarray(segs['hip'].spacing, dtype=float))


def measure_patient(args) -> list:
    """Measure one patient's CCD and neck point cloud at every swept factor.

    The image-left half is the patient's RIGHT leg, so rows are keyed by patient side
    to match the readers' tables. The neck cloud is re-derived in the same mirrored
    frame ``calculate_ccd`` uses internally for a right half, so its size and slice
    span describe what the measurement actually saw.

    :param args: ``(patient_id, root, factors)``.
    :return: A list of row dicts, one per (side, factor); empty if the patient's
        files are absent.
    """
    patient, root, factors = args
    loaded = _load_halves(patient, Path(root))
    if loaded is None:
        return []
    halves, spacing = loaded

    rows = []
    for image_side, (hip, knee) in halves.items():
        patient_side = 'right' if image_side == 'left' else 'left'
        for factor in factors:
            row = {'patient': patient, 'side': patient_side, 'x_ratio': factor,
                   'spacing_x': spacing[0], 'spacing_z': spacing[2],
                   'n_slices': hip.array.shape[2]}
            try:
                actual, projected = calculate_ccd(hip, knee, image_side, 1, False, factor)
                row['ccd'], row['ccd_projected'] = actual, projected
            except MEASUREMENT_ERRORS as exc:
                row['ccd'] = row['ccd_projected'] = np.nan
                row['ccd_err'] = type(exc).__name__
            try:
                # calculate_ccd measures a right half as the mirror of a left one
                frame = mirror_image_left_right(hip) if image_side == 'right' else hip
                radius, head = get_femoral_head_center(
                    frame.array, side='left', segmentation_label=1, x_ratio=factor,
                    isotropic=False, spacing=frame.spacing)
                points, centre = get_femoral_neck_center(
                    frame.array, (radius, head), side='left', segmentation_label=1,
                    x_ratio=factor)
                unique = np.unique(points.round(6), axis=0)
                row['head_radius_voxels'] = radius
                row['neck_points'] = len(points)
                row['neck_voxels'] = len(unique)
                row['neck_slice_span'] = float(unique[:, 2].max() - unique[:, 2].min()) + 1
                row['neck_centre_z'] = centre[2]
                row['head_centre_z'] = head[2]
            except Exception as exc:                      # noqa: BLE001 - diagnostic only
                row['neck_voxels'] = np.nan
                row['neck_err'] = type(exc).__name__
            rows.append(row)
    return rows


def measure_torsion(args) -> list:
    """Measure Lee and Murphy femoral torsion at the two named factors.

    This is the check that confines task #6 to the CCD: on the non-isotropic MRI path
    ``x_ratio`` reaches only the femoral-head slab bound, which is inert on this
    geometry, so both methods should be bit-identical between the two factors.

    :param args: ``(patient_id, root)``.
    :return: A list of row dicts, one per (side, method), carrying the value at each
        of the two factors and their difference.
    """
    patient, root = args
    loaded = _load_halves(patient, Path(root))
    if loaded is None:
        return []
    halves, _ = loaded

    rows = []
    for image_side, (hip, knee) in halves.items():
        patient_side = 'right' if image_side == 'left' else 'left'
        for method in ('lee', 'murphy'):
            row = {'patient': patient, 'side': patient_side, 'method': method}
            for name, factor in (('pipeline', PIPELINE_FACTOR), ('aspect', ASPECT_FACTOR)):
                try:
                    row[name] = calculate_femoral_torsion(
                        hip, knee, side=image_side, method=method, x_ratio=factor)
                except MEASUREMENT_ERRORS as exc:
                    row[name] = np.nan
                    row[f'{name}_err'] = type(exc).__name__
            row['difference'] = row['aspect'] - row['pipeline']
            rows.append(row)
    return rows


def summarise(sweep: pd.DataFrame, reader_means: dict,
              thin_slices: int = THIN_NECK_SLICES) -> pd.DataFrame:
    """Reduce the long sweep to one row per factor.

    Every column is a criterion by which a factor could be judged, deliberately
    including two that disagree -- the reader-free left/right agreement and the bias
    against the readers -- because that disagreement is the reason the script refuses
    to pick a factor.

    :param sweep: The long sweep frame, with ``patient``, ``side``, ``x_ratio``,
        ``ccd`` and the neck-cloud columns.
    :param reader_means: ``{(measure, side): Series}`` as
        ``analyse_reader_bias.reader_means`` returns, indexed by patient.
    :param thin_slices: Neck clouds spanning at most this many slices count as thin.
    :return: One row per factor: ``x_ratio, n, ccd_mean, ccd_sd, lr_pearson, lr_mad,
        bias_left, bias_right, mad_left, mad_right, neck_voxels_median,
        neck_slice_span_median, thin_halves``.
    """
    rows = []
    for factor, part in sweep.groupby('x_ratio'):
        wide = part.pivot_table(index='patient', columns='side', values='ccd')
        paired = wide['left'].notna() & wide['right'].notna() if {'left', 'right'} <= set(wide) \
            else pd.Series(False, index=wide.index)
        left, right = wide.get('left'), wide.get('right')
        row = {'x_ratio': factor, 'n': int(part['ccd'].notna().sum()),
               'ccd_mean': part['ccd'].mean(), 'ccd_sd': part['ccd'].std(),
               'lr_pearson': np.nan, 'lr_mad': np.nan}
        if paired.sum() >= 1:
            row['lr_mad'] = float((left[paired] - right[paired]).abs().mean())
        if paired.sum() >= 3:
            # a correlation over one or two patients is not a number worth printing
            row['lr_pearson'] = float(np.corrcoef(left[paired], right[paired])[0, 1])

        for side in ('left', 'right'):
            reference = reader_means.get(('ccd', side))
            row[f'bias_{side}'] = row[f'mad_{side}'] = np.nan
            if reference is None:
                continue
            values = part[part['side'] == side].set_index('patient')['ccd']
            shared = reference.index.intersection(values.index)
            both = values.loc[shared].notna() & reference.loc[shared].notna()
            if both.sum():
                delta = values.loc[shared][both] - reference.loc[shared][both]
                row[f'bias_{side}'] = float(delta.mean())
                row[f'mad_{side}'] = float(delta.abs().mean())

        row['neck_voxels_median'] = part['neck_voxels'].median()
        row['neck_slice_span_median'] = part['neck_slice_span'].median()
        span = part['neck_slice_span']
        row['thin_halves'] = int((span <= thin_slices).sum())
        rows.append(row)
    return pd.DataFrame(rows).sort_values('x_ratio').reset_index(drop=True)


def write_report(out_dir: Path, summary: pd.DataFrame, torsion: pd.DataFrame,
                 sweep: pd.DataFrame, thin_slices: int = THIN_NECK_SLICES) -> None:
    """Write the human-readable ``report.md``.

    :param out_dir: The output directory (created by the caller).
    :param summary: The per-factor summary from :func:`summarise`.
    :param torsion: The per-half torsion frame from :func:`measure_torsion`.
    :param sweep: The long sweep frame, for the cohort geometry header.
    :param thin_slices: The threshold used for the ``thin_halves`` column.
    :return: Nothing; writes ``out_dir / 'report.md'``.
    """
    geometry = sweep[['spacing_x', 'spacing_z']].drop_duplicates()
    lines = [
        '# What the CCD\'s `x_ratio` slice-thickness factor is worth',
        '',
        'Backlog task #6, regenerated by `scripts/analyse_ccd_x_ratio.py`.',
        '',
        f'Cohort geometry: {len(geometry)} distinct (in-plane, slice) spacing pair(s) -- '
        + ', '.join(f'{r.spacing_x:g} x {r.spacing_z:g} mm' for r in geometry.itertuples()),
        f'Slices per image half: {sorted(sweep.n_slices.unique().tolist())}',
        f'Femoral head radius: median {sweep.head_radius_voxels.median():.1f} in-plane voxels',
        '',
        '## The sweep',
        '',
        f'`lr_*` is the within-patient left/right agreement, which involves no reader. '
        f'`thin_halves` counts image halves whose neck point cloud spans at most '
        f'{thin_slices} slices.',
        '',
        summary.to_markdown(index=False, floatfmt='.3f'),
        '',
        '## Femoral torsion is untouched',
        '',
    ]
    if torsion.empty:
        lines.append('No torsion rows were measured.')
    else:
        resolved = torsion.dropna(subset=['difference'])
        lines += [
            f'{len(resolved)} of {len(torsion)} (half x method) measurements resolved at both '
            f'factors. Maximum absolute difference between the pipeline factor and the true '
            f'slice aspect: **{resolved.difference.abs().max():.3e} degrees** '
            f'(non-zero on {int((resolved.difference != 0).sum())} of {len(resolved)}).',
            '',
            'On the non-isotropic MRI path `x_ratio` reaches only the femoral-head slab bound, '
            'which does not bite on this geometry, so the whole item is confined to the CCD.',
        ]
    (out_dir / 'report.md').write_text('\n'.join(lines) + '\n')


def write_meta(out_dir: Path, args: argparse.Namespace, patients: list) -> None:
    """Write ``report.meta.json`` -- the provenance a re-derivable table needs.

    :param out_dir: The output directory.
    :param args: The parsed command line.
    :param patients: The patient directory names measured.
    :return: Nothing; writes ``out_dir / 'report.meta.json'``.
    """
    def _git(*cmd):
        try:
            return subprocess.run(('git',) + cmd, cwd=Path(__file__).resolve().parent.parent,
                                  capture_output=True, text=True, check=True).stdout.strip()
        except Exception:                                 # noqa: BLE001 - provenance is best-effort
            return None

    meta = {
        'commit': _git('rev-parse', 'HEAD'),
        'branch': _git('rev-parse', '--abbrev-ref', 'HEAD'),
        'dirty': bool(_git('status', '--porcelain')),
        'date': datetime.now(timezone.utc).isoformat(),
        'args': {k: str(v) for k, v in vars(args).items()},
        'n_patients': len(patients),
        'factors': list(SWEEP_FACTORS),
    }
    (out_dir / 'report.meta.json').write_text(json.dumps(meta, indent=2) + '\n')


def main() -> None:
    """Parse the command line, measure the cohort and write the four output files.

    :return: Nothing.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('-o', '--out-dir', type=Path, required=True,
                        help='Directory for sweep.csv / summary.csv / torsion.csv / report.md.')
    parser.add_argument('-i', '--root', type=Path, default=DEFAULT_ROOT,
                        help='Reader-study cohort root, one directory per patient.')
    parser.add_argument('--tables', type=Path, default=DEFAULT_TABLES,
                        help="Directory holding the two readers' xlsx files.")
    parser.add_argument('--workers', type=int, default=12, help='Worker processes.')
    parser.add_argument('--skip-torsion', action='store_true',
                        help='Skip the torsion invariance check (it roughly doubles the runtime).')
    args = parser.parse_args()

    warnings.filterwarnings('ignore')
    args.out_dir.mkdir(parents=True, exist_ok=True)
    patients = sorted(p.name for p in args.root.iterdir() if p.is_dir())
    if not patients:
        raise SystemExit(f'no patient directories under {args.root}')

    with Pool(args.workers) as pool:
        nested = pool.map(measure_patient,
                          [(p, str(args.root), SWEEP_FACTORS) for p in patients])
        sweep = pd.DataFrame([row for rows in nested for row in rows])
        if args.skip_torsion:
            torsion = pd.DataFrame()
        else:
            nested = pool.map(measure_torsion, [(p, str(args.root)) for p in patients])
            torsion = pd.DataFrame([row for rows in nested for row in rows])

    load_readers, reader_means = _reader_helpers()
    means = reader_means(load_readers(args.tables))

    summary = summarise(sweep, means)
    sweep.to_csv(args.out_dir / 'sweep.csv', index=False)
    summary.to_csv(args.out_dir / 'summary.csv', index=False)
    if not torsion.empty:
        torsion.to_csv(args.out_dir / 'torsion.csv', index=False)
    write_report(args.out_dir, summary, torsion, sweep)
    write_meta(args.out_dir, args, patients)
    print(summary.to_string(index=False))
    print(f'\nwritten to {args.out_dir}')


if __name__ == '__main__':
    main()
