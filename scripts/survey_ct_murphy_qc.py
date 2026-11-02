#!/usr/bin/env python3
"""Measure the Murphy femoral-neck QC on whole-leg CT (backlog task #3).

`murphy_neck_base_qc` was built and calibrated on MRI: 2128 Augsburg / NaKo image
halves, of which 1887 yield a Murphy landmark and 5 are flagged. **CT was never
measured** -- the sshfs mount holding the NMDID cohort was down -- so three things
were unknown, and `CLAUDE.md` said so:

* the CT **landmark failure rate** and its reason codes (the MRI analogue is
  241 / 2128 halves, led by `trochanter_implausible`);
* the CT **flag rate**, and in particular how the `murphy_axis_short` threshold
  behaves there. That threshold is an angular tolerance converted to **in-plane
  voxels** (`1 / tan(MURPHY_AXIS_ANGLE_TOLERANCE_DEG)` = 14.30 voxels at 4 deg), so
  it scales with the voxel size: 15.6 mm on Augsburg, 14.3 on NaKo, ~11 mm on CT.
  It was placed using the MRI cohorts' own distribution, so its CT behaviour is a
  genuine extrapolation, not a validated setting;
* whether the hard **one-voxel raise** in `get_femoral_neck_base` ever fires. It
  never did on the 1887 MRI halves (shortest axis 6.4 voxels) and is a **new
  failure mode on CT**, where the voxels are smaller and the neck is the same size.

This script answers all three on whatever cases it is pointed at, and reports the
axis-length **distribution** rather than only the flag count -- a rate alone cannot
say whether a cohort sits far from the threshold or straddles it.

It re-uses `scripts/process_nmdid.py`'s own `load_case` and `split_legs`, so each
half is prepared exactly as the production rows are (LPI, outliers removed, arm and
spine fragments dropped by `clean_leg`), and it calls `murphy_neck_base_qc` through
the same `_ct_proximal_landmarks` the measurement uses -- keeping the metrics dict
that `get_murphy_neck_qc_ct` discards.

    python scripts/survey_ct_murphy_qc.py -i <dir|file ...> -o out/ct_murphy_qc [--workers 8]

Writes `halves.csv` (one row per case x image side: every metric, the flag string and
the failure reason), `summary.md` and `summary.meta.json`.

:return: Nothing; the findings are the written files.
"""
import argparse
import json
import logging
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

from morphometry import constants as C
from morphometry.femur import (_ct_proximal_landmarks, _landmark_failure_flag,
                               murphy_neck_base_qc)
from morphometry.image_io import reflect_image_left_right

#: The in-plane axis length, in voxels, below which `murphy_axis_short` fires.
AXIS_THRESHOLD_VOXELS = 1.0 / np.tan(np.deg2rad(C.MURPHY_AXIS_ANGLE_TOLERANCE_DEG))

#: The metric columns carried through to the CSV and summarised.
METRICS = ('axis_length_mm', 'axis_length_voxels', 'angle_per_voxel_deg',
           'head_clearance_mm', 'base_distance_to_bone_mm', 'base_in_bone',
           'base_slice_components')


def _nmdid():
    """Load `scripts/process_nmdid.py` by path (`scripts/` is not a package).

    Re-using its `load_case` / `split_legs` is the point: a survey that prepared the
    legs differently from production would describe a different neck.

    :return: The imported `process_nmdid` module.
    """
    path = Path(__file__).resolve().parent / 'process_nmdid.py'
    spec = spec_from_file_location('process_nmdid', path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def survey_case(path_str: str) -> list:
    """Run the Murphy neck QC on both image halves of one whole-leg CT case.

    :param path_str: Path to the case's `.nii.gz` segmentation.
    :return: Two row dicts (one per image side), or one row carrying `load_error` if
        the case cannot be loaded or split at all.
    """
    warnings.filterwarnings('ignore')
    logging.getLogger('morphometry').setLevel(logging.ERROR)
    path = Path(path_str)
    case = path.name.replace('_seg.nii.gz', '').replace('.nii.gz', '')
    nmdid = _nmdid()

    try:
        seg = nmdid.load_case(path)
        halves, _, _ = nmdid.split_legs(seg)
        spacing = np.abs(np.asarray(seg.spacing, dtype=float))
    except Exception as exc:                              # noqa: BLE001 - a bad case is a result
        return [{'case': case, 'side': side,
                 'load_error': f'{type(exc).__name__}: {exc}'} for side in ('left', 'right')]

    rows = []
    for side, leg in halves.items():
        row = {'case': case, 'side': side, 'spacing_x': spacing[0], 'spacing_z': spacing[2]}
        try:
            image = reflect_image_left_right(leg) if side == 'right' else leg
            hip_image, fhc, r_fh, base = _ct_proximal_landmarks(image, nmdid.FEMUR_LABEL)
            metrics = murphy_neck_base_qc(hip_image, fhc, base, head_radius=r_fh,
                                          segmentation_label=nmdid.FEMUR_LABEL)
            for key in METRICS:
                row[key] = metrics.get(key)
            row['flags'] = metrics['flags']
            row['landmark'] = True
        except Exception as exc:                          # noqa: BLE001 - the failure IS the datum
            row['landmark'] = False
            row['flags'] = _landmark_failure_flag(exc)
            row['failure_reason'] = row['flags'].split(':', 1)[-1]
            row['failure_exception'] = type(exc).__name__
        rows.append(row)
    return rows


def summarise(frame: pd.DataFrame) -> str:
    """Render the survey as markdown: failure reasons, flag rates, axis distribution.

    :param frame: The per-half frame from :func:`survey_case`.
    :return: The report text.
    """
    total = len(frame)
    n_cases = frame['case'].nunique()
    loaded = frame[frame['load_error'].isna()] if 'load_error' in frame else frame
    lines = ['# Murphy femoral-neck QC on whole-leg CT', '',
             'Backlog task #3, regenerated by `scripts/survey_ct_murphy_qc.py`.', '',
             f'Image halves: **{total}** from {n_cases} cases'
             + (f'; {total - len(loaded)} could not be loaded' if len(loaded) != total else ''),
             '']
    if loaded.empty:
        return '\n'.join(lines + ['No half could be measured.'])

    with_landmark = loaded[loaded['landmark'] == True]       # noqa: E712 - NaN-safe
    without = loaded[loaded['landmark'] != True]             # noqa: E712
    lines += [
        f'**{len(with_landmark)} of {len(loaded)} halves yield a Murphy landmark; '
        f'{len(without)} do not.** (MRI, for scale: 1887 of 2128, with 241 failures.)', '']

    if not without.empty and 'failure_reason' in without:
        counts = without['failure_reason'].value_counts()
        lines += ['## Why a landmark is not produced', '',
                  counts.rename('halves').to_frame().to_markdown(), '']
        if 'axis_unresolved' in counts.index:
            lines += [f'**`axis_unresolved` fires on {int(counts["axis_unresolved"])} halves** -- '
                      'this is the hard one-in-plane-voxel raise in `get_femoral_neck_base`, '
                      'which never fired on any of the 1887 MRI halves. It is a real CT failure '
                      'mode, not a hypothetical one.', '']
        else:
            lines += ['The hard one-in-plane-voxel raise (`axis_unresolved`) does **not** fire '
                      'on any half here.', '']

    # NB: ``.flags`` is a pandas DataFrame property, so this column needs bracket access
    flagged = with_landmark[with_landmark['flags'].astype(str) != '']
    lines += ['## Flag rate', '',
              f'**{len(flagged)} of {len(with_landmark)}** measurable halves are flagged '
              f'({100.0 * len(flagged) / max(len(with_landmark), 1):.1f}%). '
              f'MRI: 5 of 1887 (0.3%).', '']
    if not flagged.empty:
        gates = (flagged['flags'].astype(str).str.split(';').explode()
                 .str.split(':').str[0].value_counts())
        lines += [gates.rename('halves').to_frame().to_markdown(), '',
                  flagged[['case', 'side', 'axis_length_mm', 'axis_length_voxels',
                           'head_clearance_mm', 'flags']].to_markdown(index=False), '']

    axis = with_landmark['axis_length_voxels'].dropna()
    lines += ['## Axis length against the threshold', '',
              f'`murphy_axis_short` fires below '
              f'**{AXIS_THRESHOLD_VOXELS:.2f} in-plane voxels** '
              f'(`1 / tan({C.MURPHY_AXIS_ANGLE_TOLERANCE_DEG} deg)`). The threshold is in '
              'voxels, so it shrinks in millimetres as the voxels do -- which is exactly why '
              'its CT behaviour could not be inferred from the MRI cohorts.', '']
    if not axis.empty:
        # Each column gets its OWN quantile: the in-plane spacing varies per case, so the
        # millimetre and voxel orderings are not the same and pairing them row-wise would
        # misreport one of the two. The threshold is in voxels; the mm column is context.
        millimetres = with_landmark['axis_length_mm'].dropna()
        probabilities = [('min', 0.0), ('p1', .01), ('p5', .05), ('p10', .10),
                         ('p25', .25), ('median', .50)]
        lines += ['| quantile | in-plane voxels | mm (own quantile) |', '| --- | ---: | ---: |']
        for label, probability in probabilities:
            lines += [f'| {label} | {axis.quantile(probability):.2f} '
                      f'| {millimetres.quantile(probability):.1f} |']
        lines += ['',
                  f'Halves within one voxel of the threshold '
                  f'({AXIS_THRESHOLD_VOXELS:.2f} +/- 1): '
                  f'**{int(((axis - AXIS_THRESHOLD_VOXELS).abs() <= 1).sum())}** -- a cohort '
                  'straddling the threshold would make the flag rate sensitive to the tolerance; '
                  'one far from it would not.', '']
        for tolerance in (2.5, 3.0, 4.0, 5.0):
            limit = 1.0 / np.tan(np.deg2rad(tolerance))
            lines += [f'* at {tolerance} deg (threshold {limit:.2f} voxels): '
                      f'{int((axis < limit).sum())} halves flagged']
        lines += ['']

    clearance = with_landmark['head_clearance_mm'].dropna()
    if not clearance.empty:
        lines += ['## Head clearance', '',
                  f'Minimum 3-D distance of the neck base outside the fitted head sphere: '
                  f'**{clearance.min():.2f} mm** (median {clearance.median():.2f}). '
                  f'Negative means inside the sphere, which `murphy_base_in_head` flags. '
                  f'MRI minimum was 5.99 mm outside, with one NaKo hip inside.', '']
    return '\n'.join(lines)


def main() -> None:
    """Parse the command line, survey every case and write the three output files.

    :return: Nothing.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('-i', '--inputs', type=Path, nargs='+', required=True,
                        help='Directories (searched recursively) and/or single .nii.gz files.')
    parser.add_argument('-o', '--out-dir', type=Path, required=True, help='Output directory.')
    parser.add_argument('--workers', type=int, default=8, help='Worker processes.')
    parser.add_argument('--limit', type=int, default=None, help='Measure only the first N cases.')
    args = parser.parse_args()

    paths = []
    for item in args.inputs:
        if item.is_dir():
            paths += sorted(str(p) for pattern in ('*.nii', '*.nii.gz')
                            for p in item.rglob(pattern))
        elif item.exists():
            paths.append(str(item))
    if not paths:
        raise SystemExit(f'no .nii/.nii.gz files under {args.inputs}')
    if args.limit:
        paths = paths[:args.limit]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with Pool(args.workers) as pool:
        nested = pool.map(survey_case, paths, chunksize=1)
    frame = pd.DataFrame([row for rows in nested for row in rows])
    frame.to_csv(args.out_dir / 'halves.csv', index=False)
    report = summarise(frame)
    (args.out_dir / 'summary.md').write_text(report + '\n')

    def _git(*cmd):
        try:
            return subprocess.run(('git',) + cmd, cwd=Path(__file__).resolve().parent.parent,
                                  capture_output=True, text=True, check=True).stdout.strip()
        except Exception:                                 # noqa: BLE001 - best effort
            return None

    (args.out_dir / 'summary.meta.json').write_text(json.dumps({
        'commit': _git('rev-parse', 'HEAD'),
        'dirty': bool(_git('status', '--porcelain')),
        'date': datetime.now(timezone.utc).isoformat(),
        'inputs': [str(i) for i in args.inputs],
        'n_cases': len(paths),
        'axis_tolerance_deg': C.MURPHY_AXIS_ANGLE_TOLERANCE_DEG,
        'axis_threshold_voxels': float(AXIS_THRESHOLD_VOXELS),
    }, indent=2) + '\n')
    print(report)


if __name__ == '__main__':
    main()
