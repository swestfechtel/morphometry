#!/usr/bin/env python3
"""Explain the residual CCD / Lee / Murphy bias against the Augsburg readers.

Backlog task #7. Measures the 50 Augsburg reader-study patients at the current
HEAD and compares them with the two readers' manual values, decomposing the
disagreement rather than only reporting it:

* **inter-reader disagreement** first, because it is the scale every other number
  has to be read against -- a pipeline bias smaller than the two humans' mutual
  disagreement is not evidence of a pipeline defect;
* the **sign convention** (the readers recorded no negative torsion at all, so a
  retrotorted leg disagrees by construction, not by error);
* the **shared** component of the Lee and Murphy bias -- the two methods use the
  same distal condylar line and the same femoral head centre, so the mean of
  their two biases is what those shared landmarks contribute;
* the **method-specific** component -- half the difference of the two biases,
  which is the proximal neck axis and nothing else;
* **left-right agreement within a patient**, which needs no reader at all and
  separates a definitional offset (both legs shifted the same way) from scatter.

Writes ``measured.csv`` (one row per patient), ``bias.csv`` (one row per
measure x side x reader) and ``report.md`` to ``-o``. It never writes outside
that directory -- note that ``scripts/process_augsburg_reader_study.py``
overwrites a file under ``~/Data`` and a figure tree, which this must not do.

    python scripts/analyse_reader_bias.py -o out/reader_bias [--workers 12]

:return: Nothing; the findings are the written files (see
    ``docs/augsburg_reader_bias.md`` for the conclusions drawn from them).
"""
import argparse
import sys
import warnings
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from multiprocessing import Pool
from scipy import stats

from morphometry.image_io import Segmentation, split_image_left_right
from morphometry.measurements.femur import calculate_femoral_torsion
from morphometry.measurements.hip import calculate_ccd

#: Reader-study segmentations, one directory per patient.
DEFAULT_ROOT = Path('/home/simon/Data/Augsburg_large/reader_study_augsburg')
#: The two readers' manual measurements.
DEFAULT_TABLES = Path('/home/simon/Data/Augsburg_large/tables')
READERS = ('felix', 'jonathan')

#: Measurement key -> the readers' column stem. Their columns are PATIENT side.
MEASURES = {'ccd': 'CCD', 'lee': 'AT (Lee)', 'murphy': 'AT (Murphy)'}
SIDE_DE = {'left': 'links', 'right': 'rechts'}

#: The per-measurement exceptions the batch scripts already treat as "no value".
MEASUREMENT_ERRORS = (ValueError, IndexError, RuntimeError, AssertionError)


def measure_patient(args) -> dict:
    """
    Measure CCD, Lee and Murphy torsion on both legs of one reader-study patient.

    The image-left half is the patient's RIGHT leg, so every result is keyed by
    patient side to match the readers' tables.

    :param args: ``(patient_id, root)`` -- the directory name and the cohort root.
    :return: A dict with ``patient`` and, per patient side, ``ccd_*``,
        ``ccdproj_*``, ``lee_*`` and ``murphy_*`` (NaN plus an ``*_err`` entry
        where a measurement failed).
    """
    patient, root = args
    out = {'patient': patient}
    try:
        segs = {}
        for part in ('hip', 'knee'):
            seg = Segmentation('nibabel')
            seg.read_image(str(Path(root) / patient / f'{part}_seg_test.nii.gz'))
            seg.transform_coordinate_system()
            seg.remove_outliers()
            segs[part] = seg
    except FileNotFoundError:
        return out

    # The reader-study pipeline's own slice-aspect convention (see CLAUDE.md on
    # x_ratio): kept identical here so this compares the production numbers.
    x_ratio = abs(segs['hip'].spacing[2]) / 2 * abs(segs['hip'].spacing[0])
    left_hip, right_hip = split_image_left_right(segs['hip'])
    left_knee, right_knee = split_image_left_right(segs['knee'])

    for image_side, hip, knee in (('left', left_hip, left_knee),
                                  ('right', right_hip, right_knee)):
        patient_side = 'right' if image_side == 'left' else 'left'
        for method in ('lee', 'murphy'):
            try:
                out[f'{method}_{patient_side}'] = calculate_femoral_torsion(
                    hip, knee, side=image_side, method=method, x_ratio=x_ratio)
            except MEASUREMENT_ERRORS as exc:
                out[f'{method}_{patient_side}'] = np.nan
                out[f'{method}_{patient_side}_err'] = type(exc).__name__
        try:
            actual, projected = calculate_ccd(hip, knee, image_side, 1, False, x_ratio)
            out[f'ccd_{patient_side}'] = actual
            out[f'ccdproj_{patient_side}'] = projected
        except MEASUREMENT_ERRORS as exc:
            out[f'ccd_{patient_side}'] = np.nan
            out[f'ccdproj_{patient_side}'] = np.nan
            out[f'ccd_{patient_side}_err'] = type(exc).__name__
    return out


def load_readers(tables: Path) -> dict:
    """
    Load the two readers' tables, dropping their trailing mean / SD summary rows.

    :param tables: Directory holding ``reference_measurements_<reader>.xlsx``.
    :return: ``{reader: DataFrame indexed by patient}``.
    """
    readers = {}
    for name in READERS:
        frame = pd.read_excel(tables / f'reference_measurements_{name}.xlsx')
        frame = frame.rename(columns={'Unnamed: 0': 'patient'})
        # The last two rows carry the column means and SDs, not patients.
        readers[name] = frame[frame['patient'].notna()].set_index('patient')
    return readers


def reader_means(readers: dict) -> dict:
    """
    The per-leg mean of the two readers, used as a single reference.

    :param readers: As returned by :func:`load_readers`.
    :return: ``{(measure_key, patient_side): Series indexed by patient}``.
    """
    means = {}
    for side, german in SIDE_DE.items():
        for key, stem in MEASURES.items():
            columns = [r[f'{stem} {german}'] for r in readers.values()
                       if f'{stem} {german}' in r.columns]
            means[(key, side)] = pd.concat(columns, axis=1).mean(axis=1)
    return means


def bias_table(measured: pd.DataFrame, readers: dict) -> pd.DataFrame:
    """
    Pipeline-minus-reader bias for every measure x side x reader.

    Also carries the unsigned comparison (``*_abs``), because the readers
    recorded no negative torsion: see ``docs/augsburg_reader_bias.md``.

    :param measured: The per-patient pipeline values.
    :param readers: As returned by :func:`load_readers`.
    :return: One row per measure x side x reader.
    """
    rows = []
    for key, stem in MEASURES.items():
        for side, german in SIDE_DE.items():
            mine = measured[f'{key}_{side}']
            for name, frame in readers.items():
                column = f'{stem} {german}'
                if column not in frame.columns:
                    continue
                both = pd.concat([mine.rename('pipe'),
                                  frame[column].rename('read')], axis=1).dropna()
                if len(both) < 5:
                    continue
                signed = both['pipe'] - both['read']
                unsigned = both['pipe'].abs() - both['read'].abs()
                # scipy raises when every difference is zero (nothing to rank);
                # perfect agreement is "no evidence of a shift", i.e. p = 1.
                p_value = 1.0 if (signed == 0).all() else stats.wilcoxon(signed).pvalue
                rows.append(dict(
                    measure=stem, side=side, reader=name, n=len(both),
                    pipe_mean=both['pipe'].mean(), read_mean=both['read'].mean(),
                    bias=signed.mean(), bias_median=signed.median(),
                    sd=signed.std(), mad=signed.abs().mean(),
                    wilcoxon_p=p_value,
                    r=both['pipe'].corr(both['read']),
                    bias_unsigned=unsigned.mean(),
                    pipe_negative=int((both['pipe'] < 0).sum()),
                    read_negative=int((both['read'] < 0).sum()),
                ))
    return pd.DataFrame(rows)


def write_report(path: Path, measured: pd.DataFrame, readers: dict,
                 bias: pd.DataFrame) -> None:
    """
    Write the human-readable analysis: the four decompositions, in order.

    :param path: The ``report.md`` to write.
    :param measured: The per-patient pipeline values.
    :param readers: As returned by :func:`load_readers`.
    :param bias: As returned by :func:`bias_table`.
    :return: Nothing; the report is written to ``path``.
    """
    means = reader_means(readers)
    lines = ['# Pipeline vs the Augsburg readers', '',
             f'{len(measured)} patients, both legs.', '',
             '## 0. Inter-reader disagreement (the scale everything else is read against)',
             '', '| measure | side | n | felix-jonathan | mean abs diff | r |',
             '| --- | --- | ---: | ---: | ---: | ---: |']
    for stem in MEASURES.values():
        for side, german in SIDE_DE.items():
            a = readers['felix'].get(f'{stem} {german}')
            b = readers['jonathan'].get(f'{stem} {german}')
            if a is None or b is None:
                continue
            both = pd.concat([a.rename('a'), b.rename('b')], axis=1).dropna()
            diff = both['a'] - both['b']
            lines.append(f'| {stem} | {side} | {len(both)} | {diff.mean():+.2f} | '
                         f'{diff.abs().mean():.2f} | {both["a"].corr(both["b"]):+.3f} |')

    lines += ['', '## 1. Bias against each reader', '',
              bias.round(2).to_markdown(index=False), '',
              '## 2. Lee: how much of the bias is the sign convention?', '']
    for side in SIDE_DE:
        both = pd.concat([measured[f'lee_{side}'].rename('p'),
                          means[('lee', side)].rename('r')], axis=1).dropna()
        negative = both['p'] < 0
        lines.append(
            f'* {side}: all n={len(both)} bias {(both.p - both.r).mean():+.2f}; '
            f'pipeline > 0 n={(~negative).sum()} bias '
            f'{(both.p[~negative] - both.r[~negative]).mean():+.2f} '
            f'(no sign ambiguity); pipeline < 0 n={negative.sum()} signed bias '
            f'{(both.p[negative] - both.r[negative]).mean():+.2f} but unsigned '
            f'{(both.p[negative].abs() - both.r[negative]).mean():+.2f}')

    lines += ['', '## 3. Shared vs method-specific (legs where both methods resolve)', '',
              'Lee and Murphy share the distal condylar line and the head centre, so the',
              'mean of their two biases is the shared landmarks and half their difference',
              'is the proximal neck axis. Lee is taken unsigned so the convention of',
              'section 2 cannot contaminate the split.', '']
    for side in SIDE_DE:
        frame = pd.concat([measured[f'lee_{side}'].rename('pl'),
                           measured[f'murphy_{side}'].rename('pm'),
                           means[('lee', side)].rename('rl'),
                           means[('murphy', side)].rename('rm')], axis=1).dropna()
        lee_bias = frame.pl.abs() - frame.rl
        murphy_bias = frame.pm - frame.rm
        shared = (lee_bias + murphy_bias) / 2
        specific = (murphy_bias - lee_bias) / 2
        lines.append(f'* {side}: n={len(frame)}, shared {shared.mean():+.2f} '
                     f'(sd {shared.std():.2f}), method-specific +-{specific.mean():.2f}')

    lines += ['', '## 4. Left-right agreement within a patient (no reader involved)', '',
              '| source | measure | n | Pearson | mean abs L-R |',
              '| --- | --- | ---: | ---: | ---: |']
    sources = [('pipeline', lambda k: (measured[f'{k}_left'], measured[f'{k}_right']))]
    sources += [(name, lambda k, f=frame: (f[f'{MEASURES[k]} links'],
                                           f[f'{MEASURES[k]} rechts']))
                for name, frame in readers.items()]
    for label, getter in sources:
        for key in MEASURES:
            left, right = getter(key)
            both = pd.concat([left.rename('L'), right.rename('R')], axis=1).dropna()
            if len(both) < 5:
                continue
            lines.append(f'| {label} | {MEASURES[key]} | {len(both)} | '
                         f'{both.L.corr(both.R):+.3f} | {(both.L - both.R).abs().mean():.2f} |')

    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'wrote {path}')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-o', '--out', type=Path, required=True,
                        help='output directory (created; nothing is written elsewhere)')
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT,
                        help=f'reader-study segmentations (default: {DEFAULT_ROOT})')
    parser.add_argument('--tables', type=Path, default=DEFAULT_TABLES,
                        help=f'readers\' xlsx directory (default: {DEFAULT_TABLES})')
    parser.add_argument('--workers', type=int, default=12, help='parallel processes')
    args = parser.parse_args()

    warnings.filterwarnings('ignore')
    args.out.mkdir(parents=True, exist_ok=True)
    patients = sorted(p.name for p in args.root.iterdir() if p.is_dir())
    print(f'{len(patients)} patients')
    with Pool(args.workers) as pool:
        rows = pool.map(measure_patient, [(p, args.root) for p in patients])

    measured = pd.DataFrame(rows).set_index('patient')
    measured.to_csv(args.out / 'measured.csv')
    readers = load_readers(args.tables)
    bias = bias_table(measured, readers)
    bias.to_csv(args.out / 'bias.csv', index=False)
    write_report(args.out / 'report.md', measured, readers, bias)


if __name__ == '__main__':
    main()
