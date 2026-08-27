"""Test whether the thinnest cartilage point migrates posteriorly with knee flexion.

Ingests a directory of knee cartilage segmentations named
``K<subject>_<flexion>_<load>.nii.gz`` (e.g. ``K1_60_belastet.nii.gz``), groups them by
``(subject, load)`` into a series over flexion angle, and, for each series, measures where
the thinnest tibiofemoral cartilage point sits along the femoral articular surface and
whether that position trends posteriorly as flexion increases.

Because the flexion scans are separate acquisitions (the knee is repositioned), raw image
coordinates are not comparable across angles. The posterior position is therefore measured
*intrinsically* on the femur: the arc-length fraction of the thinnest point along the
femoral tibia-facing articular surface at its sagittal slice (0 = anterior/trochlea,
1 = posterior condyle). Per ``(subject, load)`` the trend is summarised with a Spearman
rank correlation and a linear slope (fraction per degree); with only four angles per
series no single test can be significant, so the cohort-level check is a one-sided binomial
sign test over how many series show a posterior trend.

Example
-------
::

    python scripts/analyse_thinnest_point_flexion.py \\
        -i /home/simon/Downloads/dfg_knees_segmentations/dfg_knees/segmentations \\
        -o results/ --screenshot results/viz
"""
import argparse
import csv
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy.stats import spearmanr

from morphometry.image_io import Segmentation
from morphometry.cartilage.knee import Tibia, Femur, find_thinnest_area, plot_thinnest_point

# ``K<subject>_<flexion>_<load>`` with a .nii / .nii.gz extension.
_NAME_RE = re.compile(r'^(K\d+)_(\d+)_([A-Za-z]+)\.nii(?:\.gz)?$')

DETAIL_COLUMNS = ['subject', 'load', 'angle', 'thin_x', 'thin_y',
                  'ap_fraction', 'combined_thickness', 'center_thickness']
SUMMARY_COLUMNS = ['subject', 'load', 'n_angles', 'spearman_rho', 'spearman_p',
                   'slope_per_deg', 'supports_hypothesis']


def parse_filename(name: str) -> Optional[tuple]:
    """
    Parse a ``K<subject>_<flexion>_<load>.nii[.gz]`` filename.

    :param name: The bare filename.
    :return: ``(subject, flexion_angle, load)`` (e.g. ``('K1', 60, 'belastet')``), or
        ``None`` if the name does not match the expected pattern.
    """
    match = _NAME_RE.match(name)
    if match is None:
        return None
    subject, angle, load = match.groups()
    return subject, int(angle), load


def _natural_subject_key(subject: str) -> int:
    """Sort key so K2 precedes K10 (numeric part of the ``K<n>`` subject id)."""
    return int(subject[1:])


def articular_ap_fraction(femur: Femur, point: tuple, spacing) -> float:
    """
    Position of an ``(x, y)`` point along the femoral tibia-facing articular surface.

    At the sagittal slice through ``x``, the femoral cartilage's inferior (tibia-facing)
    surface is taken as the most-inferior voxel per anterior-posterior coordinate, ordered
    anterior->posterior, and its cumulative physical arc length is measured. The returned
    value is the point's fraction along that arc: 0 at the anterior (trochlear) end, 1 at
    the posterior condylar end. This is intrinsic to the femoral surface and therefore
    comparable across the separately-acquired flexion scans.

    :param femur: A ``Femur`` whose ``point_cloud`` is populated.
    :param point: The ``(x, y)`` voxel coordinate to locate.
    :param spacing: The image voxel spacing ``(sx, sy, sz)``.
    :return: The arc-length fraction in ``[0, 1]``, or ``NaN`` if the slice is degenerate.
    """
    x, y = int(point[0]), int(point[1])
    column = femur.point_cloud[femur.point_cloud[:, 0] == x]
    if len(column) < 3:
        return np.nan

    ap = column[:, 1].astype(int)
    unique_ap = np.unique(ap)
    if len(unique_ap) < 2:
        return np.nan
    # Tibia-facing articular surface: the most-inferior (max z) voxel per A-P coordinate.
    inferior_z = np.array([column[ap == a][:, 2].max() for a in unique_ap])

    sy, sz = float(spacing[1]), float(spacing[2])
    segment_lengths = np.sqrt((np.diff(unique_ap) * sy) ** 2 + (np.diff(inferior_z) * sz) ** 2)
    arc = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    total = arc[-1]
    if total == 0:
        return np.nan

    idx = int(np.searchsorted(unique_ap, y))
    idx = min(idx, len(arc) - 1)
    return float(arc[idx] / total)


def process_file(path: str, femur_label: int, tibia_label: int, method: str,
                 want_objects: bool = False) -> dict:
    """
    Compute the thinnest point and its articular-surface A-P fraction for one scan.

    :param path: Path to the segmentation NIfTI.
    :param femur_label: Label of the femoral cartilage.
    :param tibia_label: Label of the tibial cartilage.
    :param method: Thickness method, 'knn' or 'mesh'.
    :param want_objects: If ``True``, also return the ``Tibia``/``Femur`` objects and
        thickness maps (needed for visualisation).
    :return: A dict with ``thin_x``, ``thin_y``, ``ap_fraction``, ``combined_thickness``,
        ``center_thickness`` (and, if requested, ``tibia``/``femur``/``tibia_thickness``/
        ``femur_thickness``/``thinnest``).
    """
    seg = Segmentation('nibabel')
    seg.read_image(path)
    seg.transform_coordinate_system()

    tibia = Tibia(seg, tibia_label)
    tibia_thickness = tibia.calculate_thickness(method)
    femur = Femur(seg, femur_label)
    femur_thickness = femur.calculate_thickness(tibia, method)
    thinnest = find_thinnest_area(tibia_thickness, femur_thickness, seg)

    result = {
        'thin_x': thinnest['point'][0],
        'thin_y': thinnest['point'][1],
        'ap_fraction': articular_ap_fraction(femur, thinnest['point'], seg.spacing),
        'combined_thickness': thinnest['combined_thickness'],
        'center_thickness': thinnest['center_thickness'],
    }
    if want_objects:
        result.update(tibia=tibia, femur=femur, tibia_thickness=tibia_thickness,
                      femur_thickness=femur_thickness, thinnest=thinnest)
    return result


def analyse_series(angles: list, fractions: list) -> dict:
    """
    Summarise the posterior-migration trend of one ``(subject, load)`` series.

    :param angles: Flexion angles (degrees).
    :param fractions: Articular-surface A-P fractions at those angles (``NaN`` allowed).
    :return: A dict with ``n_angles`` (finite points used), ``spearman_rho``,
        ``spearman_p``, ``slope_per_deg`` and ``supports_hypothesis`` (slope > 0). Fields
        are ``NaN``/``None`` when fewer than three finite points are available.
    """
    a = np.array(angles, dtype=float)
    f = np.array(fractions, dtype=float)
    finite = np.isfinite(a) & np.isfinite(f)
    a, f = a[finite], f[finite]

    if len(a) < 3:
        return {'n_angles': int(len(a)), 'spearman_rho': np.nan, 'spearman_p': np.nan,
                'slope_per_deg': np.nan, 'supports_hypothesis': None}

    rho, p = spearmanr(a, f)
    slope = float(np.polyfit(a, f, 1)[0])
    return {'n_angles': int(len(a)), 'spearman_rho': float(rho), 'spearman_p': float(p),
            'slope_per_deg': slope, 'supports_hypothesis': slope > 0}


def _sign_test_greater(successes: int, trials: int) -> float:
    """
    One-sided binomial sign-test p-value: P(X >= successes) under X ~ Binomial(trials, 0.5).

    :param successes: Number of series showing a posterior trend.
    :param trials: Number of series tested.
    :return: The one-sided p-value (1.0 if ``trials`` is 0).
    """
    if trials == 0:
        return 1.0
    try:
        from scipy.stats import binomtest
        return float(binomtest(successes, trials, 0.5, alternative='greater').pvalue)
    except ImportError:                                   # scipy < 1.7
        from scipy.stats import binom_test
        return float(binom_test(successes, trials, 0.5, alternative='greater'))


def visualise_case(case: dict, subject: str, angle: int, load: str,
                   screenshot_dir: Optional[str], html_dir: Optional[str],
                   interactive: bool):
    """
    Render one case's thinnest point, saving/showing per the requested outputs.

    :param case: A ``process_file`` result computed with ``want_objects=True``.
    :param subject: Subject id (for the output filename).
    :param angle: Flexion angle (for the output filename).
    :param load: Load configuration (for the output filename).
    :param screenshot_dir: Directory for PNG screenshots, or ``None``.
    :param html_dir: Directory for interactive HTML, or ``None``.
    :param interactive: Whether to open an interactive window (when no dir is given).
    """
    import pyvista as pv

    stem = f'{subject}_{angle}_{load}'
    save = screenshot_dir is not None or html_dir is not None
    plotter = pv.Plotter(off_screen=save, window_size=[1000, 800])
    plot_thinnest_point(case['tibia'], case['femur'], case['tibia_thickness'],
                        case['femur_thickness'], case['thinnest'], plotter=plotter, show=False)
    if save:
        plotter.camera_position = 'iso'
        if screenshot_dir is not None:
            os.makedirs(screenshot_dir, exist_ok=True)
            path = Path(screenshot_dir) / f'{stem}.png'
            plotter.screenshot(str(path))
            print(f'Saved {path}')
        if html_dir is not None:
            os.makedirs(html_dir, exist_ok=True)
            path = Path(html_dir) / f'{stem}.html'
            plotter.export_html(str(path))
            print(f'Saved {path}')
        plotter.close()
    elif interactive:
        plotter.show()


def print_summary(summaries: list):
    """
    Print the per-series trend table and the cohort-level conclusion.

    :param summaries: List of ``(subject, load, analysis)`` tuples.
    """
    print(f"\n{'subject':<8} {'load':<11} {'n':>2} {'spearman_rho':>13} {'slope/deg':>11} {'posterior?':>11}")
    print('-' * 60)
    supporting = tested = 0
    for subject, load, stats in summaries:
        if stats['supports_hypothesis'] is None:
            print(f'{subject:<8} {load:<11} {stats["n_angles"]:>2}   (insufficient finite points)')
            continue
        tested += 1
        supporting += int(stats['supports_hypothesis'])
        mark = 'yes' if stats['supports_hypothesis'] else 'no'
        print(f'{subject:<8} {load:<11} {stats["n_angles"]:>2} {stats["spearman_rho"]:>13.3f} '
              f'{stats["slope_per_deg"]:>11.5f} {mark:>11}')

    print('-' * 60)
    if tested == 0:
        print('No series had enough data to test.')
        return
    p = _sign_test_greater(supporting, tested)
    print(f'\n{supporting}/{tested} series show a posterior trend (thinnest point moves '
          f'posteriorly with flexion).')
    print(f'One-sided binomial sign test: p = {p:.4f}.')
    print('Note: only 4 flexion angles per series, so per-series correlations cannot reach '
          'significance;\n      the cohort sign test is the meaningful hypothesis check.')


def main(args: argparse.Namespace):
    """
    Run the cohort analysis according to the parsed CLI arguments.

    :param args: Parsed command-line arguments.
    """
    # Group files by (subject, load) -> {angle: path}.
    series: dict = defaultdict(dict)
    for entry in sorted(os.listdir(args.input)):
        parsed = parse_filename(entry)
        if parsed is None:
            continue
        subject, angle, load = parsed
        series[(subject, load)][angle] = os.path.join(args.input, entry)

    if not series:
        print(f'[error] no K<subject>_<angle>_<load> segmentations found in {args.input}',
              file=sys.stderr)
        sys.exit(1)

    want_objects = args.visualize or args.screenshot is not None or args.html is not None
    detail_rows: list = []
    summaries: list = []

    for key in sorted(series, key=lambda k: (_natural_subject_key(k[0]), k[1])):
        subject, load = key
        angles, fractions = [], []
        for angle in sorted(series[key]):
            path = series[key][angle]
            try:
                case = process_file(path, args.femur_label, args.tibia_label, args.method,
                                    want_objects=want_objects)
            # Deliberately broad: a cohort run should survive any single pathological scan
            # (segmentation defects surface as varied exceptions from the thickness pipeline).
            except Exception as exc:
                print(f'[warning] {subject} {angle} {load}: {type(exc).__name__}: {exc}',
                      file=sys.stderr)
                continue
            detail_rows.append([subject, load, angle, case['thin_x'], case['thin_y'],
                                case['ap_fraction'], case['combined_thickness'],
                                case['center_thickness']])
            angles.append(angle)
            fractions.append(case['ap_fraction'])
            if want_objects:
                visualise_case(case, subject, angle, load, args.screenshot, args.html,
                               args.visualize)

        summaries.append((subject, load, analyse_series(angles, fractions)))

    print_summary(summaries)

    if args.output:
        _write_csv(args.output, DETAIL_COLUMNS, detail_rows, 'detail.csv', 'per-case')
        summary_rows = [[s, l, st['n_angles'], st['spearman_rho'], st['spearman_p'],
                         st['slope_per_deg'], st['supports_hypothesis']]
                        for s, l, st in summaries]
        _write_csv(args.output, SUMMARY_COLUMNS, summary_rows, 'summary.csv', 'per-series')


def _write_csv(output_dir: str, columns: list, rows: list, filename: str, label: str):
    """
    Write ``rows`` to ``filename`` inside ``output_dir`` (this analysis emits two CSVs, so
    the output target is always a directory).
    """
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    with open(path, 'w', newline='') as fh:
        writer = csv.writer(fh)
        writer.writerow(columns)
        writer.writerows(rows)
    print(f'Wrote {label} CSV to {path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Test whether the thinnest cartilage point migrates posteriorly with knee flexion.')
    parser.add_argument('-i', '--input', required=True,
                        help='Directory of K<subject>_<angle>_<load>.nii.gz segmentations.')
    parser.add_argument('--femur-label', type=int, default=3,
                        help='Label of the femoral cartilage (default: 3).')
    parser.add_argument('--tibia-label', type=int, default=4,
                        help='Label of the tibial cartilage (default: 4).')
    parser.add_argument('--method', choices=['knn', 'mesh'], default='knn',
                        help='Thickness computation method (default: knn).')
    parser.add_argument('-o', '--output', default=None,
                        help='Directory to write detail.csv (per-case) and summary.csv (per-series).')
    parser.add_argument('--visualize', action='store_true',
                        help='Open an interactive thinnest-point view per case.')
    parser.add_argument('--screenshot', default=None,
                        help='Directory to save a PNG per case (<subject>_<angle>_<load>.png).')
    parser.add_argument('--html', default=None,
                        help='Directory to save an interactive HTML per case.')
    main(parser.parse_args())
