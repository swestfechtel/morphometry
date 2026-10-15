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
1 = posterior condyle).

The thinnest point is located per **compartment**: ``both`` (the whole knee, one point, as
:func:`find_thinnest_area` finds it), ``left`` and ``right`` (the patient-left and
patient-right tibiofemoral compartments, searched separately on the ``*LT`` / ``*RT``
tibial subregions, see :func:`split_tibia_thickness_by_compartment`). **'left'/'right' is the patient side (from the NIfTI header after LPI
reorientation), not medial/lateral**: patient-left is medial in a right knee and lateral in
a left knee, so a per-side sign test reads as a medial or lateral result only if every knee
in the cohort is the same side. A compartment whose search fails on a scan is written as NaN
for that scan (with a warning) without discarding the other compartments; a scan whose
thickness pipeline fails is written as NaN for all three compartments, so ``detail.csv``
has exactly three rows per parsed file.

Per ``(subject, load, compartment)`` the trend is summarised with a Spearman rank
correlation and a linear slope (fraction per degree); with at most four angles per series
no single test can be significant, so the cohort-level check is a one-sided binomial sign
test over how many series show a posterior trend, run separately for each compartment.
These per-compartment sign tests are separate, uncorrected tests, and the two load series
of one subject are not independent.

Outputs (``-o <dir>``): ``detail.csv`` in long format, one row per (scan, compartment);
``summary.csv``, one row per (subject, load, compartment).

Known bad-scan signals are flagged, not fixed or dropped: ``detail.csv``'s last column
``qc_flags`` holds the scan's ``;``-separated ``name:value`` flags from
:func:`morphometry.cartilage.knee.scan_qc_flags` (``non_lpi_orientation:<codes>`` when the scan
did not land in LPI after reorientation, ``tibia_transfer_far:<fraction>`` when the tibial
region transfer left too much cartilage far from its subchondral bone area; empty for a clean
scan), and ``summary.csv``'s last column ``qc_flagged_angles`` lists the flexion angles of the
series' flagged scans (``;``-separated), so a trend resting on a flagged scan stays visible.
Flagged scans are measured and enter the statistics as usual. Visualisations show one scene
per scan with an arrow per compartment (``left`` red, ``right`` blue, ``both`` black).

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
from morphometry.cartilage.knee import (
    Tibia, Femur, find_thinnest_area, split_tibia_thickness_by_compartment, plot_thinnest_point,
    scan_qc_flags,
)

# ``K<subject>_<flexion>_<load>`` with a .nii / .nii.gz extension.
_NAME_RE = re.compile(r'^(K\d+)_(\d+)_([A-Za-z]+)\.nii(?:\.gz)?$')

# 'both' = whole knee; 'left' / 'right' = patient-side tibiofemoral compartments.
COMPARTMENTS = ('both', 'left', 'right')

# Per-compartment measurement fields, in detail.csv column order.
MEASUREMENT_FIELDS = ['thin_x', 'thin_y', 'ap_fraction', 'combined_thickness',
                      'center_thickness']
# The QC columns are appended last so the earlier columns keep their positions: 'qc_flags'
# is the scan's flag string (scan_qc_flags), 'qc_flagged_angles' the angles of a series'
# flagged scans.
DETAIL_COLUMNS = ['subject', 'load', 'angle', 'compartment'] + MEASUREMENT_FIELDS + ['qc_flags']
SUMMARY_COLUMNS = ['subject', 'load', 'compartment', 'n_angles', 'spearman_rho',
                   'spearman_p', 'slope_per_deg', 'supports_hypothesis', 'qc_flagged_angles']


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


def locate_thinnest_points(tibia_thickness: dict, femur_thickness: dict, image,
                           label: str = '') -> dict:
    """
    Find the thinnest cartilage area for the whole knee and for each patient-side compartment.

    Each compartment is searched independently with :func:`find_thinnest_area` (``both`` on
    the full tibial map, ``left`` / ``right`` on the tibial subregions of that side against
    the full femoral map), so a failure in one compartment (e.g. a plateau with no
    femoral overlap) does not discard the others.

    :param tibia_thickness: Per-subregion tibial thickness dict.
    :param femur_thickness: Per-subregion femoral thickness dict.
    :param image: The segmentation image (passed through to :func:`find_thinnest_area`).
    :param label: Scan identifier prefixed to failure warnings (e.g. ``'K1 30 belastet'``).
    :return: ``{compartment: result or None}`` for every entry of :data:`COMPARTMENTS`,
        where ``result`` is a :func:`find_thinnest_area` dict and ``None`` marks a failed
        compartment (a warning naming the scan and compartment is printed to stderr).
    """
    subsets = {'both': tibia_thickness, **split_tibia_thickness_by_compartment(tibia_thickness)}
    found = {}
    for compartment in COMPARTMENTS:
        try:
            found[compartment] = find_thinnest_area(subsets[compartment], femur_thickness, image)
        # Deliberately broad, like the per-scan guard in main(): one bad compartment must
        # not abort the scan's other compartments.
        except Exception as exc:
            print(f'[warning] {label} compartment {compartment}: {type(exc).__name__}: {exc}',
                  file=sys.stderr)
            found[compartment] = None
    return found


def compartment_row(femur: Optional[Femur], thinnest: Optional[dict], spacing) -> dict:
    """
    Turn one compartment's thinnest-area result into its measurement fields.

    :param femur: The scan's ``Femur`` (for the articular A-P fraction); unused if
        ``thinnest`` is ``None``.
    :param thinnest: A :func:`find_thinnest_area` result, or ``None`` for a failed compartment.
    :param spacing: The image voxel spacing ``(sx, sy, sz)``.
    :return: ``{field: value}`` for every entry of :data:`MEASUREMENT_FIELDS`; all ``NaN``
        when ``thinnest`` is ``None``.
    """
    if thinnest is None:
        return {field: np.nan for field in MEASUREMENT_FIELDS}
    return {
        'thin_x': thinnest['point'][0],
        'thin_y': thinnest['point'][1],
        'ap_fraction': articular_ap_fraction(femur, thinnest['point'], spacing),
        'combined_thickness': thinnest['combined_thickness'],
        'center_thickness': thinnest['center_thickness'],
    }


def load_scan(path: str) -> Segmentation:
    """
    Load one knee segmentation and reorient it to LPI.

    Kept separate from :func:`process_file` so that the orientation QC flag of a scan whose
    thickness pipeline later fails can still be reported (see :func:`main`).

    :param path: Path to the segmentation NIfTI.
    :return: The reoriented ``Segmentation`` (check ``orientation_ok`` for whether it is LPI).
    """
    seg = Segmentation('nibabel')
    seg.read_image(path)
    seg.transform_coordinate_system()
    return seg


def process_file(path: str, femur_label: int, tibia_label: int, method: str,
                 want_objects: bool = False, label: str = '', tibia_bone_label: int = 2,
                 seg: Optional[Segmentation] = None) -> dict:
    """
    Compute the per-compartment thinnest points and their A-P fractions for one scan.

    :param path: Path to the segmentation NIfTI.
    :param femur_label: Label of the femoral cartilage.
    :param tibia_label: Label of the tibial cartilage.
    :param method: Thickness method, 'knn' or 'mesh'.
    :param want_objects: If ``True``, also return the ``Tibia``/``Femur`` objects and
        thickness maps (needed for visualisation).
    :param label: Scan identifier used in per-compartment failure warnings.
    :param tibia_bone_label: Label of the tibial bone; the tibial subregions (and hence the
        patient-side compartments) are defined on its subchondral surface.
    :param seg: The scan already loaded by :func:`load_scan`; ``None`` loads ``path``.
    :return: A dict with ``compartments``: ``{compartment: row}`` for every entry of
        :data:`COMPARTMENTS`, each row as returned by :func:`compartment_row` (all ``NaN``
        for a failed compartment), and ``qc_flags``, the scan's :func:`scan_qc_flags`
        string (``''`` = clean). If requested, also ``tibia``/``femur``/
        ``tibia_thickness``/``femur_thickness`` and ``thinnest`` (``{compartment: result}``
        for the compartments that succeeded).
    :raises Exception: Failures of the thickness pipeline itself (which affect every
        compartment) propagate to the caller.
    """
    if seg is None:
        seg = load_scan(path)

    tibia = Tibia(seg, tibia_label, bone_label=tibia_bone_label)
    tibia_thickness = tibia.calculate_thickness(method)
    femur = Femur(seg, femur_label)
    femur_thickness = femur.calculate_thickness(tibia, method)
    thinnest = locate_thinnest_points(tibia_thickness, femur_thickness, seg, label=label)

    result = {'compartments': {c: compartment_row(femur, thinnest[c], seg.spacing)
                               for c in COMPARTMENTS},
              # flagged, not fixed: the measurements above stand as computed
              'qc_flags': scan_qc_flags(seg, tibia)}
    if want_objects:
        result.update(tibia=tibia, femur=femur, tibia_thickness=tibia_thickness,
                      femur_thickness=femur_thickness,
                      thinnest={c: t for c, t in thinnest.items() if t is not None})
    return result


def failed_case(qc_flags: str = '') -> dict:
    """
    A placeholder :func:`process_file` result for a scan whose thickness pipeline failed.

    Written out so that a failed scan stays visible in ``detail.csv`` (one all-NaN row per
    compartment) instead of silently vanishing; :func:`analyse_series` drops the NaNs.

    :param qc_flags: QC flags still known for the scan (e.g. its orientation flag when it
        loaded but the thickness pipeline failed); ``''`` if none or unknown.
    :return: ``{'compartments': {compartment: all-NaN row}, 'qc_flags': qc_flags}``.
    """
    return {'compartments': {c: compartment_row(None, None, None) for c in COMPARTMENTS},
            'qc_flags': qc_flags}


def detail_rows(subject: str, load: str, angle: int, case: dict) -> list:
    """
    Long-format ``detail.csv`` rows for one scan: one row per compartment.

    :param subject: Subject id.
    :param load: Load configuration.
    :param angle: Flexion angle (degrees).
    :param case: A :func:`process_file` result (its ``qc_flags``, a per-scan value, is
        repeated on every compartment row; missing counts as ``''``).
    :return: A list of rows (in :data:`COMPARTMENTS` order), each matching
        :data:`DETAIL_COLUMNS`.
    """
    qc_flags = case.get('qc_flags', '')
    return [[subject, load, angle, compartment]
            + [case['compartments'][compartment][field] for field in MEASUREMENT_FIELDS]
            + [qc_flags]
            for compartment in COMPARTMENTS]


def analyse_series(angles: list, fractions: list) -> dict:
    """
    Summarise the posterior-migration trend of one ``(subject, load)`` series.

    :param angles: Flexion angles (degrees).
    :param fractions: Articular-surface A-P fractions at those angles (``NaN`` allowed).
    :return: A dict with ``n_angles`` (finite points used), ``spearman_rho``,
        ``spearman_p``, ``slope_per_deg`` and ``supports_hypothesis`` (slope > 0; a slope of
        exactly 0 counts as *not* supporting rather than being excluded as a tie). Fields
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
    Render one case's thinnest points, saving/showing per the requested outputs.

    All successfully located compartments are drawn into one scene, one labelled arrow
    each (``left`` red, ``right`` blue, ``both`` black).

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


def sign_test_by_compartment(summaries: list) -> dict:
    """
    Run the cohort-level posterior-trend sign test separately for each compartment.

    :param summaries: List of ``(subject, load, compartment, analysis)`` tuples, where
        ``analysis`` is an :func:`analyse_series` result. Series whose
        ``supports_hypothesis`` is ``None`` (too few finite points) are not counted.
    :return: ``{compartment: (supporting, tested, p)}`` for every compartment present in
        ``summaries``: the number of series with a posterior trend, the number of testable
        series, and the one-sided binomial sign-test p-value.
    """
    counts: dict = {}
    for _subject, _load, compartment, stats in summaries:
        supporting, tested = counts.get(compartment, (0, 0))
        if stats['supports_hypothesis'] is not None:
            supporting += int(stats['supports_hypothesis'])
            tested += 1
        counts[compartment] = (supporting, tested)
    return {c: (s, t, _sign_test_greater(s, t)) for c, (s, t) in counts.items()}


def print_summary(summaries: list):
    """
    Print the per-series trend table and the cohort-level conclusion, per compartment.

    :param summaries: List of ``(subject, load, compartment, analysis)`` tuples.
    """
    tests = sign_test_by_compartment(summaries)
    for compartment in COMPARTMENTS:
        rows = [(s, l, st) for s, l, c, st in summaries if c == compartment]
        if not rows:
            continue
        print(f'\n=== compartment: {compartment} ===')
        print(f"{'subject':<8} {'load':<11} {'n':>2} {'spearman_rho':>13} {'slope/deg':>11} {'posterior?':>11}")
        print('-' * 60)
        for subject, load, stats in rows:
            if stats['supports_hypothesis'] is None:
                print(f'{subject:<8} {load:<11} {stats["n_angles"]:>2}   (insufficient finite points)')
                continue
            mark = 'yes' if stats['supports_hypothesis'] else 'no'
            print(f'{subject:<8} {load:<11} {stats["n_angles"]:>2} {stats["spearman_rho"]:>13.3f} '
                  f'{stats["slope_per_deg"]:>11.5f} {mark:>11}')
        print('-' * 60)

        supporting, tested, p = tests[compartment]
        if tested == 0:
            print('No series had enough data to test.')
            continue
        print(f'{supporting}/{tested} series show a posterior trend (thinnest point moves '
              f'posteriorly with flexion).')
        print(f'One-sided binomial sign test: p = {p:.4f}.')

    print('\nNotes:')
    print('- At most 4 flexion angles per series, so per-series correlations cannot reach '
          'significance;\n  the per-compartment cohort sign tests are the meaningful '
          'hypothesis checks.')
    print("- 'left'/'right' is the PATIENT side (NIfTI header after LPI reorientation). "
          'Patient-left is medial in a right knee\n  and lateral in a left knee, so a per-side '
          'test reads as medial/lateral only if every knee in the cohort is the same side.')
    print('- The per-compartment sign tests are separate, uncorrected tests (no multiplicity '
          'correction), and the\n  loaded/unloaded series of one subject are not independent.')


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
    all_detail_rows: list = []
    summaries: list = []

    for key in sorted(series, key=lambda k: (_natural_subject_key(k[0]), k[1])):
        subject, load = key
        # One (angles, fractions) series per compartment, over this (subject, load).
        angles: list = []
        flagged_angles: list = []
        fractions: dict = {c: [] for c in COMPARTMENTS}
        for angle in sorted(series[key]):
            path = series[key][angle]
            label = f'{subject} {angle} {load}'
            seg = None
            try:
                seg = load_scan(path)
                case = process_file(path, args.femur_label, args.tibia_label, args.method,
                                    want_objects=want_objects, label=label,
                                    tibia_bone_label=args.tibia_bone_label, seg=seg)
                want_render = want_objects
            # Deliberately broad: a cohort run should survive any single pathological scan
            # (segmentation defects surface as varied exceptions from the thickness pipeline).
            # The scan is kept as all-NaN rows so it stays visible in detail.csv.
            except Exception as exc:
                print(f'[warning] {label}: {type(exc).__name__}: {exc} '
                      f'(all compartments written as NaN)', file=sys.stderr)
                # a scan that loaded still reports its orientation flag
                case = failed_case(scan_qc_flags(seg) if seg is not None else '')
                want_render = False
            if case['qc_flags']:
                print(f"[qc] {label}: {case['qc_flags']}", file=sys.stderr)
                flagged_angles.append(angle)
            all_detail_rows.extend(detail_rows(subject, load, angle, case))
            angles.append(angle)
            for compartment in COMPARTMENTS:
                fractions[compartment].append(case['compartments'][compartment]['ap_fraction'])
            if want_render:
                visualise_case(case, subject, angle, load, args.screenshot, args.html,
                               args.visualize)

        for compartment in COMPARTMENTS:
            stats = analyse_series(angles, fractions[compartment])
            # QC bookkeeping, not a statistic: which of this series' scans carry a QC flag
            stats['qc_flagged_angles'] = ';'.join(str(a) for a in flagged_angles)
            summaries.append((subject, load, compartment, stats))

    print_summary(summaries)

    if args.output:
        _write_csv(args.output, DETAIL_COLUMNS, all_detail_rows, 'detail.csv',
                   'per-(scan, compartment)')
        _write_csv(args.output, SUMMARY_COLUMNS, summary_rows(summaries), 'summary.csv',
                   'per-(series, compartment)')


def summary_rows(summaries: list) -> list:
    """
    ``summary.csv`` rows: one per ``(subject, load, compartment)`` series.

    Series with too few finite angles are kept (NaN statistics, ``supports_hypothesis``
    ``None``) so that they remain visible in the output.

    :param summaries: List of ``(subject, load, compartment, analysis)`` tuples; ``analysis``
        may carry ``qc_flagged_angles`` (the ``;``-joined angles of the series' QC-flagged
        scans, set by :func:`main`; missing counts as ``''``).
    :return: A list of rows matching :data:`SUMMARY_COLUMNS`.
    """
    return [[s, l, c, st['n_angles'], st['spearman_rho'], st['spearman_p'],
             st['slope_per_deg'], st['supports_hypothesis'], st.get('qc_flagged_angles', '')]
            for s, l, c, st in summaries]


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
    parser.add_argument('--tibia-bone-label', type=int, default=2,
                        help='Label of the tibial bone; the tibial subregions are defined on '
                             'its subchondral surface (default: 2).')
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
