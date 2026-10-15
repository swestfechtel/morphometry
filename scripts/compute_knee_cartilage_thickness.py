"""Compute (and optionally visualise) knee cartilage thickness from segmentation(s).

CLI wrapper around :class:`morphometry.cartilage.knee.Tibia` / ``Femur``. It ingests one
or more knee cartilage segmentations (NIfTI files and/or directories), computes
per-subregion tibial and femoral cartilage thickness, prints a summary, and can
optionally:

* write a per-subregion summary CSV (``-o/--output``),
* dump the full per-point thickness maps as JSON (``--dump-json``),
* display the 3D visualisations (``--visualize``) interactively, or save them as PNG
  screenshots (``--screenshot``) and/or self-contained interactive HTML (``--html``).

Single vs. batch: one input keeps flat outputs; multiple inputs (several files, or a
directory searched recursively) are batched into a single combined CSV (with a leading
``subject`` column) and a single combined JSON (keyed by subject). All visualisation
files land flat in the ``--screenshot`` / ``--html`` directories, named
``<subject>_knee_<view>.<ext>``. Each scan's ``subject`` id is its filename stem,
disambiguated by parent directory names when stems collide (e.g.
``P07/1Relaxed/1Relaxed.nii.gz`` -> ``P07_1Relaxed``).

Examples
--------
::

    # single scan
    python scripts/compute_knee_cartilage_thickness.py -i knee_seg.nii.gz \\
        --femur-label 3 --tibia-label 4 --method knn \\
        -o thickness.csv --visualize both --screenshot ./out --html ./out

    # batch: a whole directory tree, combined outputs + all visualisations in one dir
    python scripts/compute_knee_cartilage_thickness.py -i /data/knees \\
        -o results/ --dump-json results/ --screenshot results/viz --html results/viz

The segmentation is expected to label the femoral and tibial cartilage with distinct
integer labels (defaults 3 and 4, matching the Duesseldorf/T1rho convention), and the tibial
bone (``--tibia-bone-label``, default 2): the tibial subregions (Wirth & Eckstein 2008) are
defined on the tibial subchondral bone surface. Tibial subregion keys are patient-side
(``clt`` = central patient-left plate); ``--laterality-table`` (CSV ``subject,laterality``)
adds the paper's medial/lateral names (``cMT`` ... ``pLT``) in an extra CSV column.

Known bad-scan signals are flagged, not fixed or dropped: the last CSV column ``qc_flags``
(and a per-subject ``qc_flags`` entry in the JSON) holds ``;``-separated ``name:value`` flags
from :func:`morphometry.cartilage.knee.scan_qc_flags` -- ``non_lpi_orientation:<codes>`` when
the scan did not land in LPI after reorientation, ``tibia_transfer_far:<fraction>`` when the
tibial region transfer left too much cartilage far from its subchondral bone area. Empty for
a clean scan; the measurements are written as usual either way. A scan for which neither
bone yields a result keeps one placeholder CSV row (empty ``bone`` / ``subregion``, NaN
statistics) so that it, and its flags, stay visible; so does a scan whose processing raises
(unreadable file, uncaught error, rendering failure), with its orientation flag if it loaded.
"""
import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Allow running the script standalone (repo root on the path), matching the other
# scripts in this directory.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from morphometry.image_io import Segmentation
from morphometry.cartilage.knee import (
    Tibia, Femur, plot_knee_segments, plot_knee_thickness, scan_qc_flags,
)
from morphometry.cartilage.tibia_regions import anatomical_tibia_names

# Column order for the summary CSV. The stdout table omits the leading ``subject``
# column (it is printed as a per-subject header instead).
# 'anatomical_subregion' is the paper's medial/lateral name of a tibial subregion (e.g. cMT),
# filled only when the knee's laterality is known (--laterality-table); 'subregion' keeps the
# patient-side key (clt = central patient-LEFT plate) in every row.
# 'qc_flags' is the scan's QC flag string (see scan_qc_flags), repeated on each of its rows;
# appended last so the earlier columns keep their positions.
SUMMARY_COLUMNS = ['subject', 'bone', 'subregion', 'mean_mm', 'std_mm', 'min_mm', 'max_mm', 'n_points',
                   'anatomical_subregion', 'qc_flags']

# Default filenames used when an output path is given as a directory.
DEFAULT_CSV_NAME = 'summary.csv'
DEFAULT_JSON_NAME = 'thickness.json'


def load_segmentation(input_path: str, remove_outliers: bool = False) -> Segmentation:
    """
    Load a knee cartilage segmentation and place it in the LPI orientation the
    thickness code requires.

    :param input_path: Path to the segmentation NIfTI (``.nii`` / ``.nii.gz``).
    :param remove_outliers: Whether to apply :meth:`Segmentation.remove_outliers`
        after loading (drops small disconnected label islands).
    :return: The loaded, LPI-oriented ``Segmentation``.
    """
    seg = Segmentation('nibabel')
    seg.read_image(input_path)
    seg.transform_coordinate_system()
    if remove_outliers:
        seg.remove_outliers()
    return seg


def compute_thickness(seg: Segmentation, femur_label: int, tibia_label: int,
                      method: str, tibia_bone_label: int = 2,
                      laterality: Optional[str] = None) -> Tuple[Optional[Tibia], Optional[Femur], dict, dict]:
    """
    Compute tibial and femoral cartilage thickness for a segmentation.

    The two bones are computed independently and each is wrapped so a failure on one
    (e.g. a mask that cannot be split into two plates) does not abort the other,
    mirroring the per-measurement tolerance used elsewhere in the codebase.

    :param seg: The LPI-oriented knee cartilage ``Segmentation``.
    :param femur_label: The label of the femoral cartilage.
    :param tibia_label: The label of the tibial cartilage.
    :param method: Thickness method, ``'knn'`` or ``'mesh'``.
    :param tibia_bone_label: The label of the tibial bone (the tibial subregions are
        defined on its subchondral surface).
    :param laterality: ``'left'`` / ``'right'`` knee or ``None``; passed to ``Tibia`` (adds
        the medial/lateral region names, the subdivision itself does not depend on it).
    :return: ``(tibia, femur, tibia_results, femur_results)``. ``tibia`` / ``femur`` are
        the computed objects (``None`` if that bone failed), and the results are the
        per-subregion thickness dicts (empty ``{}`` on failure).
    """
    tibia: Optional[Tibia] = None
    femur: Optional[Femur] = None
    tibia_results: dict = {}
    femur_results: dict = {}

    try:
        tibia = Tibia(seg, cartilage_label=tibia_label, bone_label=tibia_bone_label,
                      laterality=laterality)
        tibia_results = tibia.calculate_thickness(method)
    except (RuntimeError, AssertionError, ValueError) as exc:
        print(f'[warning] tibial cartilage thickness failed: {exc}', file=sys.stderr)
        tibia = None

    # The femur is defined relative to the tibial plateau, so it needs a valid Tibia.
    if tibia is not None:
        try:
            femur = Femur(seg, cartilage_label=femur_label)
            femur_results = femur.calculate_thickness(tibia, method)
        except (RuntimeError, AssertionError, ValueError) as exc:
            print(f'[warning] femoral cartilage thickness failed: {exc}', file=sys.stderr)
            femur = None
    else:
        print('[warning] skipping femur: it requires a valid tibia.', file=sys.stderr)

    return tibia, femur, tibia_results, femur_results


def _finite_values(thickness_map: dict) -> np.ndarray:
    """
    Extract the finite (non-``NaN``, non-``None``) thickness values from a per-point map.

    :param thickness_map: ``{(x, y): thickness}`` for one subregion.
    :return: A 1-D array of finite thickness values (possibly empty).
    """
    return np.array(
        [v for v in thickness_map.values() if v is not None and not math.isnan(float(v))],
        dtype=float,
    )


def summarize(results_by_bone: Dict[str, dict], subject: str,
              laterality: Optional[str] = None, qc_flags: str = '') -> List[list]:
    """
    Build per-subregion summary statistics for one subject's bones.

    :param results_by_bone: ``{bone_name: {subregion: {(x, y): thickness}}}``.
    :param subject: Identifier of the scan the results belong to (first CSV column).
    :param laterality: ``'left'`` / ``'right'`` knee or ``None``; with a value, tibial rows
        get the paper's medial/lateral name in the ``anatomical_subregion`` column.
    :param qc_flags: The scan's QC flag string (:func:`scan_qc_flags`), written into the
        ``qc_flags`` column of every row of this subject (``''`` = no flags).
    :return: A list of rows ``[subject, bone, subregion, mean, std, min, max, n_points,
        anatomical_subregion, qc_flags]``, with ``NaN`` statistics for subregions that have
        no finite values and an empty ``anatomical_subregion`` where it is unknown.
    """
    names = anatomical_tibia_names(laterality) or {}
    rows: List[list] = []
    for bone, results in results_by_bone.items():
        for subregion, thickness_map in results.items():
            anatomical = names.get(subregion, '') if bone == 'tibia' else ''
            values = _finite_values(thickness_map)
            if len(values) == 0:
                rows.append([subject, bone, subregion, np.nan, np.nan, np.nan, np.nan, 0, anatomical,
                             qc_flags])
            else:
                rows.append([
                    subject, bone, subregion,
                    float(values.mean()), float(values.std()),
                    float(values.min()), float(values.max()), int(len(values)), anatomical,
                    qc_flags,
                ])
    return rows


def failed_row(subject: str, qc_flags: str = '') -> list:
    """
    The summary row of a scan for which neither bone produced any thickness result.

    :param subject: Identifier of the scan.
    :param qc_flags: The scan's QC flag string (``''`` = no flags).
    :return: ``[subject, '', '', NaN, NaN, NaN, NaN, 0, '', qc_flags]`` (empty ``bone`` and
        ``subregion`` mark the failure), matching :data:`SUMMARY_COLUMNS`.
    """
    return [subject, '', '', np.nan, np.nan, np.nan, np.nan, 0, '', qc_flags]


def load_laterality_table(path: Optional[str]) -> Dict[str, str]:
    """
    Read a per-subject knee laterality table.

    The CSV needs a header with the columns ``subject`` and ``laterality``; ``laterality``
    is ``left`` or ``right`` (case-insensitive; an empty cell means unknown). ``subject``
    is matched against the scan's subject id (its filename stem, see
    :func:`assign_subject_ids`), so a table keyed ``K1_0_belastet`` addresses that file.

    :param path: Path to the CSV, or ``None``.
    :return: ``{subject: 'left' | 'right'}`` (empty without a path).
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
                raise ValueError(f"{path}: laterality of {row['subject']!r} must be left or right, got {value!r}")
            table[row['subject'].strip()] = value
    return table


def print_summary(rows: List[list]):
    """
    Print one subject's per-subregion summary as an aligned table on stdout.

    :param rows: Summary rows for a single subject as produced by :func:`summarize`.
    """
    if not rows:
        print('No thickness results to report.')
        return
    print(f'=== {rows[0][0]} ===')
    header = f"{'bone':<6} {'subregion':<10} {'mean':>8} {'std':>8} {'min':>8} {'max':>8} {'n':>7}"
    print(header)
    print('-' * len(header))
    for row in rows:
        _subject, bone, subregion, mean, std, mn, mx, n = row[:8]
        print(f'{bone:<6} {subregion:<10} {mean:>8.3f} {std:>8.3f} {mn:>8.3f} {mx:>8.3f} {n:>7d}')


def _resolve_output_file(output_path: str, default_name: str) -> str:
    """
    Resolve an output path that may name a file or a directory.

    A path that is an existing directory, or that has no file extension, is treated as a
    directory and ``default_name`` is written inside it; a path with an extension (e.g.
    ``out.csv``) is used verbatim as a file. This keeps the CSV and JSON outputs distinct
    even when both are given the same directory. Parent directories are created.

    :param output_path: The user-supplied output path.
    :param default_name: Filename to use when ``output_path`` denotes a directory.
    :return: The resolved destination file path.
    """
    if os.path.isdir(output_path) or Path(output_path).suffix == '':
        output_path = os.path.join(output_path, default_name)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    return output_path


def write_csv(rows: List[list], output_path: str):
    """
    Write per-subregion summary statistics (one or many subjects) to a CSV file.

    :param rows: Summary rows as produced by :func:`summarize` (may span subjects).
    :param output_path: Destination CSV path, or a directory to write ``summary.csv`` into.
    """
    output_path = _resolve_output_file(output_path, DEFAULT_CSV_NAME)
    with open(output_path, 'w', newline='') as fh:
        writer = csv.writer(fh)
        writer.writerow(SUMMARY_COLUMNS)
        writer.writerows(rows)
    print(f'Wrote summary CSV to {output_path}')


def _results_to_serializable(results: dict) -> dict:
    """
    Convert a per-subregion thickness dict into a JSON-serialisable structure.

    Tuple ``(x, y)`` keys become ``[x, y, thickness]`` triples (thickness ``null`` for
    ``NaN``), avoiding non-string dict keys while preserving the point coordinates.

    :param results: ``{subregion: {(x, y): thickness}}``.
    :return: ``{subregion: [[x, y, thickness_or_null], ...]}``.
    """
    serialisable: dict = {}
    for subregion, thickness_map in results.items():
        points = []
        for (x, y), value in thickness_map.items():
            if value is None or math.isnan(float(value)):
                thickness = None
            else:
                thickness = float(value)
            points.append([float(x), float(y), thickness])
        serialisable[subregion] = points
    return serialisable


def subject_payload(tibia_results: dict, femur_results: dict, qc_flags: str = '') -> dict:
    """
    Build the JSON-serialisable payload of full per-point thickness maps for one subject.

    :param tibia_results: The tibial per-subregion thickness dict.
    :param femur_results: The femoral per-subregion thickness dict.
    :param qc_flags: The scan's QC flag string (:func:`scan_qc_flags`, ``''`` = no flags).
    :return: ``{'tibia': ..., 'femur': ..., 'qc_flags': ...}`` with serialisable point lists.
    """
    return {
        'tibia': _results_to_serializable(tibia_results),
        'femur': _results_to_serializable(femur_results),
        'qc_flags': qc_flags,
    }


def write_json(payload_by_subject: dict, output_path: str):
    """
    Write the full per-point thickness maps, keyed by subject, to a single JSON file.

    :param payload_by_subject: ``{subject: {'tibia': ..., 'femur': ..., 'qc_flags': ...}}``.
    :param output_path: Destination JSON path, or a directory to write ``thickness.json``
        into.
    """
    output_path = _resolve_output_file(output_path, DEFAULT_JSON_NAME)
    with open(output_path, 'w') as fh:
        json.dump(payload_by_subject, fh, indent=2)
    print(f'Wrote full thickness maps to {output_path}')


def visualize(tibia: Optional[Tibia], femur: Optional[Femur], tibia_results: dict,
              femur_results: dict, mode: str, name_prefix: str,
              screenshot_dir: Optional[str] = None, html_dir: Optional[str] = None):
    """
    Render the requested 3D cartilage visualisation(s).

    For each requested view a PyVista scene is built and then, depending on the outputs
    requested, saved as a PNG (``screenshot_dir``) and/or exported as a self-contained
    interactive HTML (``html_dir``). When neither output directory is given, an
    interactive window is opened per view instead. Output files are named
    ``<name_prefix>_knee_<view>.<ext>`` so many subjects can share one directory.

    :param tibia: The computed ``Tibia`` (required to visualise; ``None`` skips).
    :param femur: The computed ``Femur`` (required to visualise; ``None`` skips).
    :param tibia_results: The tibial thickness dict (for the heatmap view).
    :param femur_results: The femoral thickness dict (for the heatmap view).
    :param mode: One of ``'none'``, ``'segments'``, ``'thickness'``, ``'both'``.
    :param name_prefix: Filename stem (typically the subject id) for saved files.
    :param screenshot_dir: If set, save a PNG per view into this directory.
    :param html_dir: If set, export an interactive HTML per view into this directory.
    """
    if mode == 'none':
        if screenshot_dir is not None or html_dir is not None:
            print('[warning] --screenshot/--html given but --visualize is none; nothing to render.',
                  file=sys.stderr)
        return
    if tibia is None or femur is None:
        print('[warning] skipping visualisation: both tibia and femur are required.',
              file=sys.stderr)
        return

    import pyvista as pv

    targets = []
    if mode in ('segments', 'both'):
        targets.append('segments')
    if mode in ('thickness', 'both'):
        targets.append('thickness')

    save = screenshot_dir is not None or html_dir is not None
    if screenshot_dir is not None:
        os.makedirs(screenshot_dir, exist_ok=True)
    if html_dir is not None:
        os.makedirs(html_dir, exist_ok=True)

    for target in targets:
        plotter = pv.Plotter(off_screen=save, window_size=[1000, 800])
        if target == 'segments':
            plot_knee_segments(tibia, femur, plotter=plotter, show=False)
        else:
            plot_knee_thickness(tibia, femur, tibia_results, femur_results,
                                plotter=plotter, show=False)
        if save:
            plotter.camera_position = 'iso'
            if screenshot_dir is not None:
                png_path = Path(screenshot_dir) / f'{name_prefix}_knee_{target}.png'
                plotter.screenshot(str(png_path))
                print(f'Saved {png_path}')
            if html_dir is not None:
                html_path = Path(html_dir) / f'{name_prefix}_knee_{target}.html'
                plotter.export_html(str(html_path))
                print(f'Saved {html_path}')
            plotter.close()
        else:
            plotter.show()


NIFTI_SUFFIXES = ('.nii', '.nii.gz')


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


def resolve_inputs(inputs: List[str]) -> List[str]:
    """
    Expand the given input paths into a sorted, de-duplicated list of NIfTI files.

    Each entry may be a NIfTI file (used directly) or a directory (searched recursively
    for ``*.nii`` / ``*.nii.gz``).

    :param inputs: The raw ``--input`` values (files and/or directories).
    :return: Sorted, de-duplicated list of NIfTI file paths.
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
            print(f'[error] input path not found: {entry}', file=sys.stderr)
            sys.exit(1)
    if not found:
        print('[error] no NIfTI (.nii/.nii.gz) files found in the given input(s).',
              file=sys.stderr)
        sys.exit(1)
    return sorted(found)


def assign_subject_ids(paths: List[str]) -> Dict[str, str]:
    """
    Assign a unique, human-readable subject id to each input path.

    Each id is built from trailing path components ending in the file stem (without the
    NIfTI suffix). Ids start as the stem alone and, wherever several paths would share an
    id, all of the colliding paths grow by one more parent directory — symmetrically —
    until every id is unique. Consecutive duplicate components are collapsed, so a nested
    ``<subject>/1Relaxed/1Relaxed.nii.gz`` layout yields ``<subject>_1Relaxed`` rather than
    ``<subject>_1Relaxed_1Relaxed``.

    :param paths: The resolved (absolute) NIfTI file paths.
    :return: A mapping ``{path: subject_id}`` with distinct ids.
    """
    # Path components with the filename replaced by its NIfTI-stripped stem.
    parts = {p: list(Path(p).parts) for p in paths}
    for comps in parts.values():
        comps[-1] = _strip_nifti_suffix(comps[-1])
    depth = {p: 1 for p in paths}

    def make_id(path: str) -> str:
        comps = parts[path][-depth[path]:]
        collapsed: List[str] = []
        for comp in comps:
            if not collapsed or collapsed[-1] != comp:
                collapsed.append(comp)
        return '_'.join(collapsed)

    ids = {p: make_id(p) for p in paths}
    changed = True
    while changed:
        changed = False
        groups: Dict[str, List[str]] = {}
        for path, subject in ids.items():
            groups.setdefault(subject, []).append(path)
        for group in groups.values():
            if len(group) > 1:
                for path in group:
                    if depth[path] < len(parts[path]):
                        depth[path] += 1
                        ids[path] = make_id(path)
                        changed = True
    return ids


def process_one(path: str, subject: str, args: argparse.Namespace,
                seg: Optional[Segmentation] = None) -> Tuple[List[list], dict]:
    """
    Run the full pipeline for one segmentation and emit its summary + visualisation.

    Visualisation outputs (PNG and/or interactive HTML) are written flat into the
    ``--screenshot`` / ``--html`` directories, named with the ``subject`` prefix so many
    subjects share one directory.

    :param path: Path to the segmentation NIfTI.
    :param subject: The subject id for this scan.
    :param args: Parsed command-line arguments (labels, method, visualise mode, ...).
    :param seg: The scan already loaded by :func:`load_segmentation` (``main`` loads it
        first so a scan that fails later still reports its orientation flag); ``None``
        loads ``path`` here.
    :return: ``(summary_rows, json_payload)`` for this subject.
    """
    if seg is None:
        seg = load_segmentation(path, remove_outliers=args.remove_outliers)
    laterality = args.laterality_by_subject.get(subject)
    tibia, femur, tibia_results, femur_results = compute_thickness(
        seg, femur_label=args.femur_label, tibia_label=args.tibia_label, method=args.method,
        tibia_bone_label=args.tibia_bone_label, laterality=laterality,
    )

    # Flag (do not fix or drop) the known bad-scan signals; the measurements stand as computed.
    qc_flags = scan_qc_flags(seg, tibia)
    if qc_flags:
        print(f'[qc] {subject}: {qc_flags}', file=sys.stderr)

    rows = summarize({'tibia': tibia_results, 'femur': femur_results}, subject, laterality,
                     qc_flags)
    if not rows:
        # Both bones failed: keep the scan visible (and its QC flags in the CSV) as one
        # placeholder row instead of letting it vanish from the summary.
        rows = [failed_row(subject, qc_flags)]
    print_summary(rows)

    # Passing --screenshot/--html signals intent to render; if the user did not also pick
    # a view, default to rendering both rather than silently doing nothing.
    mode = args.visualize
    if (args.screenshot is not None or args.html is not None) and mode == 'none':
        mode = 'both'
    visualize(tibia, femur, tibia_results, femur_results, mode=mode, name_prefix=subject,
              screenshot_dir=args.screenshot, html_dir=args.html)

    return rows, subject_payload(tibia_results, femur_results, qc_flags)


def main(args: argparse.Namespace):
    """
    Run the end-to-end pipeline for one or more segmentations per the parsed CLI args.

    Multiple inputs are batched — one combined CSV (with a ``subject`` column) and one
    combined JSON (keyed by subject). All visualisation outputs (PNG / interactive HTML)
    go flat into the ``--screenshot`` / ``--html`` directories, named ``<subject>_knee_*``.

    :param args: Parsed command-line arguments.
    """
    paths = resolve_inputs(args.input)
    subject_ids = assign_subject_ids(paths)
    args.laterality_by_subject = load_laterality_table(getattr(args, 'laterality_table', None))
    if args.laterality_by_subject:
        unknown = [s for s in subject_ids.values() if s not in args.laterality_by_subject]
        if unknown:
            print(f'[info] no laterality for {len(unknown)} scan(s), e.g. {unknown[0]}; their '
                  'tibial rows keep patient-side names only.', file=sys.stderr)
    if len(paths) > 1:
        print(f'[info] batch mode: {len(paths)} segmentations.', file=sys.stderr)
    if (args.screenshot is not None or args.html is not None) and args.visualize == 'none':
        print('[info] --screenshot/--html given without --visualize; rendering both views.',
              file=sys.stderr)

    all_rows: List[list] = []
    payloads: dict = {}
    for path in paths:
        subject = subject_ids[path]
        seg = None
        try:
            seg = load_segmentation(path, remove_outliers=args.remove_outliers)
            rows, payload = process_one(path, subject, args, seg=seg)
        except Exception as exc:  # keep the batch going if one scan is unreadable/degenerate
            print(f'[error] {subject}: {type(exc).__name__}: {exc}', file=sys.stderr)
            # Keep the scan visible as one placeholder row; a scan that loaded still
            # reports its orientation flag (the tibial flag needs a finished Tibia).
            all_rows.append(failed_row(subject, scan_qc_flags(seg) if seg is not None else ''))
            continue
        all_rows.extend(rows)
        payloads[subject] = payload

    if args.output:
        write_csv(all_rows, args.output)
    if args.dump_json:
        write_json(payloads, args.dump_json)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Compute (and optionally visualise) knee cartilage thickness from a segmentation.',
    )
    parser.add_argument('-i', '--input', required=True, nargs='+',
                        help='One or more knee cartilage segmentation NIfTI files, and/or '
                             'directories (searched recursively for .nii/.nii.gz). Multiple '
                             'inputs are batched into combined CSV/JSON outputs.')
    parser.add_argument('--femur-label', type=int, default=3,
                        help='Label of the femoral cartilage in the segmentation (default: 3).')
    parser.add_argument('--tibia-label', type=int, default=4,
                        help='Label of the tibial cartilage in the segmentation (default: 4).')
    parser.add_argument('--tibia-bone-label', type=int, default=2,
                        help='Label of the tibial bone; the tibial subregions are defined on '
                             'its subchondral surface (default: 2).')
    parser.add_argument('--laterality-table', default=None,
                        help='Optional CSV with columns subject,laterality (left/right knee). '
                             'Adds the medial/lateral tibial subregion names (cMT ... pLT) in '
                             'the anatomical_subregion CSV column; subregion keeps the '
                             'patient-side key.')
    parser.add_argument('--method', choices=['knn', 'mesh'], default='knn',
                        help='Thickness computation method (default: knn).')
    parser.add_argument('-o', '--output', default=None,
                        help='Optional path to write the per-subregion summary CSV.')
    parser.add_argument('--dump-json', default=None,
                        help='Optional path to write the full per-point thickness maps as JSON.')
    parser.add_argument('--visualize', choices=['none', 'segments', 'thickness', 'both'],
                        default='none',
                        help='Show the 3D segment and/or thickness-heatmap visualisation (default: none).')
    parser.add_argument('--screenshot', default=None,
                        help='Render visualisations off-screen and save PNGs (named '
                             '<subject>_knee_<view>.png) into this directory instead of '
                             'opening interactive windows.')
    parser.add_argument('--html', default=None,
                        help='Export interactive, self-contained HTML visualisations (named '
                             '<subject>_knee_<view>.html) into this directory.')
    parser.add_argument('--remove-outliers', action='store_true',
                        help='Apply Segmentation.remove_outliers() after loading.')
    main(parser.parse_args())
