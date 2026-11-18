"""Measure the held-out NMDID annotations against the pipeline, for choosing (set A) or quoting (set B) thresholds.

The NMDID outlier review (``docs/evidence/nmdid_outlier_review.md``) left three threshold defects open
that must be chosen on held-out legs, never on the 84 reviewed cases:

* **#4** the 40 mm lower bound of the lesser-trochanter distance test (``femur.py::_find_trochanter_minor``);
* **#6** the hip crop as 30 % of the visible femur (``femur.py::_ct_proximal_landmarks``), to become a
  fixed distance below the femur top;
* **#7** the knee crop ending at the fibula top (``image_io.py::split_ct_image``), to become
  ``max(fibula top, tibia top + X mm)``.

The user annotated 40 random cases (``docs/evidence/nmdid_annotation/``): three lesser-trochanter
points and the two posterior tibial-condyle points per leg. This script prepares every leg exactly
as ``scripts/process_nmdid.py`` does (``load_case`` + ``split_legs``; a right image half is
reflected, as every CT landmark does) and writes one row per leg with:

* femur: the head centre (production fit), the femur top, the in-plane-ignoring (x, z) distance from
  the head centre to each LT point measured exactly as the search's plausibility test measures it,
  the crop length needed to reach ``LT_inferior``, the current 30 % crop length, and the production
  Murphy neck-base level against the annotated ``LT_superior`` level;
* tibia: the tibia top, the fibula top, the annotated plateau level, and -- for the legacy rule
  (fibula top), the production rule (``split_ct_image`` as it is now) and every candidate ``X`` --
  the tibial reference layer the condylar line is read on and the
  angle between that line and the annotated posterior condylar line.

**Discipline.** Run with ``--set A`` to choose (done 2026-11-17: #7 -> 15 mm; #4 and #6 left
unchanged, since the bounds never bit on set A); ``--set B`` only once the thresholds are frozen in the
code, and quote B without re-tuning. The script refuses ``--set B`` unless ``--quote`` is given, so
looking at B is never an accident.

Annotation files are 3D Slicer markups (``<case>_<R|L>.mrk.json``, LPS). ``R``/``L`` is the
**patient** side: R lies in the image-left half, L in the image-right half (checked on all 80 legs,
2026-11-17).

Usage::

    python scripts/analyse_nmdid_annotations.py --set A -o results/nmdid_annotation_A.csv
"""
import argparse
import csv
import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from morphometry.femur import _ct_proximal_landmarks  # noqa: E402
from morphometry.hip import get_femoral_head_center_ct  # noqa: E402
from morphometry.image_io import reflect_image_left_right, split_ct_image  # noqa: E402
from morphometry.knee import get_knee_reference_line  # noqa: E402
from scripts.process_nmdid import (FEMUR_LABEL, FIBULA_LABEL, PATELLA_LABEL, PELVIS_LABEL, TIBIA_LABEL,  # noqa: E402
                                   load_case, split_legs)

NMDID = Path('/home/simon/sshfs/hpcproject/workspace_simon/nmdid')
SETS_CSV = Path(__file__).resolve().parents[1] / 'docs' / 'evidence' / 'nmdid_annotation' / 'annotation_sets.csv'
LT_LABELS = ('LT_superior', 'LT_tip', 'LT_inferior')
PATIENT_TO_IMAGE = {'R': 'left', 'L': 'right'}

#: Candidate knee-crop extensions below the tibia top, in mm (#7). A grid, not a fit: the choice is
#: made by reading the set-A table, and recorded with its reason.
KNEE_EXTENSIONS_MM = (0, 6, 9, 12, 15, 18, 21, 24, 30, 36, 45, 60)

log = logging.getLogger('analyse_nmdid_annotations')


def read_points(path: Path) -> dict:
    """Read one leg's markups file.

    :param path: The ``.mrk.json`` file.
    :return: ``{label: RAS world point (3,)}`` for placed points, and ``{label: None}`` for unplaced
        ones (their description, the annotator's reason, is logged).
    """
    markups = json.loads(path.read_text())['markups'][0]
    flip = np.array([-1., -1., 1.]) if markups.get('coordinateSystem', 'LPS') == 'LPS' else np.ones(3)
    points = {}
    for p in markups['controlPoints']:
        if p.get('positionStatus') == 'defined':
            points[p['label']] = np.asarray(p['position'], float) * flip
        else:
            points[p['label']] = None
            log.info('%s %s unplaced: %s', path.name, p['label'], p.get('description', ''))
    return points


def to_left_frame_index(leg, world: np.ndarray) -> np.ndarray:
    """Map an RAS world point into the continuous voxel index of the left-frame leg.

    :param leg: The leg Segmentation as split (not reflected).
    :param world: The RAS world point.
    :return: The continuous index in ``leg``; the caller reflects x for a right half.
    """
    return (np.linalg.inv(leg.affine) @ np.r_[world, 1.])[:3]


def plausibility_distance(image, point_idx: np.ndarray, fhc_idx: np.ndarray) -> float:
    """The distance the lesser-trochanter search tests, computed exactly as it computes it.

    ``femur.py::_find_trochanter_minor.check_plausibility`` casts the candidate to ``int16`` and the
    head centre to ``int32``, maps both to physical space and measures in the (x, z) plane only.
    :param image: The (left-frame) Segmentation the search runs on.
    :param point_idx: The candidate point, continuous index.
    :param fhc_idx: The head centre, continuous index.
    :return: The distance in mm.
    """
    tm = np.asarray(image.transform_index_to_physical_point(np.asarray(point_idx).astype(np.int16)))
    fh = np.asarray(image.transform_index_to_physical_point(np.asarray(fhc_idx).astype(np.int32)))
    return float(np.linalg.norm(tm[[0, 2]] - fh[[0, 2]]))


def undirected_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    """Angle between two in-plane lines, ignoring direction, in [0, 90].

    :param a: Direction of the first line (x, y), index space (in-plane isotropic).
    :param b: Direction of the second line.
    :return: The acute angle in degrees.
    """
    cos = abs(float(np.dot(a, b))) / (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.degrees(np.arccos(min(1., cos))))


def femur_row(leg_left, pts: dict, sz: float) -> dict:
    """The femoral (#4, #6) quantities of one leg.

    :param leg_left: The leg in the left frame (a right half already reflected).
    :param pts: Annotated points as left-frame continuous indices (``None`` if unplaced).
    :param sz: Slice spacing in mm.
    :return: The femoral columns of the row.
    """
    row = {}
    arr = leg_left.array
    zs = np.argwhere(arr == FEMUR_LABEL)[:, 2]
    femur_top, femur_end = int(zs.min()), int(zs.max())
    row['femur_top_z'] = femur_top
    row['crop30_mm'] = int(0.30 * (femur_end - femur_top + 1)) * sz
    try:
        _, fhc_phys = get_femoral_head_center_ct(leg_left, FEMUR_LABEL, 'left')
        fhc = np.asarray(leg_left.transform_physical_point_to_index(fhc_phys), float)
        row['head_z'] = round(float(fhc[2]), 2)
    except Exception as exc:  # noqa: BLE001 -- a survey: record and go on
        row['head_error'] = f'{type(exc).__name__}: {exc}'
        fhc = None
    for label in LT_LABELS:
        p = pts.get(label)
        if p is None:
            continue
        row[f'{label}_z'] = round(float(p[2]), 2)
        if fhc is not None:
            row[f'd_{label}_mm'] = round(plausibility_distance(leg_left, p, fhc), 2)
    if pts.get('LT_inferior') is not None:
        row['need_crop_mm'] = round((pts['LT_inferior'][2] - femur_top) * sz, 2)
        row['crop30_reaches_LT_inferior'] = int(row['need_crop_mm'] < row['crop30_mm'])
    if pts.get('LT_superior') is not None:
        row['head_top_to_LT_superior_mm'] = round((pts['LT_superior'][2] - femur_top) * sz, 2)
        try:
            _, _, _, base = _ct_proximal_landmarks(leg_left, FEMUR_LABEL)
            row['murphy_base_z'] = round(float(base[2]), 2)
            # the neck base is the first slice above the LT's superior extent, so ~ -1 slice is right
            row['murphy_base_minus_LT_superior_mm'] = round((float(base[2]) - pts['LT_superior'][2]) * sz, 2)
        except Exception as exc:  # noqa: BLE001
            row['murphy_error'] = f'{type(exc).__name__}: {str(exc)[:80]}'
    return row


def tibia_row(leg_left, pts: dict, sz: float) -> dict:
    """The tibial (#7) quantities of one leg, for the current knee crop and each candidate.

    :param leg_left: The leg in the left frame (a right half already reflected).
    :param pts: Annotated points as left-frame continuous indices (``None`` if unplaced).
    :param sz: Slice spacing in mm.
    :return: The tibial columns of the row.
    """
    row = {}
    arr = leg_left.array
    if pts.get('tib_post_med') is None or pts.get('tib_post_lat') is None:
        return row
    tib = np.argwhere(arr == TIBIA_LABEL)[:, 2]
    fib = np.argwhere(arr == FIBULA_LABEL)[:, 2]
    pat = np.argwhere(arr == PATELLA_LABEL)[:, 2]
    if not len(tib) or not len(fib) or not len(pat):
        row['tibia_error'] = 'tibia, fibula or patella label missing'
        return row
    tibia_top, fibula_top, knee_start = int(tib.min()), int(fib.min()), int(pat.min())
    plateau = float(pts['tib_post_med'][2])
    row.update(tibia_top_z=tibia_top, fibula_top_z=fibula_top, plateau_z=round(plateau, 2),
               plateau_below_tibia_top_mm=round((plateau - tibia_top) * sz, 2),
               fibula_top_below_tibia_top_mm=round((fibula_top - tibia_top) * sz, 2))
    annotated = (pts['tib_post_lat'] - pts['tib_post_med'])[:2]

    def evaluate(end: int, tag: str):
        """Read the tibial condylar line on the crop [knee_start, end) and compare it."""
        if end <= knee_start:
            row[f'{tag}_error'] = 'empty crop'
            return
        crop = np.zeros_like(arr)
        crop[:, :, knee_start:end] = arr[:, :, knee_start:end]
        try:
            layer, start, stop = get_knee_reference_line(crop, 'tibia', TIBIA_LABEL)
        except Exception as exc:  # noqa: BLE001
            row[f'{tag}_error'] = f'{type(exc).__name__}'
            return
        row[f'{tag}_layer_minus_plateau_mm'] = round((layer - plateau) * sz, 2)
        row[f'{tag}_angle_err_deg'] = round(undirected_angle_deg((stop - start)[:2], annotated), 2)
        row[f'{tag}_line_len_vox'] = round(float(np.linalg.norm((stop - start)[:2])), 2)

    # 'legacy': the pre-2026-11-18 rule (end at the fibula top); 'production': whatever
    # split_ct_image does now (since #7: max(fibula top, tibia top + C.CT_KNEE_CROP_BELOW_TIBIA_TOP_MM))
    evaluate(fibula_top, 'legacy')
    try:
        _, (_, knee, _) = split_ct_image(leg_left, None, femur_label=FEMUR_LABEL, hip_label=PELVIS_LABEL,
                                         patella_label=PATELLA_LABEL, fibula_label=FIBULA_LABEL,
                                         tibia_label=TIBIA_LABEL)
        evaluate(knee_start + knee.array.shape[2], 'production')
    except ValueError as exc:  # production refuses this leg's split: record it, as production would NaN it
        row['production_error'] = f'ValueError: {str(exc)[:80]}'
    for x in KNEE_EXTENSIONS_MM:
        evaluate(max(fibula_top, tibia_top + int(np.ceil(x / sz))), f'x{x}')
    return row


def measure_leg(case: str, patient_side: str, halves: dict) -> dict:
    """All quantities of one annotated leg.

    :param case: Six-digit case id.
    :param patient_side: 'R' or 'L'.
    :param halves: The per-image-side Segmentations from ``split_legs``.
    :return: The row.
    """
    image_side = PATIENT_TO_IMAGE[patient_side]
    leg = halves[image_side]
    width = leg.array.shape[0]
    leg_left = reflect_image_left_right(leg) if image_side == 'right' else leg
    sz = float(leg.spacing[2])
    pts = {}
    for label, world in read_points(NMDID / 'annotations' / f'{case}_{patient_side}.mrk.json').items():
        if world is None:
            pts[label] = None
            continue
        idx = to_left_frame_index(leg, world)
        if image_side == 'right':
            idx[0] = width - 1 - idx[0]
        pts[label] = idx
    row = {'case': case, 'side': patient_side, 'image_side': image_side, 'slice_mm': sz,
           'inplane_mm': float(leg.spacing[0])}
    for fn in (femur_row, tibia_row):
        try:
            row.update(fn(leg_left, pts, sz))
        except Exception as exc:  # noqa: BLE001
            row[f'{fn.__name__}_error'] = f'{type(exc).__name__}: {str(exc)[:80]}'
    return row


def main():
    """Parse arguments, measure every leg of the requested set, write the CSV."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--set', choices=('A', 'B'), required=True)
    parser.add_argument('--quote', action='store_true', help='required for set B (thresholds frozen)')
    parser.add_argument('-o', '--output', type=Path, required=True)
    args = parser.parse_args()
    if args.set == 'B' and not args.quote:
        raise SystemExit('set B is for quoting frozen thresholds; pass --quote once they are in the code')
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logging.getLogger('morphometry').setLevel(logging.ERROR)

    with open(SETS_CSV) as fh:
        cases = [r['case'] for r in sorted(csv.DictReader(fh), key=lambda r: int(r['order'])) if r['set'] == args.set]
    rows = []
    for case in cases:
        halves, _, _ = split_legs(load_case(NMDID / 'segmentations' / f'case-{case}_seg.nii.gz'))
        for side in ('R', 'L'):
            rows.append(measure_leg(case, side, halves))
            log.info('%s', {k: v for k, v in rows[-1].items() if 'x' not in k[:3]})
    fields = list(dict.fromkeys(k for r in rows for k in r))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f'wrote {len(rows)} legs to {args.output}')


if __name__ == '__main__':
    main()
