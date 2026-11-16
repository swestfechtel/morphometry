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
from morphometry.hip import get_femoral_head_center_ct
from morphometry.knee import get_condylar_line_qc
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

# ``qc_flags`` carries, in this order: the segmentation-integrity flags of ``qc_flags()`` (a label in
# several pieces, a missing bone) and the extent flags of ``extent_flags()`` (a bone cut by the volume
# border or the split line, a tibia above the femur); the Murphy femoral-neck flags of ``murphy_qc()``;
# the femoral / tibial posterior-condylar-line flags of ``condylar_qc()``; and the plausibility flags
# of ``plausibility_flags()`` (an impossible femoral-head radius, an uncovered head, a tilted bone).
# Every family only flags: a flagged case is measured and written exactly as an unflagged one.
FIELDNAMES = ['case', 'side', 'femoral_torsion', 'tibial_torsion', 'ccd', 'ccd_projected',
              'femur_length', 'tibia_length', 'femoral_offset', 'mechanical_axis_deviation',
              'hip_knee_ankle_angle', 'center_edge_angle', 'acetabular_version',
              'status', 'qc_flags', 'failed', 'seconds']

# Measurement code raises these on anatomically implausible masks; one bad case
# must not abort a cohort run, so they are caught per measurement and stored as NaN.
MEASUREMENT_ERRORS = (RuntimeError, AssertionError, ValueError, IndexError, TypeError,
                      ZeroDivisionError, np.linalg.LinAlgError)

log = logging.getLogger('process_nmdid')


def _measure(results: dict, failures: list, key: str, fn, nan_failures: list = None):
    """Run one measurement, storing NaN and recording the failure if it raises.

    :param results: The per-side result dict to write into.
    :param failures: A list collecting the names of the measurements that failed.
    :param key: Result key; a tuple of keys when the measurement returns several values.
    :param fn: A zero-argument callable performing the measurement.
    :param nan_failures: If given, a list collecting the keys for which the measurement *returned*
        NaN (kept apart from ``failures`` so the caller can attribute them per side). Used for the
        centre-edge angle, which since 2026-11-16 returns NaN for one side (rather than raising and
        losing both) when that side's edge is not above the head centre; without it the refused side
        would read status ``ok`` with no flag.
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
        if nan_failures is not None and not np.isfinite(results[k]):
            nan_failures.append(k)


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


# --- plausibility gates (flag only, never repair) ---------------------------------------------------
#
# None of these bounds was fitted to the NMDID cohort or to the outlier review that motivated them
# (2026-11-13): each is either geometric or a stated convention, so the flag rates they produce on
# this cohort are not biased by having been chosen on it.

#: The adult femoral-head radius range the package checks fits against (CLAUDE.md, Open item 1). The
#: CT fit is known to inflate the radius -- 17 of the 92 verification halves still fit outside the
#: range after the two-sided box shipped (32 before it; docs/evidence/femoral_head_slab.md) -- so the
#: range itself is deliberately NOT the gate. A leg is
#: flagged only when the fitted radius lies outside it by ``FEMORAL_HEAD_RADIUS_IMPLAUSIBILITY_FACTOR``,
#: i.e. when the fit cannot be a femoral head at all (a near-planar fit seeded by a femur sliver at
#: the split line, a hollow prosthesis tube, a fit on a mislabelled humerus). The factor is a round
#: convention, not a value read off the data.
ADULT_FEMORAL_HEAD_RADIUS_MM = (18.0, 26.0)
FEMORAL_HEAD_RADIUS_IMPLAUSIBILITY_FACTOR = 2.0
FEMORAL_HEAD_RADIUS_BOUNDS_MM = (ADULT_FEMORAL_HEAD_RADIUS_MM[0] / FEMORAL_HEAD_RADIUS_IMPLAUSIBILITY_FACTOR,
                                 ADULT_FEMORAL_HEAD_RADIUS_MM[1] * FEMORAL_HEAD_RADIUS_IMPLAUSIBILITY_FACTOR)

#: The angular error budget of one axial reference line, the same 4 degrees as the package's Murphy
#: and condylar-line gates (``C.MURPHY_AXIS_ANGLE_TOLERANCE_DEG``, ``C.CONDYLAR_LINE_ANGLE_TOLERANCE_DEG``),
#: kept as its own literal as those two are.
TILT_ANGLE_TOLERANCE_DEG = 4.0

#: Bones whose contact with the volume border or the split line is flagged by ``extent_flags``. The
#: pelvis is left out on purpose: the pubic symphysis sits on the midline, so it meets the split by
#: anatomy.
EXTENT_LABELS = (FEMUR_LABEL, TIBIA_LABEL, FIBULA_LABEL)


def axial_angle_error_bound_deg(tilt_deg: float) -> float:
    """The largest error a bone tilt can put into an in-plane angle read on an axial slice.

    An axial slice through a bone whose long axis is tilted by ``theta`` from the scanner z-axis
    is the bone's true cross-section stretched by ``1 / cos(theta)`` along the tilt direction, so
    a line at angle ``phi`` to that direction is read at ``atan(tan(phi) * cos(theta))``. The
    error is largest at ``tan(phi) = 1 / sqrt(cos(theta))``, where it is
    ``atan(1 / sqrt(cos(theta))) - atan(sqrt(cos(theta)))``. The torsion angles compose two such
    lines, whose errors may partly cancel, so this bounds one line, not the torsion.
    :param tilt_deg: The tilt of the bone's long axis from the z-axis, in degrees (0 to < 90).
    :return: The worst-case angular error of one axial line, in degrees.
    """
    c = np.cos(np.radians(tilt_deg))
    return float(np.degrees(np.arctan(1 / np.sqrt(c)) - np.arctan(np.sqrt(c))))


def tilt_threshold_deg(tolerance_deg: float) -> float:
    """The bone tilt at which ``axial_angle_error_bound_deg`` reaches ``tolerance_deg``.

    Found by bisection (the bound rises monotonically with the tilt on [0, 90)).
    :param tolerance_deg: The tolerated worst-case error of one axial line, in degrees.
    :return: The tilt threshold in degrees (29.6 for 4 degrees).
    """
    low, high = 0.0, 89.999
    for _ in range(60):
        mid = (low + high) / 2
        if axial_angle_error_bound_deg(mid) < tolerance_deg:
            low = mid
        else:
            high = mid
    return (low + high) / 2


#: The tilt above which an axial reference line can be off by more than ``TILT_ANGLE_TOLERANCE_DEG``
#: (about 29.6 degrees). Derived, not chosen.
TILT_THRESHOLD_DEG = tilt_threshold_deg(TILT_ANGLE_TOLERANCE_DEG)


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


def extent_flags(array: np.ndarray, side: str, spacing) -> str:
    """Flag bones that a volume border or the split line cuts, and a tibia lying above the femur.

    A bone that runs into a border plane is not whole in this half, so every length, axis and
    landmark drawn from it describes the visible part only -- a femur in the last axial slice
    gives a femur length that is a lower bound, not a measurement -- and nothing else catches
    that: the truncated bone is usually one intact piece. Four planes are checked per bone of
    ``EXTENT_LABELS``: the first and last axial slice (``at_fov_top`` / ``at_fov_bottom``; in LPI
    a smaller z is more superior), the sagittal plane at the split line (``at_split``: a leg
    crossing the midline, or the other leg's bone reaching into this half) and the outer sagittal
    border (``at_lateral_edge``: an abducted leg leaving the field of view). The flag's value is
    the number of that bone's voxels in the plane, so a one-voxel touch reads as such.
    **``tibia:at_split`` is not rare on sound legs**: on 120 control legs with no other flag it
    fired on 12, usually on both legs of a case at once (legs lying together at the midline), so
    on its own it marks a leg to look at, not a broken one. No other extent flag fired on them.

    ``tibia:above_femur_midshaft:<mm>`` marks a tibia whose most superior slice lies above the
    midpoint of the femur's superior-inferior extent, which no knee, flexed or straight, can
    produce: it is upper-body bone carrying the tibia label (typically on a scan whose field of
    view stops at the thigh) and makes a plausible-looking tibia length out of it. The value is
    how far above the midpoint it starts.
    :param array: A cleaned single-leg label array (one image half, LPI).
    :param side: The image side, 'left' or 'right' -- it decides which sagittal border is the split
        line (the left half's last x index, the right half's first).
    :param spacing: The half's voxel spacing in mm, (x, y, z).
    :return: The ``;``-separated flag string, empty when no bone is cut, or
        ``extent_qc_failed:<exception>`` if the check itself failed. Never raises.
    """
    try:
        return _extent_flags(array, side, spacing)
    except Exception as exc:  # QC must never cost a row: an escape here would lose the whole case
        log.warning('extent QC itself failed (%s side): %s: %s', side, type(exc).__name__, exc)
        return f'extent_qc_failed:{type(exc).__name__}'


def _extent_flags(array: np.ndarray, side: str, spacing) -> str:
    """The body of :func:`extent_flags`, which may raise; see there for the flags and parameters."""
    split_x, lateral_x = (array.shape[0] - 1, 0) if side == 'left' else (0, array.shape[0] - 1)
    planes = (('at_fov_top', array[:, :, 0]), ('at_fov_bottom', array[:, :, -1]),
              ('at_split', array[split_x]), ('at_lateral_edge', array[lateral_x]))
    flags = []
    for value in EXTENT_LABELS:
        for tag, plane in planes:
            touching = int(np.count_nonzero(plane == value))
            if touching:
                flags.append(f'{LABEL_NAMES[value]}:{tag}:{touching}')

    femur_z = np.flatnonzero((array == FEMUR_LABEL).any(axis=(0, 1)))
    tibia_z = np.flatnonzero((array == TIBIA_LABEL).any(axis=(0, 1)))
    if femur_z.size and tibia_z.size:
        midshaft = (femur_z[0] + femur_z[-1]) / 2
        if tibia_z[0] < midshaft:
            flags.append(f'tibia:above_femur_midshaft:{(midshaft - tibia_z[0]) * float(spacing[2]):.0f}')
    return ';'.join(flags)


def bone_tilt_deg(array: np.ndarray, value: int, spacing) -> float | None:
    """The angle between one bone's long axis and the scanner z-axis.

    The long axis is the first principal axis of the bone's largest 26-connected component, in
    millimetres (so the 0.8 x 0.8 x 3 mm voxels do not bias it). Only the largest component is
    used so that a speck or a mislabelled fragment elsewhere in the half cannot swing the axis.
    For a whole femur the neck and head pull the principal axis a few degrees off the shaft; for
    a femur truncated near the trochanters they dominate it, so on a femur that also carries an
    ``at_fov_bottom`` flag the tilt is not a shaft tilt.
    :param array: A cleaned single-leg label array.
    :param value: The bone's label.
    :param spacing: The voxel spacing in mm, (x, y, z).
    :return: The tilt in degrees in [0, 90], or None when the bone is absent or too small for an axis.
    """
    mask = array == value
    if not mask.any():
        return None
    components, _ = connected_components(mask, np.ones((3, 3, 3), dtype=bool))
    sizes = np.bincount(components.ravel())
    sizes[0] = 0
    points = np.argwhere(components == int(np.argmax(sizes))) * np.asarray(spacing, dtype=float)
    if len(points) < 3:
        return None
    points -= points.mean(axis=0)
    _, vectors = np.linalg.eigh(points.T @ points)
    axis = vectors[:, -1]  # eigh sorts ascending: the last vector is the principal axis
    return float(np.degrees(np.arccos(min(1.0, abs(axis[2])))))


def plausibility_flags(leg: Segmentation, side: str, row: dict) -> str:
    """Flag the measurements of one leg that rest on an impossible landmark or a tilted bone.

    Appended last to this row's ``qc_flags``; never raises and never changes a value.

    * ``femoral_head_radius_implausible:<mm>`` -- the CT femoral-head sphere fit (the one every
      hip measurement of this row uses) returned a radius outside ``FEMORAL_HEAD_RADIUS_BOUNDS_MM``.
      The CCD, femoral offset, femoral torsion, CE angle and acetabular version of the row all
      rest on that centre. ``femoral_head_landmark_failed:<exception>`` if the fit itself raised.
    * ``center_edge_not_above_head:<deg>`` -- a CE angle of 90 degrees or more, i.e. the lateral
      acetabular edge point found is not above the femoral head centre. That is geometry, not a
      threshold: the head is uncovered (dislocated) or the edge point is not a roof. The CE angle
      is unsigned, so a value above 90 is not a large coverage either. **Since 2026-11-16 the
      measurement itself refuses such a side** (``calculate_center_edge_angle`` returns NaN for
      it), so this flag cannot fire on new runs; it is kept for reading older tables. The refused
      side is listed as ``center_edge_angle_<side>`` in that row's ``failed`` column instead.
    * ``femur_tilt:<deg>`` / ``tibia_tilt:<deg>`` -- the bone's long axis (``bone_tilt_deg``) is
      tilted from the scanner z-axis by more than ``TILT_THRESHOLD_DEG``, beyond which an axial
      reference line can be off by more than ``TILT_ANGLE_TOLERANCE_DEG``: the torsion that bone
      contributes to is read on oblique sections (a flexed knee, an abducted hip).
    :param leg: One cleaned single-leg Segmentation.
    :param side: The image side, 'left' or 'right'.
    :param row: This leg's result row (``center_edge_angle`` is read from it).
    :return: The ``;``-separated flag string, empty when nothing is implausible.
    """
    flags = []
    try:
        radius, _ = get_femoral_head_center_ct(leg, segmentation_label=FEMUR_LABEL, side=side)
        low, high = FEMORAL_HEAD_RADIUS_BOUNDS_MM
        if not low <= radius <= high:
            flags.append(f'femoral_head_radius_implausible:{radius:.1f}')
    except Exception as exc:  # a failed fit is reported, never raised
        flags.append(f'femoral_head_landmark_failed:{type(exc).__name__}')

    center_edge = row.get('center_edge_angle', np.nan)
    if np.isfinite(center_edge) and center_edge >= 90.:
        flags.append(f'center_edge_not_above_head:{center_edge:.1f}')

    try:
        for value, name in ((FEMUR_LABEL, 'femur'), (TIBIA_LABEL, 'tibia')):
            tilt = bone_tilt_deg(leg.array, value, leg.spacing)
            if tilt is not None and tilt > TILT_THRESHOLD_DEG:
                flags.append(f'{name}_tilt:{tilt:.1f}')
    except Exception as exc:  # QC must never cost a row
        log.warning('bone tilt QC itself failed (%s side): %s: %s', side, type(exc).__name__, exc)
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


def condylar_qc(knee: Segmentation, side: str) -> str:
    """QC-flag the femoral and tibial posterior condylar lines of one leg's knee crop, never raising.

    Appended to this row's ``qc_flags`` after the Murphy flags, femur first, in the format of
    ``compute_torsion.py`` and ``process_augsburg_large_dataset.py``. A ``femoral_*`` flag marks
    this row's femoral torsion as untrustworthy, a ``tibial_*`` flag its tibial torsion; the values
    themselves are unchanged. ``get_condylar_line_qc`` resolves the line exactly as the consumers
    do (binarise the bone label, reflect a right half first), so the flags describe the line that
    was actually measured. Like ``murphy_qc``, it already reports a failed search as a
    ``*_landmark_failed`` flag, so the handler here only fires if the QC itself is broken, and logs
    at WARNING for the same reason.
    :param knee: The leg's knee crop from ``split_ct_image`` (keeps its geometry, so the
        in-plane isotropy check runs and a millimetre length is available).
    :param side: The image side, 'left' or 'right'.
    :return: The ``;``-separated flag string, empty when both lines look sound.
    """
    flags = []
    for bone, label in (('femur', FEMUR_LABEL), ('tibia', TIBIA_LABEL)):
        try:
            flags.append(get_condylar_line_qc(knee, bone, segmentation_label=label, side=side))
        except Exception as exc:  # QC must never cost a row
            log.warning('%s condylar-line QC itself failed (%s side): %s: %s -- its flags are missing '
                        'because the QC did not run, not because the line is sound',
                        bone, side, type(exc).__name__, exc)
    return ';'.join(f for f in flags if f)


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
    flags = {side: ';'.join(f for f in (qc_flags(array), extent_flags(array, side, raw[side].spacing)) if f)
             for side, array in arrays.items()}
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
    refused = []  # both-sides keys returned as NaN for one side only (attributed per side below)
    _measure(both, both_failures, ('center_edge_angle_left', 'center_edge_angle_right'),
             lambda: calculate_center_edge_angle(seg, femur_label=FEMUR_LABEL, acetabulum_label=PELVIS_LABEL,
                                                 ct=True, project=True,
                                                 acetabulum_radius_factor=ACETABULUM_RADIUS_FACTOR),
             nan_failures=refused)
    _measure(both, both_failures, ('acetabular_version_left', 'acetabular_version_right'),
             lambda: calculate_acetabular_anteversion(seg, femur_label=FEMUR_LABEL, acetabulum_label=PELVIS_LABEL,
                                                      ct=True,
                                                      acetabulum_radius_factor=ACETABULUM_RADIUS_FACTOR))

    rows = []
    for side in ('left', 'right'):
        leg = legs[side]
        row = {'case': case, 'side': side}
        # a raised both-sides measurement fails both rows (pre-existing); a one-side NaN only its own
        failures = list(both_failures) + [k for k in refused if k.endswith(f'_{side}')]

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

        knee_flags = ''
        if masks is None:
            # no knee crop: both torsions are NaN and listed in ``failed``, so there is no condylar
            # line to flag; ``failed`` (not an empty ``qc_flags``) is what says so for this row
            for key in ('femoral_torsion', 'tibial_torsion', 'mechanical_axis_deviation', 'hip_knee_ankle_angle'):
                row[key] = np.nan
                failures.append(key)
        else:
            knee, ankle = masks[1], masks[2]
            knee_flags = condylar_qc(knee, side)
            _measure(row, failures, 'femoral_torsion',
                     lambda: calculate_femoral_torsion_ct(leg, knee, side=side, segmentation_label=FEMUR_LABEL))
            _measure(row, failures, 'tibial_torsion',
                     # pass the Segmentations, not bare arrays, so their spacing reaches the
                     # in-plane isotropy check of the index-space angle composition (task #10)
                     lambda: calculate_tibial_torsion(knee, ankle, TIBIA_LABEL, TIBIA_LABEL,
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
        # The condylar-line flags play the same role for both torsions' distal reference line.
        row['qc_flags'] = ';'.join(f for f in (flags[side], murphy_qc(leg, side), knee_flags,
                                               plausibility_flags(leg, side, row)) if f)
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
