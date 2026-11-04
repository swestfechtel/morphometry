#!/usr/bin/env python3
"""Survey the in-plane voxel spacing of the whole-leg CT cohort (backlog task #10).

Task #10 composes the axial-plane angles in **index** space, which assumes equal
in-plane x/y voxel edges, and ``geometry.validate_in_plane_isotropy`` now raises a
``ValueError`` when they differ by more than ``C.IN_PLANE_SPACING_RTOL``. The CT
femoral torsion checks its ``Segmentation`` arguments unconditionally, so a case
with rectangular in-plane voxels makes ``process_nmdid.py`` write a NaN
``femoral_torsion`` where it previously wrote a sheared number. Until this survey
that risk was unquantified: the CT spacing was quoted from the acquisition protocol
(0.8 x 0.8 x 3.0 mm), not from a measurement, because the sshfs mount holding the
cohort was down.

**The zooms are permuted into the post-LPI axis order**, which is the whole point:
``transform_coordinate_system`` permutes and flips voxel axes, so the slice axis can
end up *in* the plane -- that is exactly how ``dfg_knees`` comes to violate the
precondition (0.75 x 0.3125 mm in plane after reorientation). Surveying the raw
header zooms would miss it.

Headers only: ``nibabel`` reads the header without touching the voxel data, so this
runs over a cohort of this size in minutes rather than the hours a full load would
take over sshfs.

    python scripts/survey_ct_spacing.py -i <dir> [<dir> ...] -o out/ct_spacing

Writes ``spacing.csv`` (one row per file: the raw zooms, the achieved axis codes, the
post-LPI zooms, the in-plane ratio and whether the isotropy check would pass),
``summary.md`` and ``summary.meta.json``.

:return: Nothing; the findings are the written files.
"""
import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from multiprocessing import Pool
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import nibabel as nib
import numpy as np
import pandas as pd

from morphometry import constants as C

#: The orientation the pipelines request before measuring.
TARGET_AXCODES = ('L', 'P', 'I')


def post_lpi_zooms(affine: np.ndarray, zooms: np.ndarray, shape: tuple):
    """Permute header zooms into the axis order the LPI reorientation produces.

    This replicates ``image_io.Image.transform_coordinate_system`` exactly, on the header
    alone: the orientation transform from the image's own axes to ``TARGET_AXCODES``, the
    resulting affine, and the axis codes that affine actually achieves. Deriving the
    transform relative to RAS instead (plain ``io_orientation``) would describe a
    reorientation the pipelines never perform and would report every image as RAS.

    ``apply_orientation``'s transform is read per OUTPUT axis -- ``transform[i, 0]`` is the
    input axis that becomes output axis ``i`` -- so the zooms permute by that column.

    :param affine: The image's 4x4 index-to-world affine, as stored in the header.
    :param zooms: The header's three voxel edge lengths, in header axis order.
    :param shape: The image's array shape, needed for the inverse orientation affine.
    :return: ``(permuted_zooms, achieved_axcodes)`` -- the zooms in post-LPI axis order
        and the axis codes the reorientation actually achieves (which need not be the
        requested LPI on a strongly oblique scan; that is the `non_lpi_orientation`
        condition the knee scripts flag).
    """
    orientation = nib.orientations.io_orientation(affine)
    target = nib.orientations.axcodes2ornt(TARGET_AXCODES)
    transform = nib.orientations.ornt_transform(orientation, target)
    order = transform[:, 0].astype(int)
    permuted = np.asarray(zooms, dtype=float)[order]
    new_affine = affine @ nib.orientations.inv_ornt_aff(transform, shape)
    return permuted, nib.orientations.aff2axcodes(new_affine)


def survey_file(path_str: str) -> dict:
    """Read one segmentation's header and derive its post-LPI in-plane spacing.

    :param path_str: Path to a ``.nii`` / ``.nii.gz`` file.
    :return: A row dict; ``error`` is set and the spacing columns are NaN if the header
        cannot be read.
    """
    path = Path(path_str)
    row = {'file': path.name, 'directory': str(path.parent)}
    try:
        image = nib.load(str(path))
        zooms = np.asarray(image.header.get_zooms()[:3], dtype=float)
        affine = np.asarray(image.affine, dtype=float)
        row['raw_x'], row['raw_y'], row['raw_z'] = zooms
        row['raw_axcodes'] = ''.join(nib.orientations.aff2axcodes(affine))
        permuted, achieved = post_lpi_zooms(affine, zooms, image.shape[:3])
        row['lpi_x'], row['lpi_y'], row['lpi_z'] = permuted
        row['achieved_axcodes'] = ''.join(achieved)
        row['orientation_ok'] = tuple(achieved) == TARGET_AXCODES
        larger = max(permuted[0], permuted[1])
        row['in_plane_abs_diff'] = abs(permuted[0] - permuted[1])
        row['in_plane_ratio'] = permuted[0] / permuted[1] if permuted[1] else np.nan
        row['bit_identical'] = permuted[0] == permuted[1]
        row['isotropy_ok'] = bool(row['in_plane_abs_diff'] <= C.IN_PLANE_SPACING_RTOL * larger)
        row['slice_aspect'] = permuted[2] / permuted[0] if permuted[0] else np.nan
    except Exception as exc:                              # noqa: BLE001 - a bad header is a result
        row['error'] = f'{type(exc).__name__}: {exc}'
        for key in ('lpi_x', 'lpi_y', 'lpi_z', 'in_plane_abs_diff', 'in_plane_ratio'):
            row[key] = np.nan
        row['isotropy_ok'] = False
    return row


def summarise(frame: pd.DataFrame) -> str:
    """Render the survey as markdown.

    :param frame: The per-file survey frame from :func:`survey_file`.
    :return: The report text.
    """
    ok = frame[frame.get('error').isna()] if 'error' in frame else frame
    lines = ['# Whole-leg CT in-plane spacing survey', '',
             'Backlog task #10, regenerated by `scripts/survey_ct_spacing.py`.', '',
             f'Files read: **{len(frame)}**'
             + (f' ({int(frame["error"].notna().sum())} unreadable)' if 'error' in frame else ''),
             '']
    if ok.empty:
        return '\n'.join(lines + ['No readable file.'])

    failing = ok[~ok.isotropy_ok]
    lines += [
        f'**In-plane isotropy check: {int(ok.isotropy_ok.sum())} of {len(ok)} pass** '
        f'(tolerance `IN_PLANE_SPACING_RTOL` = {C.IN_PLANE_SPACING_RTOL}).',
        f'Bit-identical in-plane zooms: **{int(ok.bit_identical.sum())} of {len(ok)}**.',
        '',
        '## Distinct post-LPI spacings',
        '',
    ]
    grouped = (ok.groupby([ok.lpi_x.round(8), ok.lpi_y.round(8), ok.lpi_z.round(8)])
                 .size().reset_index(name='files')
                 .rename(columns={'lpi_x': 'in-plane x', 'lpi_y': 'in-plane y', 'lpi_z': 'slice'})
                 .sort_values('files', ascending=False))
    lines += [grouped.to_markdown(index=False), '']

    lines += ['## Orientation', '',
              f'Reorientation achieves LPI on **{int(ok.orientation_ok.sum())} of {len(ok)}**; '
              f'the others are listed in `spacing.csv` with their `achieved_axcodes`.', '']
    not_lpi = ok[~ok.orientation_ok]
    if not not_lpi.empty:
        lines += [not_lpi[['file', 'raw_axcodes', 'achieved_axcodes',
                           'lpi_x', 'lpi_y', 'lpi_z']].to_markdown(index=False), '']

    lines += ['## Cases the isotropy check would refuse', '']
    if failing.empty:
        lines += ['**None.** No case in this survey would have its CT femoral torsion turned '
                  'into a NaN row by the task #10 assert.', '']
    else:
        lines += [f'**{len(failing)}** -- each would write a NaN `femoral_torsion` in '
                  '`process_nmdid.py` where it previously wrote a sheared number.', '',
                  failing[['file', 'lpi_x', 'lpi_y', 'in_plane_ratio',
                           'achieved_axcodes']].to_markdown(index=False), '']
    return '\n'.join(lines)


def main() -> None:
    """Parse the command line, survey every file found and write the outputs.

    :return: Nothing.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('-i', '--inputs', type=Path, nargs='+', required=True,
                        help='Directories (searched recursively) and/or single .nii/.nii.gz files.')
    parser.add_argument('-o', '--out-dir', type=Path, required=True, help='Output directory.')
    parser.add_argument('--workers', type=int, default=16, help='Worker processes.')
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

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with Pool(args.workers) as pool:
        rows = pool.map(survey_file, paths, chunksize=4)
    frame = pd.DataFrame(rows)
    frame.to_csv(args.out_dir / 'spacing.csv', index=False)
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
        'n_files': len(paths),
        'tolerance': C.IN_PLANE_SPACING_RTOL,
    }, indent=2) + '\n')
    print(report)


if __name__ == '__main__':
    main()
