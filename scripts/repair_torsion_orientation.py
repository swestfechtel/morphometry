#!/usr/bin/env python3
"""Repair torsion examinations whose whole-leg volume was split upside-down.

``transform_coordinate_system`` reorients to LPI using only the DICOM affine, so
a series with bad orientation metadata (acquired/stored upside-down: the affine
says LPI while the voxels run ankle -> knee -> hip) is *not* corrected, and
``_split_volume`` mislabels ankle as hip. The ingest pipeline now detects and
fixes this automatically (``api.ingest.dicom._orient_superoinferior``), but
examinations ingested *before* that fix keep the wrong split on disk.

This script re-derives the corrected volumes from the retained raw ``original``
volume — re-running the exact current ingest pipeline (transform -> orient ->
split) — overwrites the stored ``transformed`` / ``hip`` / ``knee`` / ``ankle``
volumes and the ``shape`` / ``knee_offset`` / ``ankle_offset`` columns, and
resets the examination to ``unprocessed`` (clearing the now-stale masks,
landmarks and torsion values) so it can be segmented + recomputed cleanly.

Only examinations that are actually upside-down are touched; correctly-oriented
ones are reported as "ok" and left alone (unless ``--force``).

Usage:
    python scripts/repair_torsion_orientation.py 0003420100 [more accessions...]
    python scripts/repair_torsion_orientation.py --all
    python scripts/repair_torsion_orientation.py --all --dry-run

Requires the same MORPH_API_* settings (storage dir / database) the API uses.
"""
import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from api.db import repository                       # noqa: E402
from api.db.engine import session_scope             # noqa: E402
from api.db.models import Examination               # noqa: E402
from api.ingest.dicom import _orient_superoinferior, _slice_footprint, _split_volume  # noqa: E402
from api.runtime import get_engine, get_store       # noqa: E402
from api.schemas.enums import ExaminationStatus, ExaminationType  # noqa: E402


def _is_upside_down(transformed) -> bool:
    """True if the LPI volume's low-z footprint is smaller than its high-z end."""
    footprint = _slice_footprint(transformed.array)
    q = max(1, len(footprint) // 4)
    return float(footprint[:q].mean()) < float(footprint[-q:].mean())


def repair_examination(accession: str, *, force: bool = False, dry_run: bool = False) -> str:
    """Re-split one torsion examination from its stored raw ``original`` volume.

    :param accession: the examination id to repair.
    :param force: re-split and reset even if the volume already looks correct.
    :param dry_run: detect and report only; do not write volumes or the DB row.
    :return: a short human-readable status string for logging.
    """
    store = get_store()
    with session_scope(get_engine()) as session:
        row = repository.get_examination(session, accession)
        if row is None:
            return f"{accession}: not found"
        if row.examination_type != ExaminationType.TORSION.value:
            return f"{accession}: skipped (not a torsion examination)"
        source_paths = dict(row.source_paths or {})

    original_rel = source_paths.get("original")
    if not original_rel or not store.abspath(original_rel).exists():
        return f"{accession}: skipped (raw 'original' volume missing — cannot re-derive)"

    # Reproduce the current ingest pipeline: transform -> orient -> split.
    transformed = store.load_image(original_rel).copy()
    transformed.transform_coordinate_system()
    upside_down = _is_upside_down(transformed)
    if not upside_down and not force:
        return f"{accession}: ok (already superior-first; no change)"

    transformed = _orient_superoinferior(transformed)
    regions = _split_volume(transformed)
    knee_offset = int(regions["hip"].shape[2])
    ankle_offset = knee_offset + int(regions["knee"].shape[2])

    verb = "would repair" if dry_run else "repaired"
    detail = "flipped" if upside_down else "re-split (forced)"
    if dry_run:
        return f"{accession}: {verb} ({detail}); knee_offset={knee_offset} ankle_offset={ankle_offset}"

    # Overwrite the derived volumes in place (same kinds -> same file paths).
    source_paths["transformed"] = store.save_volume(accession, "transformed", transformed)
    for region in ("hip", "knee", "ankle"):
        source_paths[region] = store.save_volume(accession, region, regions[region])

    with session_scope(get_engine()) as session:
        row = repository.get_examination(session, accession)
        row.source_paths = source_paths
        row.shape = list(transformed.shape)
        row.knee_offset = knee_offset
        row.ankle_offset = ankle_offset
        # The old masks/landmarks/torsion were computed on the wrong split.
        row.mask_paths = None
        row.landmarks = None
        row.torsion_values = None
        row.status = ExaminationStatus.UNPROCESSED.value
        repository.upsert_examination(session, row)

    return (f"{accession}: {verb} ({detail}); reset to unprocessed — "
            f"re-run segmentation + torsion. knee_offset={knee_offset} ankle_offset={ankle_offset}")


def _all_torsion_ids() -> list[str]:
    with session_scope(get_engine()) as session:
        return [e.id for e in repository.list_examinations(session)
                if e.examination_type == ExaminationType.TORSION.value]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("accessions", nargs="*", help="examination ids to repair")
    parser.add_argument("--all", action="store_true", help="scan every torsion examination")
    parser.add_argument("--force", action="store_true",
                        help="re-split + reset even if the volume already looks correct")
    parser.add_argument("--dry-run", action="store_true",
                        help="detect and report only; write nothing")
    args = parser.parse_args()

    accessions = args.accessions
    if args.all:
        accessions = _all_torsion_ids()
    if not accessions:
        parser.error("pass one or more accession numbers, or --all")

    for accession in accessions:
        print(repair_examination(accession, force=args.force, dry_run=args.dry_run))


if __name__ == "__main__":
    main()
