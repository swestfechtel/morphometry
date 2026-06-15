#!/usr/bin/env python3
"""One-off migration of legacy pickled examinations into the new DB + file store.

The pre-refactor API pickled whole ``TorsionExamination`` / ``XRayExamination``
objects (full volumes + masks + base64 PNG lists) to ``api/data/{accession}.pkl``.
This script unpickles each one (using the retained ``api.examination`` classes)
and writes it into the new layout: lightweight row in SQLite + ``.nii.gz`` image
files + encoded PNGs on disk. Idempotent by accession.

Usage:
    python scripts/migrate_pickles.py [--pickle-dir api/data] [--replace]

Run once, before retiring the legacy pickles. Requires the same MORPH_API_*
settings (storage dir / database) the API uses.
"""
import argparse
import pickle
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from api.db import repository                       # noqa: E402
from api.db.engine import session_scope             # noqa: E402
from api.db.models import Examination               # noqa: E402
from api.examination import TorsionExamination, XRayExamination  # noqa: E402  (legacy classes)
from api.runtime import get_engine, get_store       # noqa: E402
from api.schemas.enums import ExaminationStatus, ExaminationType  # noqa: E402


def _study_fields(exam) -> dict:
    def _safe(getter):
        try:
            return getter()
        except Exception:  # noqa: BLE001
            return None
    return {
        "study_date": _safe(lambda: exam.study_date),
        "study_time": _safe(lambda: exam.study_time),
        "study_description": _safe(lambda: exam.study_description),
        "patient_name": "Anonymised",
    }


def migrate_one(pkl_path: Path, replace: bool) -> str:
    exam = pickle.loads(pkl_path.read_bytes())
    accession = exam.identifier
    store, engine = get_store(), get_engine()

    with session_scope(engine) as session:
        if repository.get_examination(session, accession) is not None and not replace:
            return f"skip {accession} (exists)"

    if isinstance(exam, TorsionExamination):
        source_paths = {}
        for kind, attr in (("original", "original_image"), ("transformed", "transformed_image"),
                           ("hip", "hip"), ("knee", "knee"), ("ankle", "ankle")):
            image = getattr(exam, attr, None)
            if image is not None:
                source_paths[kind] = store.save_volume(accession, kind, image)
        mask_paths = {}
        for region in ("hip", "knee", "ankle"):
            mask = getattr(exam, f"{region}_mask", None)
            if mask is not None:
                mask_paths[region] = store.save_mask(accession, region, mask)
        encoded = None
        if getattr(exam, "image_b64", None) and getattr(exam, "image_segmentation_b64", None):
            encoded = store.save_encoded(accession, exam.image_b64, exam.image_segmentation_b64)
        torsion = exam.get_torsion_values() if getattr(exam, "femoral_torsion_left", None) is not None else None
        row = Examination(
            id=accession, examination_type=ExaminationType.TORSION.value, status=exam.status,
            source_paths=source_paths, mask_paths=mask_paths or None, encoded_paths=encoded,
            torsion_values=torsion, landmarks=getattr(exam, "landmarks", None),
            shape=list(exam.transformed_image.shape) if getattr(exam, "transformed_image", None) else None,
            **_study_fields(exam),
        )
    elif isinstance(exam, XRayExamination):
        encoded = store.save_encoded(accession, [exam.to_base64()], [])
        row = Examination(id=accession, examination_type=ExaminationType.XRAY.value,
                          status=ExaminationStatus.PROCESSED.value, encoded_paths=encoded,
                          landmarks=exam.landmarks, **_study_fields(exam))
    else:
        return f"skip {accession} (unknown type {type(exam).__name__})"

    with session_scope(engine) as session:
        repository.upsert_examination(session, row)
    return f"migrated {accession}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pickle-dir", default="api/data", help="Directory of legacy *.pkl files")
    parser.add_argument("--replace", action="store_true", help="Overwrite existing rows")
    args = parser.parse_args()

    for pkl in sorted(Path(args.pickle_dir).glob("*.pkl")):
        try:
            print(migrate_one(pkl, args.replace))
        except Exception as exc:  # noqa: BLE001
            print(f"FAILED {pkl.name}: {exc}")


if __name__ == "__main__":
    main()
