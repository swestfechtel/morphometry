"""Turn uploaded DICOM into stored images + an Examination row.

Ported from the old ``FileController.save_files`` / ``save_torsion_series`` /
``TorsionExamination.split_series``, but writing ``.nii.gz`` to the Store and a
lightweight row to the DB instead of pickling a god-object. These functions do
blocking image work and are meant to be called via ``run_in_threadpool`` from the
async endpoints (or directly from the worker for the Orthanc path).
"""
import logging
import random
import string
from datetime import datetime
from pathlib import Path

import nibabel as nib
import numpy as np
import ruptures

from morphometry.image_io import Image

from api.db import repository
from api.db.engine import session_scope
from api.db.models import Examination
from api.errors import DuplicateError
from api.runtime import get_engine, get_settings, get_store
from api.schemas.enums import ExaminationStatus, ExaminationType

logger = logging.getLogger("api")


def _random_accession() -> str:
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=10))


def _study_fields(metadata) -> dict:
    """Extract the small subset of DICOM tags the app displays/stores."""
    def _fmt(tag, in_fmt, out_fmt):
        try:
            return datetime.strptime(metadata[tag].value, in_fmt).strftime(out_fmt)
        except Exception:  # noqa: BLE001 - missing/odd tags are non-fatal
            return None

    def _val(tag):
        try:
            return str(metadata[tag].value)
        except Exception:  # noqa: BLE001
            return None

    return {
        "study_date": _fmt((0x0008, 0x0020), "%Y%m%d", "%Y-%m-%d"),
        "study_time": _fmt((0x0008, 0x0030), "%H%M%S", "%H:%M"),
        "study_description": _val((0x0008, 0x1030)),
        "patient_name": "Anonymised",
        "dicom_metadata": {
            "study_date": _fmt((0x0008, 0x0020), "%Y%m%d", "%Y-%m-%d"),
            "study_description": _val((0x0008, 0x1030)),
            "accession_number": _val((0x0008, 0x0050)),
        },
    }


def _accession(metadata) -> str:
    accession = ""
    try:
        accession = str(metadata.AccessionNumber)
    except Exception:  # noqa: BLE001
        pass
    if not accession:
        accession = _random_accession()
        logger.warning("No accession number in metadata; using generated id %s", accession)
    return accession


def _split_volume(transformed: Image) -> dict[str, Image]:
    """Split a stacked whole-leg volume into hip/knee/ankle via changepoint detection."""
    arr = transformed.array
    cleaned = np.where(arr < 50, 0, arr)
    num_pixels = np.array([np.count_nonzero(cleaned[:, :, z]) for z in range(arr.shape[2])])
    # ruptures returns [bkp1, bkp2, len(signal)] for n_bkps=2; the trailing length is ignored
    breakpoints = ruptures.KernelCPD().fit_predict(num_pixels, 2)
    knee_hip, ankle_knee = breakpoints[0], breakpoints[1]
    affine = transformed.affine
    return {
        "hip": Image.from_nibabel(nib.Nifti1Image(arr[:, :, :knee_hip], affine=affine)),
        "knee": Image.from_nibabel(nib.Nifti1Image(arr[:, :, knee_hip:ankle_knee], affine=affine)),
        "ankle": Image.from_nibabel(nib.Nifti1Image(arr[:, :, ankle_knee:], affine=affine)),
    }


def _check_duplicate(accession: str) -> None:
    settings = get_settings()
    with session_scope(get_engine()) as session:
        existing = repository.get_examination(session, accession)
    if existing is None:
        return
    if settings.on_duplicate == "replace":
        logger.warning("Replacing existing examination %s", accession)
        get_store().delete_examination(accession)
        with session_scope(get_engine()) as session:
            repository.delete_examination(session, accession)
    else:
        raise DuplicateError(f"Examination {accession} already exists")


def _persist_torsion(accession: str, study: dict, original: Image, transformed: Image,
                     regions: dict[str, Image]) -> str:
    """Save volumes + insert the examination row; return the examination id."""
    store = get_store()
    source_paths = {
        "original": store.save_volume(accession, "original", original),
        "transformed": store.save_volume(accession, "transformed", transformed),
        "hip": store.save_volume(accession, "hip", regions["hip"]),
        "knee": store.save_volume(accession, "knee", regions["knee"]),
        "ankle": store.save_volume(accession, "ankle", regions["ankle"]),
    }
    knee_offset = int(regions["hip"].shape[2])
    ankle_offset = knee_offset + int(regions["knee"].shape[2])
    with session_scope(get_engine()) as session:
        repository.upsert_examination(session, Examination(
            id=accession,
            examination_type=ExaminationType.TORSION.value,
            status=ExaminationStatus.UNPROCESSED.value,
            shape=list(transformed.shape),
            knee_offset=knee_offset,
            ankle_offset=ankle_offset,
            source_paths=source_paths,
            **study,
        ))
    return accession


def ingest_torsion_from_dir(dicom_dir: Path) -> str:
    """Ingest a single stacked DICOM series (UI single upload / Orthanc finalize)."""
    metadata = Image.read_dicom_metadata(str(dicom_dir))
    accession = _accession(metadata)
    _check_duplicate(accession)

    nib_image, tmp = Image.dicom_to_nibabel(str(dicom_dir))
    try:
        original = Image.from_nibabel(nib_image)
        transformed = original.copy()
        transformed.transform_coordinate_system()
    finally:
        tmp.cleanup()

    regions = _split_volume(transformed)
    return _persist_torsion(accession, _study_fields(metadata), original, transformed, regions)


def ingest_xray(image_path: Path) -> str:
    """Ingest a single 2D x-ray image (stored as a base64 PNG)."""
    import base64
    from io import BytesIO
    from PIL import Image as PILImage

    store = get_store()
    accession = _random_accession()
    image = PILImage.open(image_path).convert("RGB")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    encoded = store.save_encoded(accession, [base64.b64encode(buffer.getvalue()).decode("ascii")], [])

    landmarks = {
        "longitudinal_firstmetatarsal_axis": {"start": [50, 50], "end": [150, 150]},
        "longitudinal_phalanx_axis": {"start": [100, 100], "end": [200, 200]},
    }
    with session_scope(get_engine()) as session:
        repository.upsert_examination(session, Examination(
            id=accession,
            examination_type=ExaminationType.XRAY.value,
            status=ExaminationStatus.PROCESSED.value,
            patient_name="Anonymised",
            encoded_paths=encoded,
            landmarks=landmarks,
        ))
    return accession


def ingest_torsion_multi_from_dirs(hip_dir: Path, knee_dir: Path, ankle_dir: Path) -> str:
    """Ingest three separate DICOM series (hip/knee/ankle), already split."""
    metadata = Image.read_dicom_metadata(str(hip_dir))
    accession = _accession(metadata)
    _check_duplicate(accession)

    images, tmps = {}, []
    try:
        for region, directory in (("hip", hip_dir), ("knee", knee_dir), ("ankle", ankle_dir)):
            nib_image, tmp = Image.dicom_to_nibabel(str(directory))
            tmps.append(tmp)
            img = Image.from_nibabel(nib_image)
            img.transform_coordinate_system()
            images[region] = img

        shapes = [images[r].array.shape[:2] for r in ("hip", "knee", "ankle")]
        if len(set(shapes)) != 1:
            from api.errors import IngestError
            raise IngestError(f"In-plane shapes differ between series: {shapes}")

        combined = np.concatenate([images[r].array for r in ("hip", "knee", "ankle")], axis=2)
        transformed = Image.from_nibabel(nib.Nifti1Image(combined, affine=images["hip"].affine))
    finally:
        for tmp in tmps:
            tmp.cleanup()

    return _persist_torsion(accession, _study_fields(metadata), transformed.copy(), transformed, images)
