"""Build response schemas from DB rows (loading encoded images from the Store)."""
from api.db.models import Examination
from api.schemas.docker_io import _sanitize
from api.schemas.enums import ExaminationStatus, ExaminationType
from api.schemas.examination import ExaminationSummary, TorsionDetail, TorsionValues, XRayDetail
from api.storage.store import Store


def to_summary(row: Examination) -> ExaminationSummary:
    return ExaminationSummary(
        patient_name=row.patient_name or "Anonymised",
        study_date=row.study_date,
        study_time=row.study_time,
        study_description=row.study_description,
        accession_number=row.id,
        status=ExaminationStatus(row.status),
    )


def to_detail(row: Examination, store: Store):
    """Build the discriminated detail response, loading encoded slices on demand."""
    summary = to_summary(row).model_dump()

    if row.examination_type == ExaminationType.XRAY.value:
        paths = (row.encoded_paths or {}).get("image", [])
        image = store.load_encoded_b64(paths)[0] if paths else None
        return XRayDetail(**summary, image=image, landmarks=row.landmarks or {})

    encoded = row.encoded_paths or {}
    return TorsionDetail(
        **summary,
        image=store.load_encoded_b64(encoded.get("image", [])),
        segmentation=store.load_encoded_b64(encoded.get("segmentation", [])),
        shape=row.shape,
        knee_offset=row.knee_offset,
        ankle_offset=row.ankle_offset,
        torsion=TorsionValues(**_sanitize(row.torsion_values)) if row.torsion_values else TorsionValues(),
        landmarks=_sanitize(row.landmarks) if row.landmarks else {},
    )
