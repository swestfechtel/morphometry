"""Examination request/response schemas."""
from typing import Literal

from pydantic import BaseModel

from api.schemas.enums import ExaminationStatus


class ExaminationSummary(BaseModel):
    """List-view summary of an examination (no images)."""
    patient_name: str = "Anonymised"
    study_date: str | None = None
    study_time: str | None = None
    study_description: str | None = None
    accession_number: str
    status: ExaminationStatus
    # Id of a currently queued/running job for this examination (else null). Lets the
    # list show live progress and poll the right job; null means nothing is in flight.
    active_job_id: str | None = None


class ExaminationList(BaseModel):
    """List response envelope — the UI reads ``result.examinations``."""
    examinations: list[ExaminationSummary]


class TorsionValues(BaseModel):
    """The six torsion angles, NaN-sanitized for display."""
    femoral_torsion_left: float = 0
    femoral_torsion_right: float = 0
    femoral_torsion_left_murphy: float = 0
    femoral_torsion_right_murphy: float = 0
    tibial_torsion_left: float = 0
    tibial_torsion_right: float = 0


class TorsionDetail(ExaminationSummary):
    """Detail view for a torsion examination (matches the legacy GET payload)."""
    type: Literal["torsion"] = "torsion"
    image: list[str] = []
    segmentation: list[str] = []
    shape: list[int] | None = None
    knee_offset: int | None = None
    ankle_offset: int | None = None
    torsion: TorsionValues = TorsionValues()
    landmarks: dict = {}


class SeriesInfo(BaseModel):
    """One candidate DICOM series offered for selection on a pending upload."""
    uid: str
    description: str | None = None
    modality: str | None = None
    instances: int = 0
    rows: int | None = None
    cols: int | None = None
    preview_count: int = 0


class PendingDetail(ExaminationSummary):
    """Detail view for a pending_selection examination: its candidate series.

    Preview slices are served separately as PNGs at
    ``/examinations/{id}/series/{uid}/preview/{index}.png`` (0 … preview_count-1).
    """
    type: Literal["pending"] = "pending"
    series: list[SeriesInfo] = []


class ExaminationUpdate(BaseModel):
    """Whitelisted, validated fields a client may PATCH (no blind setattr)."""
    status: ExaminationStatus | None = None
    landmarks: dict | None = None
