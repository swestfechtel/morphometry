"""Examination request/response schemas."""
from typing import Literal

from pydantic import BaseModel

from api.schemas.enums import ExaminationStatus, ExaminationType


class ExaminationSummary(BaseModel):
    """List-view summary of an examination (no images)."""
    patient_name: str = "Anonymised"
    study_date: str | None = None
    study_time: str | None = None
    study_description: str | None = None
    accession_number: str
    status: ExaminationStatus


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


class XRayDetail(ExaminationSummary):
    """Detail view for an x-ray examination."""
    type: Literal["xray"] = "xray"
    image: str | None = None
    landmarks: dict = {}


class ExaminationUpdate(BaseModel):
    """Whitelisted, validated fields a client may PATCH (no blind setattr)."""
    status: ExaminationStatus | None = None
    landmarks: dict | None = None
