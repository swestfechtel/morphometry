"""SQLModel table definitions.

Only lightweight metadata lives in the database; large images are files on disk
(paths recorded in the JSON columns here). Irregular/nested data (torsion values,
landmarks, DICOM tag subset, file-path maps, shape) are JSON blobs read whole;
scalars used for listing/filtering are real columns.
"""
from datetime import datetime, timezone

from sqlalchemy import JSON, Column
from sqlmodel import Field, SQLModel

from api.schemas.enums import ExaminationStatus, ExaminationType, JobKind, JobState


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Examination(SQLModel, table=True):
    __tablename__ = "examinations"

    id: str = Field(primary_key=True)  # = accession number
    examination_type: str = ExaminationType.TORSION.value
    status: str = ExaminationStatus.UNPROCESSED.value

    patient_name: str | None = None
    study_date: str | None = None
    study_time: str | None = None
    study_description: str | None = None

    dicom_metadata: dict = Field(default_factory=dict, sa_column=Column(JSON))
    torsion_values: dict | None = Field(default=None, sa_column=Column(JSON))
    landmarks: dict | None = Field(default=None, sa_column=Column(JSON))
    source_paths: dict = Field(default_factory=dict, sa_column=Column(JSON))
    mask_paths: dict | None = Field(default=None, sa_column=Column(JSON))
    encoded_paths: dict | None = Field(default=None, sa_column=Column(JSON))
    shape: list | None = Field(default=None, sa_column=Column(JSON))
    knee_offset: int | None = None   # hip sub-volume slice count
    ankle_offset: int | None = None  # hip + knee sub-volume slice count

    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)


class Job(SQLModel, table=True):
    __tablename__ = "jobs"

    id: str = Field(primary_key=True)  # uuid4 (distinct from examination id)
    examination_id: str = Field(foreign_key="examinations.id", index=True)
    kind: str = JobKind.FULL.value
    status: str = JobState.QUEUED.value
    rq_job_id: str | None = None
    error: str | None = None

    created_at: datetime = Field(default_factory=_utcnow)
    started_at: datetime | None = None
    finished_at: datetime | None = None
