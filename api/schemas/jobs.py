"""Job request/response schemas."""
from pydantic import BaseModel

from api.schemas.enums import JobKind, JobState


class JobCreated(BaseModel):
    """Returned by the model-dispatch endpoints (202 Accepted)."""
    job_id: str
    examination_id: str


class JobStatus(BaseModel):
    """Durable job status, read from the database."""
    job_id: str
    examination_id: str
    kind: JobKind
    status: JobState
    error: str | None = None
