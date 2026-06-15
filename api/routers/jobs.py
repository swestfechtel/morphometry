"""Model-dispatch and job-status endpoints."""
import uuid

from fastapi import APIRouter, Depends, status
from sqlmodel import Session

from api import runtime
from api.db import repository
from api.db.engine import session_scope
from api.db.models import Job
from api.deps import get_queue, get_session
from api.errors import NotFoundError
from api.schemas.enums import ExaminationStatus, JobKind, JobState
from api.schemas.jobs import JobCreated, JobStatus
from api.tasks.queue import TaskQueue

router = APIRouter(tags=["jobs"])

_TORSION_TASK = "api.tasks.torsion.run_torsion"


def _dispatch(examination_id: str, kind: JobKind, mode: str, queue: TaskQueue) -> JobCreated:
    """Create a (committed) job row, enqueue it, then record the RQ id.

    The job row is committed before enqueueing so the worker (a separate process —
    or the eager in-process queue) reliably sees it.
    """
    engine = runtime.get_engine()
    job_id = str(uuid.uuid4())
    with session_scope(engine) as session:
        if repository.get_examination(session, examination_id) is None:
            raise NotFoundError(f"Examination {examination_id} not found")
        repository.create_job(session, Job(id=job_id, examination_id=examination_id, kind=kind.value))

    rq_id = queue.enqueue(_TORSION_TASK, examination_id, job_id, mode)

    with session_scope(engine) as session:
        job = repository.get_job(session, job_id)
        job.rq_job_id = rq_id
        repository.update_job(session, job)
    return JobCreated(job_id=job_id, examination_id=examination_id)


@router.post("/model/torsion/{examination_id}", status_code=status.HTTP_202_ACCEPTED, response_model=JobCreated)
def compute_torsion(examination_id: str, session: Session = Depends(get_session),
                    queue: TaskQueue = Depends(get_queue)):
    examination = repository.get_examination(session, examination_id)
    if examination is None:
        raise NotFoundError(f"Examination {examination_id} not found")
    # full pipeline if not yet segmented, otherwise just the torsion stage
    if examination.status == ExaminationStatus.UNPROCESSED.value:
        return _dispatch(examination_id, JobKind.FULL, "full", queue)
    return _dispatch(examination_id, JobKind.TORSION, "torsion", queue)


@router.post("/model/segmentation/{examination_id}", status_code=status.HTTP_202_ACCEPTED, response_model=JobCreated)
def compute_segmentation(examination_id: str, session: Session = Depends(get_session),
                         queue: TaskQueue = Depends(get_queue)):
    if repository.get_examination(session, examination_id) is None:
        raise NotFoundError(f"Examination {examination_id} not found")
    return _dispatch(examination_id, JobKind.SEGMENTATION, "segmentation", queue)


@router.get("/jobs/{job_id}", response_model=JobStatus)
def get_job(job_id: str, session: Session = Depends(get_session)):
    job = repository.get_job(session, job_id)
    if job is None:
        raise NotFoundError(f"Job {job_id} not found")
    return JobStatus(job_id=job.id, examination_id=job.examination_id,
                     kind=JobKind(job.kind), status=JobState(job.status), error=job.error)
