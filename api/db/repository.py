"""CRUD helpers over the SQLModel session.

These replace the metadata responsibilities of the old FileController (the image
side moves to ``api.storage``). All functions take an explicit Session so they
are trivially testable and free of global state.
"""
from datetime import datetime, timezone

from sqlmodel import Session, select

from api.db.models import Examination, Job
from api.schemas.enums import JobState


def _touch(examination: Examination) -> None:
    examination.updated_at = datetime.now(timezone.utc)


# --- examinations ------------------------------------------------------------
def upsert_examination(session: Session, examination: Examination) -> Examination:
    """Insert or update an examination row."""
    _touch(examination)
    session.merge(examination)
    session.flush()
    return examination


def get_examination(session: Session, examination_id: str) -> Examination | None:
    return session.get(Examination, examination_id)


def list_examinations(session: Session) -> list[Examination]:
    return list(session.exec(select(Examination)).all())


def delete_examination(session: Session, examination_id: str) -> bool:
    examination = session.get(Examination, examination_id)
    if examination is None:
        return False
    # remove dependent job rows first (jobs.examination_id FK -> examinations.id);
    # flush so the job DELETEs execute before the examination DELETE (no relationship
    # is defined, so the unit-of-work won't order them for us).
    for job in session.exec(select(Job).where(Job.examination_id == examination_id)).all():
        session.delete(job)
    session.flush()
    session.delete(examination)
    return True


# --- jobs --------------------------------------------------------------------
def create_job(session: Session, job: Job) -> Job:
    session.add(job)
    session.flush()
    return job


def get_job(session: Session, job_id: str) -> Job | None:
    return session.get(Job, job_id)


def update_job(session: Session, job: Job) -> Job:
    session.add(job)
    session.flush()
    return job


def list_jobs_by_status(session: Session, status: str) -> list[Job]:
    return list(session.exec(select(Job).where(Job.status == status)).all())


def active_jobs_by_examination(session: Session) -> dict[str, str]:
    """Map each examination id to its latest still-active (queued/running) job id.

    Used by the examinations list so a row can tell it has processing in flight
    (hide "Start Processing", show progress) and knows which job to poll.
    """
    active = (JobState.QUEUED.value, JobState.RUNNING.value)
    jobs = session.exec(
        select(Job).where(Job.status.in_(active)).order_by(Job.created_at)
    ).all()
    # created_at ascending -> a newer job overwrites an older one, so latest wins
    return {job.examination_id: job.id for job in jobs}
