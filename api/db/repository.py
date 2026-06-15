"""CRUD helpers over the SQLModel session.

These replace the metadata responsibilities of the old FileController (the image
side moves to ``api.storage``). All functions take an explicit Session so they
are trivially testable and free of global state.
"""
from datetime import datetime, timezone

from sqlmodel import Session, select

from api.db.models import Examination, Job


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
