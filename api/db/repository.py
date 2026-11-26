"""CRUD helpers over the SQLModel session.

These replace the metadata responsibilities of the old FileController (the image
side moves to ``api.storage``). All functions take an explicit Session so they
are trivially testable and free of global state.
"""
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal

from sqlalchemy import and_, func, or_
from sqlmodel import Session, select

from api.db.models import Examination, Job, User
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


@dataclass(frozen=True)
class VersionFilter:
    """Provenance-based filter for :func:`list_examinations`.

    :param stages: provenance stage keys to inspect (``segmentation`` and/or ``torsion``); OR-combined.
    :param mode: ``digest`` (recorded digest equals ``digests[stage]``), ``unreleased`` (stage record has
        ``released: false``) or ``none`` (no record for the stage, incl. null provenance).
    :param digests: stage -> image ref to match; used by mode ``digest`` only.
    """
    stages: tuple[str, ...]
    mode: Literal["digest", "unreleased", "none"]
    digests: dict[str, str] = field(default_factory=dict)


def _version_clause(vf: VersionFilter):
    """Build the SQL WHERE clause for a :class:`VersionFilter` (json_extract on the provenance column)."""
    clauses = []
    for stage in vf.stages:
        if vf.mode == "digest":
            if stage in vf.digests:
                clauses.append(func.json_extract(Examination.provenance, f"$.{stage}.digest") == vf.digests[stage])
        elif vf.mode == "unreleased":
            # JSON false is extracted as integer 0
            clauses.append(func.json_extract(Examination.provenance, f"$.{stage}.released") == 0)
        else:
            clauses.append(func.json_extract(Examination.provenance, f"$.{stage}").is_(None))
    return or_(*clauses) if clauses else and_(False)


def list_examinations(session: Session, *, version_filter: VersionFilter | None = None) -> list[Examination]:
    """List examinations, optionally restricted by recorded image provenance.

    :param session: open DB session.
    :param version_filter: if given, only rows matching it (evaluated in SQL); ``None`` returns all rows.
    :return: the matching examinations.
    """
    stmt = select(Examination)
    if version_filter is not None:
        stmt = stmt.where(_version_clause(version_filter))
    return list(session.exec(stmt).all())


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


# --- users -------------------------------------------------------------------
def get_user(session: Session, username: str) -> User | None:
    return session.get(User, username)


def list_users(session: Session) -> list[User]:
    return list(session.exec(select(User).order_by(User.username)).all())


def upsert_user(session: Session, user: User) -> User:
    user.updated_at = datetime.now(timezone.utc)
    session.merge(user)
    session.flush()
    return user


def delete_user(session: Session, username: str) -> bool:
    user = session.get(User, username)
    if user is None:
        return False
    session.delete(user)
    return True


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
