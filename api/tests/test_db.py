"""Unit tests for the DB layer (engine + repository) against a temp SQLite file."""
import uuid

import pytest

from api.db import repository
from api.db.engine import init_db, make_engine, session_scope
from api.db.models import Examination, Job
from api.schemas.enums import ExaminationStatus, JobState


@pytest.fixture
def engine(tmp_path):
    eng = make_engine(f"sqlite:///{tmp_path/'t.db'}")
    init_db(eng)
    return eng


def test_examination_roundtrip_and_update(engine):
    with session_scope(engine) as s:
        repository.upsert_examination(s, Examination(
            id="ACC1", examination_type="torsion", study_date="2024-01-01",
            dicom_metadata={"0008,0050": "ACC1"}, source_paths={"hip": "ACC1/source/hip.nii.gz"}))

    with session_scope(engine) as s:
        ex = repository.get_examination(s, "ACC1")
        assert ex is not None and ex.source_paths["hip"].endswith("hip.nii.gz")
        first_updated = ex.updated_at
        ex.status = ExaminationStatus.PROCESSED.value
        ex.torsion_values = {"femoral_torsion_left": 1.0}
        repository.upsert_examination(s, ex)

    with session_scope(engine) as s:
        ex = repository.get_examination(s, "ACC1")
        assert ex.status == "processed"
        assert ex.torsion_values["femoral_torsion_left"] == 1.0
        assert ex.updated_at >= first_updated
        assert [e.id for e in repository.list_examinations(s)] == ["ACC1"]


def test_delete_examination(engine):
    with session_scope(engine) as s:
        repository.upsert_examination(s, Examination(id="ACC2", examination_type="torsion"))
    with session_scope(engine) as s:
        assert repository.delete_examination(s, "ACC2") is True
    with session_scope(engine) as s:
        assert repository.get_examination(s, "ACC2") is None
        assert repository.delete_examination(s, "missing") is False


def test_delete_examination_with_jobs(engine):
    # deleting an examination must also remove its dependent job rows (FK constraint)
    with session_scope(engine) as s:
        repository.upsert_examination(s, Examination(id="ACCJ", examination_type="torsion"))
        repository.create_job(s, Job(id=str(uuid.uuid4()), examination_id="ACCJ", kind="full"))
        repository.create_job(s, Job(id=str(uuid.uuid4()), examination_id="ACCJ", kind="torsion"))
    with session_scope(engine) as s:
        assert repository.delete_examination(s, "ACCJ") is True
    with session_scope(engine) as s:
        assert repository.get_examination(s, "ACCJ") is None
        assert repository.list_jobs_by_status(s, "queued") == []


def test_job_lifecycle(engine):
    with session_scope(engine) as s:
        repository.upsert_examination(s, Examination(id="ACC3", examination_type="torsion"))
        job_id = str(uuid.uuid4())
        repository.create_job(s, Job(id=job_id, examination_id="ACC3", kind="full"))

    with session_scope(engine) as s:
        job = repository.get_job(s, job_id)
        assert job.status == "queued"
        job.status = JobState.RUNNING.value
        repository.update_job(s, job)

    with session_scope(engine) as s:
        assert len(repository.list_jobs_by_status(s, "running")) == 1
        assert repository.get_job(s, job_id).status == "running"
        # distinct job id, not the accession (fixes the legacy collision)
        assert job_id != "ACC3"
