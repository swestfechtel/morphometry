"""Tests for the torsion job function with docker + encode mocked out."""
import subprocess
import uuid

import nibabel as nib
import numpy as np
import pytest

from morphometry.image_io import Image
from api.db import repository
from api.db.engine import session_scope
from api.db.models import Examination, Job


def _seed(runtime):
    """Create an examination with source volumes + a queued job; return (engine, store, ids)."""
    store, engine = runtime.get_store(), runtime.get_engine()

    def img():
        return Image.from_nibabel(nib.Nifti1Image(np.ones((4, 4, 4), dtype=np.int16), np.eye(4)))

    source_paths = {r: store.save_volume("ACC1", r, img()) for r in ("hip", "knee", "ankle", "transformed")}
    job_id = str(uuid.uuid4())
    with session_scope(engine) as s:
        repository.upsert_examination(s, Examination(id="ACC1", examination_type="torsion", source_paths=source_paths))
        repository.create_job(s, Job(id=job_id, examination_id="ACC1", kind="full"))
    return engine, store, job_id


@pytest.fixture(autouse=True)
def _no_encode(monkeypatch):
    # avoid matplotlib/multiprocessing; return one dummy base64 PNG per list
    monkeypatch.setattr("api.tasks.torsion.encode_torsion_images", lambda *a, **k: (["aGk="], ["aGk="]))


def test_run_torsion_success(runtime, fake_docker_run, monkeypatch):
    engine, store, job_id = _seed(runtime)
    monkeypatch.setattr(subprocess, "run", fake_docker_run(returncode=0))

    from api.tasks.torsion import run_torsion
    run_torsion("ACC1", job_id)

    with session_scope(engine) as s:
        ex = repository.get_examination(s, "ACC1")
        assert ex.status == "processed"
        assert ex.torsion_values["femoral_torsion_left"] == 1.0
        assert ex.torsion_values["tibial_torsion_right"] == 6.0
        assert ex.landmarks == {"femur": {}, "tibia": {}}
        assert ex.mask_paths and ex.encoded_paths["image"]
        assert repository.get_job(s, job_id).status == "finished"


def test_run_torsion_docker_failure(runtime, fake_docker_run, monkeypatch):
    engine, store, job_id = _seed(runtime)
    monkeypatch.setattr(subprocess, "run", fake_docker_run(returncode=1))

    from api.tasks.torsion import run_torsion
    with pytest.raises(RuntimeError):
        run_torsion("ACC1", job_id)

    with session_scope(engine) as s:
        assert repository.get_job(s, job_id).status == "failed"
        assert repository.get_examination(s, "ACC1").status == "failed"


def test_run_torsion_malformed_output(runtime, fake_docker_run, monkeypatch):
    engine, store, job_id = _seed(runtime)
    monkeypatch.setattr(subprocess, "run", fake_docker_run(returncode=0, malformed=True))

    from api.tasks.torsion import run_torsion
    with pytest.raises(Exception):
        run_torsion("ACC1", job_id)

    with session_scope(engine) as s:
        ex = repository.get_examination(s, "ACC1")
        assert ex.status == "failed"  # not 'processed' — no partial write of results
        assert ex.torsion_values is None
        assert repository.get_job(s, job_id).status == "failed"
