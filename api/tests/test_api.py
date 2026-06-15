"""End-to-end API tests via TestClient (docker + ingest mocked; eager queue)."""
import subprocess

import nibabel as nib
import numpy as np

from morphometry.image_io import Image
from api.db import repository
from api.db.engine import session_scope
from api.db.models import Examination


def _seed_examination(runtime, accession="ACC1", status="unprocessed", with_volumes=False):
    store, engine = runtime.get_store(), runtime.get_engine()
    source_paths = {}
    if with_volumes:
        def img():
            return Image.from_nibabel(nib.Nifti1Image(np.ones((4, 4, 4), dtype=np.int16), np.eye(4)))
        source_paths = {r: store.save_volume(accession, r, img()) for r in ("hip", "knee", "ankle", "transformed")}
    with session_scope(engine) as s:
        repository.upsert_examination(s, Examination(
            id=accession, examination_type="torsion", status=status,
            study_date="2024-01-01", source_paths=source_paths))


def test_health_no_auth(client):
    # health works without the API key
    client.headers.pop("X-API-Key", None)
    assert client.get("/health").json() == {"status": "ok"}


def test_auth_required(client):
    client.headers.pop("X-API-Key", None)
    assert client.get("/examinations/").status_code == 401


def test_list_and_get_examination(client, runtime):
    _seed_examination(runtime)
    listing = client.get("/examinations/").json()
    assert [e["accession_number"] for e in listing] == ["ACC1"]
    detail = client.get("/examinations/ACC1").json()
    assert detail["type"] == "torsion"
    assert detail["status"] == "unprocessed"


def test_get_missing_examination_404(client, runtime):
    resp = client.get("/examinations/NOPE")
    assert resp.status_code == 404
    assert resp.json()["code"] == "not_found"


def test_patch_examination_whitelist(client, runtime):
    _seed_examination(runtime)
    resp = client.patch("/examinations/ACC1", json={"status": "processed", "landmarks": {"a": 1}, "evil": "x"})
    assert resp.status_code == 200
    detail = client.get("/examinations/ACC1").json()
    assert detail["status"] == "processed"


def test_delete_examination(client, runtime):
    _seed_examination(runtime)
    assert client.delete("/examinations/ACC1").status_code == 205
    assert client.get("/examinations/ACC1").status_code == 404


def test_dispatch_torsion_eager_runs_pipeline(client, runtime, fake_docker_run, monkeypatch):
    _seed_examination(runtime, with_volumes=True)
    monkeypatch.setattr(subprocess, "run", fake_docker_run(returncode=0))
    monkeypatch.setattr("api.tasks.torsion.encode_torsion_images", lambda *a, **k: (["aGk="], ["aGk="]))

    created = client.post("/model/torsion/ACC1").json()
    job_id = created["job_id"]
    assert job_id != "ACC1"  # distinct uuid, not the accession

    status = client.get(f"/jobs/{job_id}").json()
    assert status["status"] == "finished"  # eager queue ran it inline
    assert client.get("/examinations/ACC1").json()["status"] == "processed"


def test_dispatch_missing_examination_404(client, runtime):
    assert client.post("/model/torsion/NOPE").status_code == 404


def test_upload_torsion_wiring(client, runtime, monkeypatch):
    # avoid real DICOM: stub the ingest, assert the endpoint plumbing returns the id
    monkeypatch.setattr("api.ingest.dicom.ingest_torsion_from_dir", lambda d: "NEWACC")
    resp = client.post("/upload/", data={"examination_type": "torsion"},
                       files={"files": ("a.dcm", b"fake", "application/dicom")})
    assert resp.status_code == 201
    assert resp.json() == {"examination_id": "NEWACC"}


def test_orthanc_stage_and_finalize(client, runtime, monkeypatch):
    metadata = '{"AccessionNumber": "ORT1", "0x0008,0x1030": "MRT Beinachsenmessung",' \
               ' "0x0008,0x103e": "T2 TSE ax 3 Stacks", "0008,0018": "uid-1"}'
    resp = client.post("/upload/orthanc", data={"metadata": metadata},
                       files={"file": ("inst", b"dicom-bytes", "application/dicom")})
    assert resp.status_code == 202
    # instance staged; eager finalize returned early due to debounce (file too fresh)
    assert len(runtime.get_store().incoming_files("ORT1")) == 1
    with session_scope(runtime.get_engine()) as s:
        assert repository.get_examination(s, "ORT1") is None

    # force finalize with no debounce; stub ingest + the model run
    monkeypatch.setattr("api.tasks.orthanc.ingest_torsion_from_dir", lambda d: _make_row(runtime, "ORT1"))
    monkeypatch.setattr("api.tasks.orthanc.run_torsion", lambda *a, **k: None)
    from api.tasks.orthanc import finalize_orthanc
    finalize_orthanc("ORT1", debounce_seconds=0)
    with session_scope(runtime.get_engine()) as s:
        assert repository.get_examination(s, "ORT1") is not None
    assert runtime.get_store().incoming_files("ORT1") == []  # staging cleared


def test_orthanc_no_matching_rule(client, runtime):
    resp = client.post("/upload/orthanc", data={"metadata": '{"AccessionNumber": "X", "0008,1030": "other"}'},
                       files={"file": ("inst", b"bytes", "application/dicom")})
    assert resp.status_code == 400


def _make_row(runtime, accession):
    with session_scope(runtime.get_engine()) as s:
        repository.upsert_examination(s, Examination(id=accession, examination_type="torsion"))
    return accession
