"""End-to-end API tests via TestClient (docker + ingest mocked; eager queue)."""
import subprocess

import nibabel as nib
import numpy as np

from morphometry.image_io import Image
from api.db import repository
from api.db.engine import session_scope
from api.db.models import Examination


def _img():
    return Image.from_nibabel(nib.Nifti1Image(np.ones((4, 4, 4), dtype=np.int16), np.eye(4)))


def _seed_examination(runtime, accession="ACC1", status="unprocessed", *, with_volumes=False,
                      with_masks=False, with_encoded=False, torsion=None, landmarks=None,
                      examination_type="torsion"):
    store, engine = runtime.get_store(), runtime.get_engine()
    fields = {}
    if with_volumes:
        fields["source_paths"] = {r: store.save_volume(accession, r, _img())
                                  for r in ("hip", "knee", "ankle", "transformed")}
        fields["knee_offset"], fields["ankle_offset"] = 4, 8
    if with_masks:
        fields["mask_paths"] = {r: store.save_mask(accession, r, _img()) for r in ("hip", "knee", "ankle")}
    if with_encoded:
        fields["encoded_paths"] = store.save_encoded(accession, ["aGk=", "aGk="], ["aGk=", "aGk="])
    with session_scope(engine) as s:
        repository.upsert_examination(s, Examination(
            id=accession, examination_type=examination_type, status=status,
            study_date="2024-01-01", torsion_values=torsion, landmarks=landmarks, **fields))


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
    assert [e["accession_number"] for e in listing["examinations"]] == ["ACC1"]
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
    resp = client.delete("/examinations/ACC1")
    assert resp.status_code == 205
    assert resp.content == b""  # 205 must carry no body
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


# --- additional endpoint+state coverage -------------------------------------
def test_auth_valid_and_invalid_key(client, runtime):
    _seed_examination(runtime)
    assert client.get("/examinations/").status_code == 200          # valid key (set by fixture)
    client.headers["X-API-Key"] = "wrong-key"
    assert client.get("/examinations/").status_code == 401


def test_patch_landmarks_roundtrip(client, runtime):
    _seed_examination(runtime, status="processed", with_encoded=True)
    lm = {"femur": {"Lee": {"left": {"p": [1.0, 2.0, 3.0]}}}, "tibia": {}}
    assert client.patch("/examinations/ACC1", json={"landmarks": lm}).status_code == 200
    detail = client.get("/examinations/ACC1").json()
    assert detail["landmarks"] == lm


def test_patch_missing_examination_404(client, runtime):
    assert client.patch("/examinations/NOPE", json={"status": "processed"}).status_code == 404


def test_torsion_detail_serves_encoded_images(client, runtime):
    _seed_examination(runtime, status="processed", with_encoded=True,
                      torsion={"femoral_torsion_left": 9.0, "femoral_torsion_right": 8.0,
                               "tibial_torsion_left": 7.0, "tibial_torsion_right": 6.0})
    d = client.get("/examinations/ACC1").json()
    assert d["type"] == "torsion"
    assert len(d["image"]) == 2 and len(d["segmentation"]) == 2  # base64 PNGs read from disk
    assert d["torsion"]["femoral_torsion_left"] == 9.0


def test_xray_detail(client, runtime):
    _seed_examination(runtime, accession="XR1", status="processed", with_encoded=True,
                      examination_type="x_ray_foot_ap", landmarks={"axis": {"start": [1, 2], "end": [3, 4]}})
    d = client.get("/examinations/XR1").json()
    assert d["type"] == "xray"
    assert isinstance(d["image"], str) and d["image"]              # single base64 image
    assert d["landmarks"]["axis"]["start"] == [1, 2]


def test_compute_segmentation_endpoint(client, runtime, fake_docker_run, monkeypatch):
    _seed_examination(runtime, with_volumes=True)
    monkeypatch.setattr(subprocess, "run", fake_docker_run(returncode=0))
    monkeypatch.setattr("api.tasks.torsion.encode_torsion_images", lambda *a, **k: (["aGk="], ["aGk="]))

    job_id = client.post("/model/segmentation/ACC1").json()["job_id"]
    assert client.get(f"/jobs/{job_id}").json()["status"] == "finished"
    assert client.get("/examinations/ACC1").json()["status"] == "segmented"  # not processed (seg only)


def test_rerun_torsion_on_segmented(client, runtime, fake_docker_run, monkeypatch):
    # an already-segmented examination dispatches mode='torsion' (skip segmentation)
    _seed_examination(runtime, status="segmented", with_masks=True)
    monkeypatch.setattr(subprocess, "run", fake_docker_run(returncode=0))
    # encode must NOT be needed on the torsion-only path; make it explode if called
    monkeypatch.setattr("api.tasks.torsion.encode_torsion_images",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("encode should not run")))

    created = client.post("/model/torsion/ACC1").json()
    assert client.get(f"/jobs/{created['job_id']}").json()["status"] == "finished"
    d = client.get("/examinations/ACC1").json()
    assert d["status"] == "processed"
    assert d["torsion"]["femoral_torsion_left"] == 1.0  # from fake torsion container


def test_job_failed_status_surfaced(client, runtime, fake_docker_run, monkeypatch):
    _seed_examination(runtime, with_volumes=True)
    monkeypatch.setattr(subprocess, "run", fake_docker_run(returncode=1))  # container fails

    job_id = client.post("/model/torsion/ACC1").json()["job_id"]
    status = client.get(f"/jobs/{job_id}").json()
    assert status["status"] == "failed"
    assert status["error"]
    assert client.get("/examinations/ACC1").json()["status"] == "failed"


def test_job_404(client, runtime):
    assert client.get("/jobs/does-not-exist").status_code == 404


def test_job_status_durable_across_restart(client, runtime, fake_docker_run, monkeypatch):
    _seed_examination(runtime, with_volumes=True)
    monkeypatch.setattr(subprocess, "run", fake_docker_run(returncode=0))
    monkeypatch.setattr("api.tasks.torsion.encode_torsion_images", lambda *a, **k: (["aGk="], ["aGk="]))
    job_id = client.post("/model/torsion/ACC1").json()["job_id"]

    # a brand-new app/client (simulated restart) sees the same persisted job
    from fastapi.testclient import TestClient
    from api.deps import get_queue
    from api.main import create_app
    from api.tasks.queue import EagerQueue
    app2 = create_app()
    app2.dependency_overrides[get_queue] = lambda: EagerQueue()
    with TestClient(app2) as c2:
        c2.headers["X-API-Key"] = "test-key"
        assert c2.get(f"/jobs/{job_id}").json()["status"] == "finished"


def test_orphaned_running_job_reconciled_on_startup(runtime):
    # a job left 'running' by a crashed worker is marked failed on app startup
    from fastapi.testclient import TestClient
    from api.db.models import Job
    from api.main import create_app

    with session_scope(runtime.get_engine()) as s:
        repository.upsert_examination(s, Examination(id="ACCR", examination_type="torsion"))
        repository.create_job(s, Job(id="stalejob", examination_id="ACCR", kind="full", status="running"))

    with TestClient(create_app()):  # triggers lifespan reconciliation
        pass
    with session_scope(runtime.get_engine()) as s:
        job = repository.get_job(s, "stalejob")
        assert job.status == "failed"
        assert "restart" in (job.error or "").lower()


def test_upload_torsion_multi_wiring(client, runtime, monkeypatch):
    monkeypatch.setattr("api.ingest.dicom.ingest_torsion_multi_from_dirs", lambda h, k, a: "MULTI1")
    files = [(field, (f"{field}.dcm", b"fake", "application/dicom"))
             for field in ("hip_files", "knee_files", "ankle_files")]
    resp = client.post("/upload/torsion/multi", files=files)
    assert resp.status_code == 201
    assert resp.json() == {"examination_id": "MULTI1"}
