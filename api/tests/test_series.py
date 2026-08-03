"""Tests for the two-phase series-selection upload flow.

Phase 1 (`/upload/torsion/series`) enumerates the DICOM series in a mixed upload,
discards junk, stages them with previews, and records a pending examination. Phase 2
(`/upload/torsion/select`) materializes the chosen series and auto-starts the full
pipeline (docker + encode faked via the eager in-process queue).
"""
import subprocess

from api.db import repository
from api.db.engine import session_scope
from api.tests.test_ingest import _write_dicom_series


def _dicom_upload_files(paths, extra_junk=True):
    """Build the multipart ``files`` list for TestClient from on-disk .dcm paths."""
    files = [("files", (p.name, p.read_bytes(), "application/dicom")) for p in paths]
    if extra_junk:
        files.append(("files", ("README.txt", b"not a dicom file", "text/plain")))
    return files


def _fake_pipeline(monkeypatch, fake_docker_run):
    """Fake docker + image encoding so the eager queue can run the full pipeline."""
    monkeypatch.setattr(subprocess, "run", fake_docker_run(returncode=0))
    monkeypatch.setattr("api.tasks.torsion.encode_torsion_images", lambda *a, **k: (["aGk="], ["aGk="]))


def test_enumerate_groups_series_and_discards_junk(client, runtime, tmp_path):
    """A mixed upload (two series + junk) -> a pending exam with two candidate series."""
    _write_dicom_series(tmp_path / "A", n_slices=18)   # whole-leg (three z-regions)
    _write_dicom_series(tmp_path / "B", n_slices=5)     # a shorter second series
    paths = list((tmp_path / "A").glob("*.dcm")) + list((tmp_path / "B").glob("*.dcm"))

    resp = client.post("/upload/torsion/series", files=_dicom_upload_files(paths))
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["type"] == "pending"
    assert body["status"] == "pending_selection"
    assert len(body["series"]) == 2
    assert {s["instances"] for s in body["series"]} == {18, 5}
    assert all(s["preview_count"] > 0 for s in body["series"])

    acc = body["accession_number"]
    store, engine = runtime.get_store(), runtime.get_engine()
    with session_scope(engine) as s:
        row = repository.get_examination(s, acc)
        assert row is not None and row.status == "pending_selection"
        assert len(row.series) == 2
    # raw DICOM + previews are staged under the examination dir
    for series in body["series"]:
        assert store.series_incoming_dir(acc, series["uid"]).exists()
        assert store.preview_path(acc, series["uid"], 0).exists()

    # the preview endpoint serves a PNG
    uid = body["series"][0]["uid"]
    pr = client.get(f"/examinations/{acc}/series/{uid}/preview/0.png")
    assert pr.status_code == 200
    assert pr.headers["content-type"] == "image/png"


def test_select_whole_leg_materializes_and_starts_pipeline(client, runtime, tmp_path,
                                                           fake_docker_run, monkeypatch):
    """Selecting a whole-leg series materializes it and auto-runs segmentation+torsion."""
    _write_dicom_series(tmp_path / "A", n_slices=18)
    _write_dicom_series(tmp_path / "B", n_slices=5)
    paths = list((tmp_path / "A").glob("*.dcm")) + list((tmp_path / "B").glob("*.dcm"))
    body = client.post("/upload/torsion/series", files=_dicom_upload_files(paths)).json()
    acc = body["accession_number"]
    whole = max(body["series"], key=lambda s: s["instances"])  # the 18-slice series

    _fake_pipeline(monkeypatch, fake_docker_run)
    sel = client.post("/upload/torsion/select",
                      json={"examination_id": acc, "mode": "whole_leg", "series_uid": whole["uid"]})
    assert sel.status_code == 202, sel.text
    assert sel.json()["examination_id"] == acc and sel.json()["job_id"]

    store, engine = runtime.get_store(), runtime.get_engine()
    with session_scope(engine) as s:
        row = repository.get_examination(s, acc)
        assert row.status == "processed"          # eager queue ran the full pipeline
        assert row.series is None                 # candidate series cleared on materialize
        assert set(row.source_paths) == {"original", "transformed", "hip", "knee", "ankle"}
    assert not (store.examination_dir(acc) / "incoming").exists()  # staging cleaned up


def test_select_regions_materializes_three_series(client, runtime, tmp_path,
                                                  fake_docker_run, monkeypatch):
    """Selecting three separate series (regions mode) concatenates + processes them."""
    for name in ("H", "K", "A"):
        _write_dicom_series(tmp_path / name, n_slices=6)  # same in-plane shape
    paths = [p for name in ("H", "K", "A") for p in (tmp_path / name).glob("*.dcm")]
    body = client.post("/upload/torsion/series", files=_dicom_upload_files(paths)).json()
    acc = body["accession_number"]
    uids = [s["uid"] for s in body["series"]]
    assert len(uids) == 3

    _fake_pipeline(monkeypatch, fake_docker_run)
    sel = client.post("/upload/torsion/select", json={
        "examination_id": acc, "mode": "regions",
        "hip": uids[0], "knee": uids[1], "ankle": uids[2]})
    assert sel.status_code == 202, sel.text

    engine = runtime.get_engine()
    with session_scope(engine) as s:
        row = repository.get_examination(s, acc)
        assert row.status == "processed"
        assert row.knee_offset == 6 and row.ankle_offset == 12


def test_select_requires_pending_status(client, runtime, tmp_path, fake_docker_run, monkeypatch):
    """A second select on an already-materialized exam is rejected."""
    _write_dicom_series(tmp_path / "A", n_slices=18)
    paths = list((tmp_path / "A").glob("*.dcm"))
    body = client.post("/upload/torsion/series", files=_dicom_upload_files(paths)).json()
    acc, uid = body["accession_number"], body["series"][0]["uid"]

    _fake_pipeline(monkeypatch, fake_docker_run)
    first = client.post("/upload/torsion/select",
                        json={"examination_id": acc, "mode": "whole_leg", "series_uid": uid})
    assert first.status_code == 202
    second = client.post("/upload/torsion/select",
                         json={"examination_id": acc, "mode": "whole_leg", "series_uid": uid})
    assert second.status_code >= 400  # no longer awaiting selection
