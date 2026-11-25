"""Tests for finding examinations by release version (digest match) and GET /releases."""
import json

import pytest
from fastapi.testclient import TestClient

from api import release_manifest as rm
from api.db.engine import session_scope
from api.db.models import Examination
from api.main import create_app

N1, N2 = "swestfechtel/nnunet_torsion@sha256:n1", "swestfechtel/nnunet_torsion@sha256:n2"
T10, T11 = "swestfechtel/torsion@sha256:t10", "swestfechtel/torsion@sha256:t11"
MANIFEST = {
    "1.0.0": {"nnunet": N1, "torsion": T10, "api": "a@sha256:a0", "frontend": "f@sha256:f0", "date": "2026-01-01"},
    "1.1.0": {"nnunet": N1, "torsion": T11, "api": "a@sha256:a1", "frontend": "f@sha256:f1", "date": "2026-02-01"},
}


def _rec(digest, released=True):
    return {"digest": digest, "released": released, "image": "x", "api_version": "1"}


EXAMS = {
    "A": {"segmentation": _rec(N1), "torsion": _rec(T10)},          # 1.0.0
    "B": {"segmentation": _rec(N1), "torsion": _rec(T11)},          # 1.1.0 (shares nnunet)
    "C": {"segmentation": _rec("local-id", False), "torsion": _rec("local-id2", False)},  # unreleased
    "D": None,                                                       # no provenance
    "E": {"torsion": _rec(T11)},                                     # no segmentation key
}


@pytest.fixture
def api(runtime, monkeypatch, tmp_path):
    p = tmp_path / "images.json"
    p.write_text(json.dumps(MANIFEST))
    monkeypatch.setattr(rm, "MANIFEST_PATH", p)
    with session_scope(runtime.get_engine()) as s:
        for eid, prov in EXAMS.items():
            s.add(Examination(id=eid, provenance=prov))
    with TestClient(create_app()) as c:
        c.headers.update({"X-API-Key": "test-key"})
        yield c


def ids(resp):
    assert resp.status_code == 200, resp.text
    return sorted(e["accession_number"] for e in resp.json()["examinations"])


def test_filter_torsion_by_version(api):
    assert ids(api.get("/examinations/?version=1.1.0&stage=torsion")) == ["B", "E"]


def test_filter_shared_nnunet_digest(api):
    assert ids(api.get("/examinations/?version=1.0.0&stage=segmentation")) == ["A", "B"]


def test_filter_any_stage(api):
    assert ids(api.get("/examinations/?version=1.0.0")) == ["A", "B"]
    assert ids(api.get("/examinations/?version=1.1.0")) == ["A", "B", "E"]


def test_filter_unreleased(api):
    assert ids(api.get("/examinations/?version=unreleased")) == ["C"]
    assert ids(api.get("/examinations/?version=unreleased&stage=torsion")) == ["C"]


def test_filter_none(api):
    assert ids(api.get("/examinations/?version=none")) == ["D", "E"]
    assert ids(api.get("/examinations/?version=none&stage=torsion")) == ["D"]
    assert ids(api.get("/examinations/?version=none&stage=segmentation")) == ["D", "E"]


def test_unknown_version_404(api):
    r = api.get("/examinations/?version=9.9.9")
    assert r.status_code == 404
    assert "1.0.0" in r.text and "1.1.0" in r.text


def test_bad_stage_422(api):
    assert api.get("/examinations/?version=1.0.0&stage=bogus").status_code == 422


def test_no_filter_unchanged(api):
    assert ids(api.get("/examinations/")) == ["A", "B", "C", "D", "E"]


def test_image_versions_in_summary(api):
    rows = {e["accession_number"]: e for e in api.get("/examinations/").json()["examinations"]}
    assert rows["A"]["image_versions"] == {"segmentation": ["1.0.0", "1.1.0"], "torsion": ["1.0.0"]}
    assert rows["B"]["image_versions"]["torsion"] == ["1.1.0"]
    assert rows["C"]["image_versions"] == {"segmentation": [], "torsion": []}
    assert rows["D"]["image_versions"] == {}


def test_unparseable_manifest_entry_skipped(api, monkeypatch, tmp_path):
    p = tmp_path / "bad.json"
    p.write_text(json.dumps({**MANIFEST, "junk": {"nnunet": N1, "torsion": T10}}))
    monkeypatch.setattr(rm, "MANIFEST_PATH", p)
    assert [r["version"] for r in api.get("/releases").json()["releases"]] == ["1.1.0", "1.0.0"]
    assert ids(api.get("/examinations/?version=1.0.0&stage=segmentation")) == ["A", "B"]
    rows = {e["accession_number"]: e for e in api.get("/examinations/").json()["examinations"]}
    assert rows["A"]["image_versions"]["segmentation"] == ["1.0.0", "1.1.0"]


def test_releases_endpoint(runtime, monkeypatch, tmp_path):
    p = tmp_path / "images.json"
    p.write_text(json.dumps(MANIFEST))
    monkeypatch.setattr(rm, "MANIFEST_PATH", p)
    monkeypatch.setenv("MORPH_API_AUTH_REQUIRED", "true")
    runtime.reset()
    with TestClient(create_app()) as c:  # no auth header
        r = c.get("/releases")
    assert r.status_code == 200
    rel = r.json()["releases"]
    assert [x["version"] for x in rel] == ["1.1.0", "1.0.0"]
    assert rel[0]["date"] == "2026-02-01"
    assert rel[0]["images"] == {"nnunet": N1, "torsion": T11, "api": "a@sha256:a1", "frontend": "f@sha256:f1"}


def test_manifest_helpers():
    assert rm.stage_digests("1.1.0", MANIFEST) == {"segmentation": N1, "torsion": T11}
    with pytest.raises(KeyError):
        rm.stage_digests("2.0.0", MANIFEST)
    assert rm.versions_for_digest("segmentation", N1, MANIFEST) == ["1.0.0", "1.1.0"]
    assert rm.versions_for_digest("torsion", None, MANIFEST) == []
