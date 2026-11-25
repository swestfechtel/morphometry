"""Tests for api.version."""
from api import version


def test_api_version_env_override(monkeypatch):
    monkeypatch.setenv("MORPH_VERSION", "9.9.9")
    monkeypatch.setenv("MORPH_GIT_COMMIT", "abc")
    assert version.get_version() == "9.9.9"
    assert version.get_git_commit() == "abc"
    assert version.is_dirty() is False


def test_version_file_matches_semver(monkeypatch):
    monkeypatch.delenv("MORPH_VERSION", raising=False)
    from versioning.semver import parse_version

    parse_version(version.get_version())


def test_version_endpoint_dev_suffix(runtime, monkeypatch, tmp_path):
    import json

    from fastapi.testclient import TestClient

    from api import release_manifest as rm
    from api.main import create_app

    p = tmp_path / "images.json"
    p.write_text(json.dumps({"1.3.0": {"nnunet": "n@sha256:a", "torsion": "t@sha256:b"}}))
    monkeypatch.setattr(rm, "MANIFEST_PATH", p)
    monkeypatch.setattr("api.settings.get_version", lambda: "1.4.0")
    monkeypatch.setattr("api.routers.version.get_version", lambda: "1.4.0")
    monkeypatch.setenv("MORPH_API_AUTH_REQUIRED", "true")
    runtime.reset()
    with TestClient(create_app()) as c:  # no auth header
        r = c.get("/version")
    assert r.status_code == 200
    body = r.json()
    assert body["version"] == "1.4.0-dev"
    assert body["images_from"] == "1.3.0"
    assert body["images"] == {"nnunet": "n@sha256:a", "torsion": "t@sha256:b"}
    assert set(body) == {"version", "images_from", "git_commit", "dirty", "images"}
