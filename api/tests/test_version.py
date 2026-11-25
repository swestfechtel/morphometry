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


import pytest  # noqa: E402


@pytest.mark.parametrize("running", ["0.0.0-dev", "garbage"])
def test_version_endpoint_non_semver(runtime, monkeypatch, tmp_path, running):
    """A non-semver MORPH_VERSION (the api Dockerfile default is 0.0.0-dev) is a 200, not a 500."""
    import json

    from fastapi.testclient import TestClient

    from api import release_manifest as rm
    from api.main import create_app

    p = tmp_path / "images.json"
    p.write_text(json.dumps({"1.0.0": {"nnunet": "n@sha256:a", "torsion": "t@sha256:b"}}))
    monkeypatch.setattr(rm, "MANIFEST_PATH", p)
    monkeypatch.setenv("MORPH_VERSION", running)
    monkeypatch.delenv("MORPH_API_NNUNET_IMAGE", raising=False)
    monkeypatch.delenv("MORPH_API_TORSION_IMAGE", raising=False)
    runtime.reset()
    with TestClient(create_app()) as c:
        r = c.get("/version")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["version"] == running
    assert body["images_from"] is None
    assert body["images"] == {"nnunet": "swestfechtel/nnunet_torsion:latest", "torsion": "swestfechtel/torsion:latest"}


def test_version_endpoint_corrupt_manifest(runtime, monkeypatch, tmp_path):
    from fastapi.testclient import TestClient

    from api import release_manifest as rm
    from api.main import create_app

    p = tmp_path / "images.json"
    p.write_text("{corrupt")
    monkeypatch.setattr(rm, "MANIFEST_PATH", p)
    monkeypatch.setenv("MORPH_VERSION", "1.0.0")
    runtime.reset()
    with TestClient(create_app()) as c:
        assert c.get("/version").status_code == 200
        assert c.get("/releases").json() == {"releases": []}
