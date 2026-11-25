"""Tests for api.release_manifest and the manifest-backed Settings image defaults."""
import json
import logging
import subprocess

import pytest

from api import release_manifest as rm
from api.settings import Settings

E13 = {"nnunet": "swestfechtel/nnunet_torsion@sha256:a", "torsion": "swestfechtel/torsion@sha256:b"}
E12 = {"nnunet": "swestfechtel/nnunet_torsion@sha256:c", "torsion": "swestfechtel/torsion@sha256:d"}


def test_images_for_newest_lower():
    assert rm.images_for("1.4.0", {"1.3.0": E13, "1.2.0": E12}) == ("1.3.0", E13)


def test_images_for_exact_wins():
    assert rm.images_for("1.2.0", {"1.3.0": E13, "1.2.0": E12}) == ("1.2.0", E12)


def test_images_for_numeric_not_lexical():
    m = {"1.9.0": E12, "1.10.0": E13}
    assert rm.images_for("1.11.0", m) == ("1.10.0", E13)


def test_images_for_empty_and_none_lower():
    assert rm.images_for("1.4.0", {}) == (None, {})
    assert rm.images_for("1.2.0", {"1.3.0": E13}) == (None, {})


def test_load_manifest_missing_and_present(tmp_path):
    assert rm.load_manifest(tmp_path / "nope.json") == {}
    p = tmp_path / "m.json"
    p.write_text(json.dumps({"1.3.0": E13}))
    assert rm.load_manifest(p) == {"1.3.0": E13}


def test_released_digests():
    assert rm.released_digests({"1.3.0": E13, "1.2.0": E12}) == set(E13.values()) | set(E12.values())


@pytest.fixture
def manifest(tmp_path, monkeypatch):
    p = tmp_path / "images.json"
    p.write_text(json.dumps({"1.3.0": E13}))
    monkeypatch.setattr(rm, "MANIFEST_PATH", p)
    monkeypatch.setattr("api.settings.get_version", lambda: "1.4.0")
    monkeypatch.setattr("api.settings._warned_latest", False, raising=False)
    return p


def test_settings_env_wins(manifest):
    s = Settings(nnunet_image="x/n:1", torsion_image="x/t:1")
    assert s.resolved_nnunet_image == "x/n:1"
    assert s.resolved_torsion_image == "x/t:1"


def test_settings_manifest_default(manifest):
    s = Settings()
    assert s.resolved_nnunet_image == E13["nnunet"]
    assert s.resolved_torsion_image == E13["torsion"]


def test_settings_latest_fallback_warns(manifest, caplog):
    manifest.write_text("{}")
    s = Settings()
    with caplog.at_level(logging.WARNING):
        assert s.resolved_nnunet_image == "swestfechtel/nnunet_torsion:latest"
        assert s.resolved_torsion_image == "swestfechtel/torsion:latest"
        s.resolved_torsion_image
    assert sum("latest" in r.message for r in caplog.records) == 1


def test_job_tmp_dir_used(runtime, fake_docker_run, monkeypatch, tmp_path):
    from api.tests.test_tasks import _seed
    monkeypatch.setenv("MORPH_API_JOB_TMP_DIR", str(tmp_path / "jobs"))
    runtime.reset()
    engine, store, job_id = _seed(runtime)
    monkeypatch.setattr("api.tasks.torsion.encode_torsion_images", lambda *a, **k: (["aGk="], ["aGk="]))
    seen = []
    inner = fake_docker_run(returncode=0)

    def spy(cmd, *a, **k):
        seen.append(next(c for c in cmd if ":/app/" in c))
        return inner(cmd, *a, **k)

    monkeypatch.setattr(subprocess, "run", spy)
    from api.tasks.torsion import run_torsion
    run_torsion("ACC1", job_id)
    assert seen and all(m.startswith(str(tmp_path / "jobs")) for m in seen)
