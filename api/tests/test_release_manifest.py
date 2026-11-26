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


# --- tolerance: non-semver running versions, unparseable keys, corrupt files ----------------------


def test_images_for_strips_suffix():
    assert rm.images_for("1.4.0-dev", {"1.3.0": E13}) == ("1.3.0", E13)
    assert rm.images_for("1.3.0-dev", {"1.3.0": E13, "1.2.0": E12}) == ("1.3.0", E13)
    assert rm.images_for("0.0.0-dev", {"1.3.0": E13}) == (None, {})


def test_images_for_garbage_version(caplog):
    with caplog.at_level(logging.WARNING):
        assert rm.images_for("garbage", {"1.3.0": E13}) == (None, {})
    assert any("garbage" in r.message for r in caplog.records)


def test_images_for_skips_unparseable_manifest_keys():
    assert rm.images_for("1.4.0", {"junk": E12, "1.3.0": E13}) == ("1.3.0", E13)


@pytest.mark.parametrize("text", ["{not json", "[1, 2]", "\xff\xfe"])
def test_load_manifest_corrupt_is_empty(tmp_path, caplog, text):
    p = tmp_path / "images.json"
    p.write_text(text)
    with caplog.at_level(logging.ERROR):
        assert rm.load_manifest(p) == {}
    assert any(r.levelno == logging.ERROR and "images.json" in r.message for r in caplog.records)


@pytest.mark.parametrize("version", ["0.0.0-dev", "garbage"])
def test_settings_resolve_non_semver_version(manifest, monkeypatch, version):
    monkeypatch.setattr("api.settings.get_version", lambda: version)
    s = Settings()
    assert s.resolved_nnunet_image == "swestfechtel/nnunet_torsion:latest"
    assert s.resolved_torsion_image == "swestfechtel/torsion:latest"


# --- MORPH_API_RELEASE_MANIFEST -------------------------------------------------------------------


def test_release_manifest_setting_default_is_none(monkeypatch):
    monkeypatch.delenv("MORPH_API_RELEASE_MANIFEST", raising=False)
    s = Settings()
    assert s.release_manifest is None
    assert s.manifest_file == rm.MANIFEST_PATH


def test_release_manifest_setting_from_env(runtime, manifest, monkeypatch, tmp_path):
    """The env setting wins over MANIFEST_PATH, for load_manifest() and for Settings' image defaults."""
    mounted = tmp_path / "mounted.json"
    mounted.write_text(json.dumps({"1.2.0": E12}))
    monkeypatch.setenv("MORPH_API_RELEASE_MANIFEST", str(mounted))
    runtime.reset()  # clear the cached settings
    assert rm.manifest_path() == mounted
    assert rm.load_manifest() == {"1.2.0": E12}
    s = Settings()
    assert s.release_manifest == mounted
    assert s.resolved_torsion_image == E12["torsion"]  # get_version() is 1.4.0 -> newest lower 1.2.0


def test_release_manifest_unset_uses_manifest_path(runtime, manifest, monkeypatch):
    monkeypatch.delenv("MORPH_API_RELEASE_MANIFEST", raising=False)
    runtime.reset()
    assert rm.manifest_path() == manifest
    assert rm.load_manifest() == {"1.3.0": E13}
