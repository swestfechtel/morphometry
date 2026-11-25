"""Tests for per-stage provenance records and the examinations.provenance column."""
import subprocess

import pytest

from api.tasks import provenance

REF = "swestfechtel/torsion:1.0.0"
DIGEST = "swestfechtel/torsion@sha256:aa"


def _patch(monkeypatch, inspected, released):
    monkeypatch.setattr(provenance, "inspect_image", lambda ref: inspected)
    monkeypatch.setattr(provenance, "released_digests", lambda: released)


def test_stage_record_released(monkeypatch):
    _patch(monkeypatch, (DIGEST, "sha256:bb"), {DIGEST})
    rec = provenance.stage_record(REF)
    assert rec["released"] is True and rec["digest"] == DIGEST and rec["image"] == REF
    assert set(rec) == {"api_version", "image", "digest", "released", "api_image",
                        "git_commit", "dirty", "ran_at"}


def test_stage_record_local_build(monkeypatch):
    _patch(monkeypatch, (None, "sha256:bb"), {DIGEST})
    rec = provenance.stage_record(REF)
    assert rec["digest"] == "sha256:bb" and rec["released"] is False


def test_stage_record_inspect_fails(monkeypatch):
    _patch(monkeypatch, (None, None), {DIGEST})
    rec = provenance.stage_record(REF)
    assert rec["digest"] is None and rec["released"] is False


def test_stage_record_never_raises(monkeypatch):
    def boom(ref):
        raise RuntimeError("x")
    monkeypatch.setattr(provenance, "inspect_image", boom)
    rec = provenance.stage_record(REF)
    assert rec["digest"] is None and rec["released"] is False and rec["image"] == REF


def test_stage_record_api_image_env(monkeypatch):
    _patch(monkeypatch, (None, None), set())
    monkeypatch.setenv("MORPH_API_SELF_IMAGE", "swestfechtel/morphometry_api@sha256:cc")
    assert provenance.stage_record(REF)["api_image"] == "swestfechtel/morphometry_api@sha256:cc"


def _fake_inspect(monkeypatch, stdout, returncode=0):
    monkeypatch.setattr(subprocess, "run", lambda cmd, **kw: subprocess.CompletedProcess(
        cmd, returncode, stdout=stdout, stderr=""))


@pytest.mark.parametrize("ref", [
    "swestfechtel/torsion:1.0.0", "swestfechtel/torsion@sha256:aa", "swestfechtel/torsion"])
def test_inspect_image_matches_repository(monkeypatch, ref):
    _fake_inspect(monkeypatch,
                  '["other/x@sha256:zz","swestfechtel/torsion@sha256:aa"] sha256:bb\n')
    assert provenance.inspect_image(ref) == ("swestfechtel/torsion@sha256:aa", "sha256:bb")


def test_inspect_image_registry_host(monkeypatch):
    _fake_inspect(monkeypatch, '["registry.example:5000/a/t@sha256:aa"] sha256:bb')
    assert provenance.inspect_image("registry.example:5000/a/t:1") == (
        "registry.example:5000/a/t@sha256:aa", "sha256:bb")


def test_inspect_image_no_repo_digest(monkeypatch):
    _fake_inspect(monkeypatch, "[] sha256:bb")
    assert provenance.inspect_image(REF) == (None, "sha256:bb")


def test_inspect_image_failure(monkeypatch):
    _fake_inspect(monkeypatch, "", returncode=1)
    assert provenance.inspect_image(REF) == (None, None)

    def missing(cmd, **kw):
        raise FileNotFoundError("docker")
    monkeypatch.setattr(subprocess, "run", missing)
    assert provenance.inspect_image(REF) == (None, None)


def test_stage_record_version_failure_keeps_digest(monkeypatch):
    _patch(monkeypatch, (DIGEST, "sha256:bb"), {DIGEST})

    def boom():
        raise RuntimeError("no VERSION")
    monkeypatch.setattr(provenance, "get_version", boom)
    rec = provenance.stage_record(REF)
    assert rec["digest"] == DIGEST and rec["released"] is True and rec["api_version"] is None
