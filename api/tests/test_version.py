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
