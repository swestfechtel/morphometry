"""Tests for versioning.semver."""
import pytest

from tests.versioning.conftest import run_git
from versioning.semver import Version, last_release_tag, parse_version


def test_parse_version_ok():
    assert parse_version("1.10.0") > parse_version("1.9.3")
    assert parse_version("1.2.3") == Version(1, 2, 3)
    assert str(parse_version("1.2.3")) == "1.2.3"


@pytest.mark.parametrize("text", ["1.2", "v1.2.0", "1.2.0-rc1", ""])
def test_parse_version_rejects(text):
    with pytest.raises(ValueError, match="invalid version"):
        parse_version(text)


def test_last_release_tag(git_repo):
    assert last_release_tag(git_repo) is None
    run_git(git_repo, "tag", "v1.0.0")
    assert last_release_tag(git_repo) == "v1.0.0"
