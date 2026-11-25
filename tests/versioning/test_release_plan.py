"""Tests for the release planner (``versioning.release_plan``); pure, no docker."""
from pathlib import Path

import pytest

from tests.versioning.conftest import run_git
from tests.versioning.test_images import MINI_FILES
from versioning.release_plan import UNLABELLED, ReleaseRefused, RemoteState, plan_release

CHANGELOG_100 = "# Changelog\n\n## [1.0.0] - 2026-11-25\n\n- torsion: first.\n"
CHANGELOG_101 = "# Changelog\n\n## [1.0.1] - 2026-11-26\n\n- torsion: shared.\n\n" + CHANGELOG_100[12:]

RELEASE_FILES = {
    **MINI_FILES,
    "VERSION": "1.0.0\n",
    "CHANGELOG.md": CHANGELOG_100,
    "requirements.txt": "numpy==1.26.4\n",
}


def _digest(char: str) -> str:
    """A syntactically valid sha256 digest made of one repeated hex character.

    :param char: one hex character.
    :return: ``"sha256:" + 64 * char``.
    """
    return "sha256:" + char * 64


#: The manifest entry of a released 1.0.0.
MANIFEST_100 = {
    "1.0.0": {
        "nnunet": "swestfechtel/nnunet_torsion@" + _digest("a"),
        "torsion": "swestfechtel/torsion@" + _digest("b"),
        "api": "swestfechtel/morphometry_api@" + _digest("c"),
        "frontend": "swestfechtel/morphometry_frontend@" + _digest("d"),
        "git_commit": "0" * 40,
        "date": "2026-11-25",
    }
}


def _write_all(repo: Path, files: dict[str, str]) -> None:
    """Write files into ``repo`` and commit them.

    :param repo: repository root.
    :param files: repo-relative path → content.
    """
    for rel, text in files.items():
        path = repo / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    run_git(repo, "add", "-A")
    run_git(repo, "commit", "-q", "-m", "change")


def _head(repo: Path) -> str:
    """Full sha of ``HEAD``.

    :param repo: repository root.
    :return: the 40-character commit sha.
    """
    return run_git(repo, "rev-parse", "HEAD")


def _remote(**overrides) -> RemoteState:
    """A healthy :class:`RemoteState` (nothing on Hub, no git tag, clean main, up to date).

    :param overrides: fields to replace.
    :return: the state.
    """
    state = dict(existing_tags={k: None for k in ("nnunet", "torsion", "api", "frontend")},
                 tag_exists_local=False, tag_exists_remote=False, clean=True, on_main=True, up_to_date=True)
    state.update(overrides)
    return RemoteState(**state)


@pytest.fixture
def initial(git_repo: Path) -> Path:
    """A repository ready for its first release (``VERSION`` 1.0.0, no tag).

    :param git_repo: the throwaway repository from ``conftest.py``.
    :return: path of the repository root.
    """
    _write_all(git_repo, RELEASE_FILES)
    return git_repo


@pytest.fixture
def second(initial: Path) -> Path:
    """``initial`` tagged ``v1.0.0``, then a change to a module both torsion and api import, as 1.0.1.

    :param initial: the initial-release repository.
    :return: path of the repository root.
    """
    run_git(initial, "tag", "-a", "v1.0.0", "-m", "1.0.0")
    _write_all(initial, {"morphometry/shared.py": "X = 2\n", "VERSION": "1.0.1\n", "CHANGELOG.md": CHANGELOG_101})
    return initial


def _actions(plan) -> dict[str, str]:
    """Image key → action of a plan.

    :param plan: list of :class:`ImageAction`.
    :return: mapping in plan order.
    """
    return {a.key: a.action for a in plan}


def test_plan_initial_release_builds_all(initial):
    plan = plan_release(initial, "1.0.0", _head(initial), _remote(), {})
    assert [a.key for a in plan] == ["nnunet", "torsion", "api", "frontend"]
    assert all(a.action == "build" and a.from_ref is None for a in plan)


def test_plan_retags_unchanged_nnunet(second):
    plan = plan_release(second, "1.0.1", _head(second), _remote(), MANIFEST_100)
    assert _actions(plan) == {"nnunet": "retag", "torsion": "build", "api": "build", "frontend": "retag"}
    by_key = {a.key: a for a in plan}
    assert by_key["nnunet"].from_ref == MANIFEST_100["1.0.0"]["nnunet"]
    assert by_key["frontend"].from_ref == MANIFEST_100["1.0.0"]["frontend"]
    assert "morphometry/shared.py" in by_key["torsion"].reason


def test_plan_retag_without_manifest_entry_refused(second):
    manifest = {"1.0.0": {k: v for k, v in MANIFEST_100["1.0.0"].items() if k != "nnunet"}}
    with pytest.raises(ReleaseRefused) as info:
        plan_release(second, "1.0.1", _head(second), _remote(), manifest)
    assert any("nnunet" in m for m in info.value.messages)


def test_plan_retag_with_foreign_repo_in_manifest_refused(second):
    manifest = {"1.0.0": {**MANIFEST_100["1.0.0"], "frontend": "someone/else@" + _digest("d")}}
    with pytest.raises(ReleaseRefused) as info:
        plan_release(second, "1.0.1", _head(second), _remote(), manifest)
    assert any("frontend" in m for m in info.value.messages)


def test_plan_refuses_existing_foreign_tag(initial):
    remote = _remote(existing_tags={"nnunet": None, "torsion": "deadbeef", "api": None, "frontend": None})
    with pytest.raises(ReleaseRefused) as info:
        plan_release(initial, "1.0.0", _head(initial), remote, {})
    assert any("torsion" in m for m in info.value.messages)
    assert not any("nnunet" in m for m in info.value.messages)


def test_plan_resumes_own_tag(initial):
    head = _head(initial)
    remote = _remote(existing_tags={"nnunet": head, "torsion": None, "api": None, "frontend": None})
    plan = plan_release(initial, "1.0.0", head, remote, {})
    assert _actions(plan) == {"nnunet": "skip", "torsion": "build", "api": "build", "frontend": "build"}


def test_plan_resumes_own_retag(second):
    """A ``:<version>`` already pointing at the retag source (an interrupted earlier run) is skipped."""
    remote = _remote(existing_tags={"nnunet": "0" * 40, "torsion": None, "api": None, "frontend": None},
                     existing_digests={"nnunet": _digest("a")})
    plan = plan_release(second, "1.0.1", _head(second), remote, MANIFEST_100)
    assert _actions(plan)["nnunet"] == "skip"
    remote = _remote(existing_tags={"nnunet": "0" * 40, "torsion": None, "api": None, "frontend": None},
                     existing_digests={"nnunet": _digest("e")})
    with pytest.raises(ReleaseRefused):
        plan_release(second, "1.0.1", _head(second), remote, MANIFEST_100)


@pytest.mark.parametrize("field", ["clean", "on_main", "up_to_date"])
def test_plan_refuses_dirty(initial, field):
    with pytest.raises(ReleaseRefused) as info:
        plan_release(initial, "1.0.0", _head(initial), _remote(**{field: False}), {})
    assert len(info.value.messages) == 1


@pytest.mark.parametrize("field", ["tag_exists_local", "tag_exists_remote"])
def test_plan_refuses_existing_git_tag(initial, field):
    with pytest.raises(ReleaseRefused) as info:
        plan_release(initial, "1.0.0", _head(initial), _remote(**{field: True}), {})
    assert any("v1.0.0" in m for m in info.value.messages)


def test_plan_refuses_check_violation(second):
    """A gated change without a VERSION bump is a ``check()`` violation."""
    _write_all(second, {"VERSION": "1.0.0\n", "morphometry/shared.py": "X = 3\n"})
    with pytest.raises(ReleaseRefused) as info:
        plan_release(second, "1.0.0", _head(second), _remote(), MANIFEST_100)
    assert any("must be greater" in m for m in info.value.messages)


def test_plan_refuses_version_mismatch(initial):
    """The version planned must be the committed ``VERSION``."""
    with pytest.raises(ReleaseRefused) as info:
        plan_release(initial, "1.0.1", _head(initial), _remote(), {})
    assert any("VERSION" in m for m in info.value.messages)


def test_plan_adopts_unlabelled_on_initial_release(initial, second_repo_factory):
    head = _head(initial)
    remote = _remote(existing_tags={"nnunet": UNLABELLED, "torsion": None, "api": None, "frontend": None})
    plan = plan_release(initial, "1.0.0", head, remote, {}, adopt=frozenset({"nnunet"}))
    assert _actions(plan)["nnunet"] == "skip"
    with pytest.raises(ReleaseRefused) as info:
        plan_release(initial, "1.0.0", head, remote, {})
    assert any("nnunet" in m for m in info.value.messages)

    repo = second_repo_factory()
    with pytest.raises(ReleaseRefused) as info:
        plan_release(repo, "1.0.1", _head(repo), remote, MANIFEST_100, adopt=frozenset({"nnunet"}))
    assert any("nnunet" in m for m in info.value.messages)


@pytest.fixture
def second_repo_factory(tmp_path: Path):
    """Build a separate repository in the state of the ``second`` fixture.

    :param tmp_path: pytest's per-test temporary directory.
    :return: a zero-argument callable returning the new repository's root.
    """
    def make() -> Path:
        repo = tmp_path / "second"
        repo.mkdir()
        for args in (("init", "-q"), ("config", "user.name", "T"), ("config", "user.email", "t@example.com"),
                     ("config", "commit.gpgsign", "false"), ("config", "tag.gpgsign", "false")):
            run_git(repo, *args)
        _write_all(repo, RELEASE_FILES)
        run_git(repo, "tag", "-a", "v1.0.0", "-m", "1.0.0")
        _write_all(repo, {"morphometry/shared.py": "X = 2\n", "VERSION": "1.0.1\n", "CHANGELOG.md": CHANGELOG_101})
        return repo
    return make
