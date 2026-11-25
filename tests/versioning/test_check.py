"""Tests for the version/changelog/golden/pin rules (``versioning.check`` and ``scripts/check_version.py``)."""
import importlib.util
from pathlib import Path

import pytest
from tests.versioning.conftest import run_git
from versioning.check import GOLDEN_ALLOWLIST, check
from versioning.semver import last_release_tag

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "check_version.py"
_spec = importlib.util.spec_from_file_location("check_version", _SCRIPT)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
main = _module.main

CHANGELOG_100 = "# Changelog\n\n## [1.0.0] - 2026-11-25\n\n- frontend: first.\n"
GOLDEN = GOLDEN_ALLOWLIST[0]

BASE = {
    "VERSION": "1.0.0\n",
    "CHANGELOG.md": CHANGELOG_100,
    "README.md": "readme\n",
    "requirements.txt": "numpy==1.26.4\nscikit_learn==1.5.2\n",
    "morphometry/requirements.txt": "numpy==1.26.4\nscikit-learn==1.5.2\n",
    "api/requirements.txt": "NumPy==1.26.4\n",
    "frontend/app/x.ts": "export const x = 1;\n",
    GOLDEN: "[1.0]\n",
}


def _write(repo: Path, rel: str, text: str) -> None:
    """Write ``text`` to ``repo/rel``, creating parent directories.

    :param repo: repository root.
    :param rel: repo-relative path.
    :param text: new file content.
    """
    path = repo / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _commit(repo: Path, **files: str) -> None:
    """Write files and commit them.

    :param repo: repository root.
    :param files: mapping of repo-relative path to content; path separators are given as ``|``.
    """
    for rel, text in files.items():
        _write(repo, rel.replace("|", "/"), text)
    run_git(repo, "add", "-A")
    run_git(repo, "commit", "-q", "-m", "change")


@pytest.fixture
def released(git_repo: Path) -> Path:
    """A repository whose base commit is tagged ``v1.0.0`` and holds a compliant layout.

    :param git_repo: the throwaway repository from ``conftest.py``.
    :return: path of the repository root.
    """
    for rel, text in BASE.items():
        _write(git_repo, rel, text)
    run_git(git_repo, "add", "-A")
    run_git(git_repo, "commit", "-q", "-m", "release layout")
    run_git(git_repo, "tag", "v1.0.0")
    return git_repo


def test_no_gated_change_passes(released):
    _commit(released, **{"README.md": "edited\n"})
    assert check(released) == []


def test_gated_change_without_bump_fails(released):
    _commit(released, **{"frontend|app|x.ts": "export const x = 2;\n"})
    problems = check(released)
    assert len(problems) == 1
    assert "must be greater" in problems[0] and "v1.0.0" in problems[0]
    assert "frontend" in problems[0] and "frontend/app/x.ts" in problems[0]


def test_gated_change_with_bump_and_changelog_passes(released):
    _commit(released, **{
        "frontend|app|x.ts": "export const x = 2;\n",
        "VERSION": "1.0.1\n",
        "CHANGELOG.md": "## [1.0.1] - 2026-11-26\n\n- frontend: x.\n\n" + CHANGELOG_100,
    })
    assert check(released) == []


def test_bump_without_changelog_fails(released):
    _commit(released, **{"frontend|app|x.ts": "export const x = 2;\n", "VERSION": "1.0.1\n"})
    problems = check(released)
    assert len(problems) == 1 and "CHANGELOG.md" in problems[0] and "1.0.1" in problems[0]


def test_golden_change_needs_minor(released):
    entry = "## [1.0.1] - 2026-11-26\n\n- torsion: x.\n\n" + CHANGELOG_100
    _commit(released, **{
        GOLDEN.replace("/", "|"): "[2.0]\n",
        "frontend|app|x.ts": "export const x = 2;\n",
        "VERSION": "1.0.1\n",
        "CHANGELOG.md": entry,
    })
    problems = check(released)
    assert len(problems) == 1
    assert f"golden {GOLDEN} changed since v1.0.0: bump at least MINOR" in problems[0]
    _commit(released, **{
        "VERSION": "1.1.0\n",
        "CHANGELOG.md": entry.replace("1.0.1", "1.1.0"),
    })
    assert check(released) == []


def test_golden_change_alone_needs_minor(released):
    """A golden change with no gated-path change still demands a MINOR bump."""
    _commit(released, **{GOLDEN.replace("/", "|"): "[2.0]\n"})
    problems = check(released)
    assert any("bump at least MINOR" in p for p in problems)


def test_pin_mismatch(released):
    _commit(released, **{"morphometry|requirements.txt": "numpy==1.26.3\nscikit-learn==1.5.2\n"})
    problems = check(released)
    assert any("numpy" in p and "1.26.3" in p and "1.26.4" in p for p in problems)


def test_pin_name_normalisation_and_missing_api_file(released):
    """``scikit_learn`` vs ``scikit-learn`` and ``NumPy`` vs ``numpy`` are equal; absent api file is fine."""
    assert check(released) == []
    run_git(released, "rm", "-q", "api/requirements.txt")
    run_git(released, "commit", "-q", "-m", "rm")
    assert [p for p in check(released) if " pins " in p] == []


def test_initial_release(git_repo):
    for rel, text in BASE.items():
        _write(git_repo, rel, text)
    run_git(git_repo, "add", "-A")
    run_git(git_repo, "commit", "-q", "-m", "layout")
    assert check(git_repo) == []
    _write(git_repo, "CHANGELOG.md", "# Changelog\n")
    run_git(git_repo, "commit", "-q", "-am", "drop changelog")
    problems = check(git_repo)
    assert len(problems) == 1 and "CHANGELOG.md" in problems[0]


@pytest.mark.parametrize("text", ["1.2", "v1.1.0", "1.1.0-rc1"])
def test_malformed_version(released, text):
    _commit(released, VERSION=text + "\n")
    problems = check(released)
    assert problems and "VERSION" in problems[0]


def test_shallow_unreachable_tags(git_repo):
    main_branch = run_git(git_repo, "symbolic-ref", "--short", "HEAD")
    run_git(git_repo, "checkout", "-q", "--orphan", "other")
    _write(git_repo, "VERSION", "1.0.0\n")
    run_git(git_repo, "add", "-A")
    run_git(git_repo, "commit", "-q", "-m", "orphan")
    run_git(git_repo, "tag", "v1.0.0")
    run_git(git_repo, "checkout", "-q", "-f", main_branch)
    _write(git_repo, "VERSION", "1.0.0\n")
    _write(git_repo, "CHANGELOG.md", CHANGELOG_100)
    run_git(git_repo, "add", "-A")
    run_git(git_repo, "commit", "-q", "-m", "v")
    problems = check(git_repo)
    assert any("shallow clone" in p for p in problems)


def test_cli_exit_codes(released, capsys):
    assert main(["--repo", str(released)]) == 0
    assert "version check ok (1.0.0)" in capsys.readouterr().out
    _commit(released, **{"frontend|app|x.ts": "export const x = 2;\n"})
    assert main(["--repo", str(released)]) == 1
    assert "must be greater" in capsys.readouterr().out


def test_tag_reachable_from_head_ref_not_head(released):
    """``head_ref`` selects the tag search: a tag only reachable from it is found (and used)."""
    run_git(released, "checkout", "-q", "-b", "feature")
    _commit(released, **{"frontend|app|x.ts": "export const x = 2;\n", "VERSION": "1.0.1\n",
                         "CHANGELOG.md": "## [1.0.1] - 2026-11-26\n\n- frontend: x.\n\n" + CHANGELOG_100})
    run_git(released, "tag", "v1.0.1")
    run_git(released, "checkout", "-q", "-")
    # HEAD (the released commit) reaches only v1.0.0; the feature tip reaches v1.0.1.
    assert last_release_tag(released) == "v1.0.0"
    assert last_release_tag(released, "feature") == "v1.0.1"
    assert check(released, "feature") == []


def test_pin_missing_from_root(released):
    _commit(released, **{"api|requirements.txt": "NumPy==1.26.4\nfastapi==0.1.0\n"})
    problems = check(released)
    assert any("api/requirements.txt" in p and "fastapi" in p and "not pinned" in p for p in problems)


def test_every_torsion_golden_is_allow_listed():
    """A new torsion/knee-rotation golden must not escape the MINOR-bump rule."""
    golden_dir = Path(__file__).resolve().parents[1] / "golden"
    found = {f"tests/golden/{p.name}" for pattern in ("*torsion*.json", "knee_rotation*.json")
             for p in golden_dir.glob(pattern)}
    assert found and found <= set(GOLDEN_ALLOWLIST)
    assert all((golden_dir.parent.parent / rel).is_file() for rel in GOLDEN_ALLOWLIST)
