"""Shared fixtures for the versioning tests."""
import subprocess
from pathlib import Path

import pytest


def run_git(repo: Path, *args: str) -> str:
    """Run a git command in ``repo`` and return its stripped stdout.

    :param repo: working directory of the repository.
    :param args: git sub-command and arguments.
    :return: stripped standard output.
    """
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    """A throwaway git repository holding exactly one commit.

    ``user.name`` / ``user.email`` are set locally so commits and tags work
    regardless of the host's git configuration.

    :param tmp_path: pytest's per-test temporary directory.
    :return: path of the repository root.
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    run_git(repo, "init", "-q")
    run_git(repo, "config", "user.name", "Test")
    run_git(repo, "config", "user.email", "test@example.com")
    run_git(repo, "config", "commit.gpgsign", "false")
    run_git(repo, "config", "tag.gpgsign", "false")
    (repo / "a.txt").write_text("a\n")
    run_git(repo, "add", "a.txt")
    run_git(repo, "commit", "-q", "-m", "init")
    return repo
