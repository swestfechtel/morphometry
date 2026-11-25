"""Shared git helpers (standard library only).

Every git call in ``versioning`` goes through :func:`git`, whose default environment has the
variables that redirect git to another repository removed: git exports an absolute ``GIT_DIR``
to hooks, which would otherwise make a call in ``repo`` answer for the hook's repository.
"""
import os
import subprocess
from pathlib import Path


def env_without_git_overrides() -> dict[str, str]:
    """The current environment minus variables that redirect git to another repository.

    :return: a copy of ``os.environ`` without ``GIT_DIR``, ``GIT_WORK_TREE``,
        ``GIT_INDEX_FILE``, ``GIT_COMMON_DIR`` and ``GIT_PREFIX``.
    """
    overrides = {"GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_COMMON_DIR", "GIT_PREFIX"}
    return {k: v for k, v in os.environ.items() if k not in overrides}


def git(repo: Path, *args: str, env: dict[str, str] | None = None, check: bool = True) -> subprocess.CompletedProcess:
    """Run a git command in ``repo`` and return the completed process.

    :param repo: working directory of the repository.
    :param args: git sub-command and arguments.
    :param env: full environment for the subprocess; default :func:`env_without_git_overrides`.
    :param check: raise on a non-zero exit status.
    :return: the :class:`subprocess.CompletedProcess` (text mode, output captured).
    :raises subprocess.CalledProcessError: if ``check`` and git exits non-zero.
    """
    return subprocess.run(
        ["git", *args], cwd=repo, check=check, capture_output=True, text=True,
        env=env_without_git_overrides() if env is None else env,
    )
