"""Single source of the release version and build provenance.

The version lives in the repo-root ``VERSION`` file. Docker images do not ship
``.git``, so builds bake ``MORPH_VERSION`` / ``MORPH_GIT_COMMIT`` into the
environment, which win over the file and git.
"""
import os
import subprocess
from pathlib import Path

# The repo root is the parent of the ``api/`` directory.
_REPO_ROOT = Path(__file__).resolve().parent.parent


def get_version() -> str:
    """Return the release version.

    :return: ``MORPH_VERSION`` if set, else the stripped content of the repo-root ``VERSION``.
    """
    env = os.environ.get("MORPH_VERSION")
    if env:
        return env
    return (_REPO_ROOT / "VERSION").read_text().strip()


def _git(*args: str) -> str | None:
    """Run git in the repo root; return stripped stdout or ``None`` on any failure."""
    try:
        done = subprocess.run(
            ["git", *args], cwd=_REPO_ROOT, capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return done.stdout.strip()


def get_git_commit() -> str | None:
    """Return the full git commit sha of the running code.

    :return: ``MORPH_GIT_COMMIT`` if set, else ``git rev-parse HEAD``, else ``None``.
    """
    env = os.environ.get("MORPH_GIT_COMMIT")
    if env:
        return env
    return _git("rev-parse", "HEAD") or None


def is_dirty() -> bool | None:
    """Report whether the working tree has uncommitted changes.

    :return: ``False`` when ``MORPH_GIT_COMMIT`` is set (image builds are clean by
        construction); otherwise whether ``git status --porcelain`` is non-empty;
        ``None`` if git is unavailable.
    """
    if os.environ.get("MORPH_GIT_COMMIT"):
        return False
    status = _git("status", "--porcelain")
    return None if status is None else bool(status)
