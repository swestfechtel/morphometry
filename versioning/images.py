"""The four docker images, the files that affect each ("gated paths"), and what changed between refs.

Standard library only. A file is *gated* for an image when changing it can change what the
image runs; :func:`changed_images` intersects ``git diff`` with those sets so the version
check and the release script know which images a commit range touches.

Gated sets ("what affects an image", see ``docs/superpowers/specs/2026-11-25-api-versioning-design.md``):

- **nnunet**: tracked files under ``api/docker/nnunet_torsion/`` except ``checkpoints/`` (which
  is untracked anyway); this includes the committed ``checkpoints.sha256`` manifest.
- **torsion**: the ``morphometry`` import closure of ``morphometry/docker/compute_torsion.py``,
  plus that entry point, ``morphometry/Dockerfile`` and ``morphometry/requirements.txt``.
- **api**: tracked files under ``api/`` except ``api/docker/``, ``api/tests/``, ``api/data/``,
  ``api/logs/``, ``*.ipynb`` and ``api/tests.py``; plus the ``morphometry`` import closure of
  every ``.py`` among them; plus ``api/Dockerfile``, ``api/requirements.txt``, ``.dockerignore``.
- **frontend**: tracked files under ``frontend/`` except ``node_modules/``, ``.next/``,
  ``*.tsbuildinfo`` and ``.env*``.

Every gated set is filtered to the files tracked at the ref examined, so a file a later task
has not created yet (e.g. ``api/Dockerfile``) is simply absent.
"""
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from versioning.closure import import_closure

#: Entry point of the torsion image; its import closure is the image's gated code.
TORSION_ENTRY = "morphometry/docker/compute_torsion.py"

#: Placeholder "changed file" reported for every image when there is no previous release.
INITIAL_RELEASE = "<initial release>"


def _git(repo: Path, *args: str, env: dict[str, str] | None = None) -> str:
    """Run a git command in ``repo`` and return its standard output.

    :param repo: working directory of the repository.
    :param args: git sub-command and arguments.
    :param env: full environment for the subprocess, or ``None`` to inherit the current one.
    :return: standard output, unstripped (``-z`` output keeps its NUL separators).
    :raises subprocess.CalledProcessError: if git exits non-zero.
    """
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True, env=env
    ).stdout


def tracked_files(root: Path) -> set[str]:
    """List the files of a source tree, as git tracks them.

    :param root: either the top level of a git work tree — then ``git ls-files`` is used, so
        untracked files (``node_modules/``, checkpoints, build output) are ignored — or a plain
        directory holding an export of one ref (see :func:`export_ref`), whose every file is
        tracked at that ref by construction.
    :return: POSIX-style paths relative to ``root``.
    """
    root = Path(root).resolve()
    top = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"], cwd=root, capture_output=True, text=True
    )
    if top.returncode == 0 and Path(top.stdout.strip()).resolve() == root:
        return {p for p in _git(root, "ls-files", "-z").split("\0") if p}
    files = set()
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            files.add((Path(dirpath) / name).relative_to(root).as_posix())
    return files


def export_ref(repo: Path, ref: str, dest: Path) -> None:
    """Write the tracked files of ``ref`` into ``dest`` without touching the repo's index or work tree.

    Uses a throwaway index (``GIT_INDEX_FILE``): ``read-tree`` loads ``ref`` into it and
    ``checkout-index`` writes exactly those files under ``dest`` (unaffected by
    ``export-ignore`` attributes, unlike ``git archive``).

    :param repo: path of the git repository.
    :param ref: any commit-ish (sha, tag, ``HEAD``).
    :param dest: directory to create and fill; must not already hold files.
    :raises subprocess.CalledProcessError: if ``ref`` does not resolve.
    """
    dest = Path(dest).resolve()
    dest.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        env = {**os.environ, "GIT_INDEX_FILE": str(Path(tmp) / "index")}
        _git(repo, "read-tree", f"{ref}^{{tree}}", env=env)
        _git(repo, "checkout-index", "--all", f"--prefix={dest}/", env=env)


def _paths(paths: set[Path]) -> set[str]:
    """Convert repo-relative ``Path`` objects to POSIX strings.

    :param paths: repo-relative paths.
    :return: the same paths as POSIX-style strings.
    """
    return {p.as_posix() for p in paths}


def _nnunet_gated(root: Path) -> set[str]:
    """Gated files of the nnUNet image.

    :param root: source tree (work tree or export of one ref).
    :return: tracked files under ``api/docker/nnunet_torsion/`` except ``checkpoints/``.
    """
    prefix = "api/docker/nnunet_torsion/"
    return {
        f for f in tracked_files(root)
        if f.startswith(prefix) and not f.startswith(prefix + "checkpoints/")
    }


def _torsion_gated(root: Path) -> set[str]:
    """Gated files of the torsion image.

    :param root: source tree (work tree or export of one ref).
    :return: the import closure of :data:`TORSION_ENTRY` plus the entry point, Dockerfile and
        pinned requirements, filtered to tracked files.
    :raises versioning.closure.UnresolvedImport: if the closure walker cannot resolve an import.
    """
    tracked = tracked_files(root)
    gated = {TORSION_ENTRY, "morphometry/Dockerfile", "morphometry/requirements.txt"}
    if TORSION_ENTRY in tracked:
        gated |= _paths(import_closure([Path(root) / TORSION_ENTRY], Path(root)))
    return gated & tracked


_API_EXCLUDED_PREFIXES = ("api/docker/", "api/tests/", "api/data/", "api/logs/")


def _api_gated(root: Path) -> set[str]:
    """Gated files of the api image.

    :param root: source tree (work tree or export of one ref).
    :return: the api's own sources (see module docstring for the exclusions), the
        ``morphometry`` import closure of its ``.py`` files, and ``api/Dockerfile``,
        ``api/requirements.txt``, ``.dockerignore``; filtered to tracked files.
    :raises versioning.closure.UnresolvedImport: if the closure walker cannot resolve an import.
    """
    tracked = tracked_files(root)
    own = {
        f for f in tracked
        if f.startswith("api/")
        and not f.startswith(_API_EXCLUDED_PREFIXES)
        and not f.endswith(".ipynb")
        and f != "api/tests.py"
    }
    entries = [Path(root) / f for f in sorted(own) if f.endswith(".py")]
    closure = _paths(import_closure(entries, Path(root))) if entries else set()
    return (own | closure | {"api/Dockerfile", "api/requirements.txt", ".dockerignore"}) & tracked


def _frontend_gated(root: Path) -> set[str]:
    """Gated files of the frontend image.

    :param root: source tree (work tree or export of one ref).
    :return: tracked files under ``frontend/`` except ``node_modules/``, ``.next/``,
        ``*.tsbuildinfo`` and ``.env*``.
    """
    gated = set()
    for f in tracked_files(root):
        parts = f.split("/")
        if parts[0] != "frontend":
            continue
        if "node_modules" in parts or ".next" in parts:
            continue
        if parts[-1].endswith(".tsbuildinfo") or parts[-1].startswith(".env"):
            continue
        gated.add(f)
    return gated


@dataclass(frozen=True)
class ImageSpec:
    """One released docker image.

    :param key: short image key used everywhere (``nnunet``, ``torsion``, ``api``, ``frontend``).
    :param repo: Docker Hub repository, e.g. ``swestfechtel/torsion``.
    :param context: repo-relative docker build context directory (``.`` = repo root).
    :param dockerfile: repo-relative path of the Dockerfile.
    :param gated: function mapping a source-tree root to the repo-relative files that affect
        this image (see the module docstring).
    """

    key: str
    repo: str
    context: str
    dockerfile: str
    gated: Callable[[Path], set[str]]


#: The four images in release (build) order: nnunet and torsion first, because the api image
#: is built with their references for this release.
IMAGES: dict[str, ImageSpec] = {
    spec.key: spec
    for spec in (
        ImageSpec("nnunet", "swestfechtel/nnunet_torsion", "api/docker/nnunet_torsion",
                  "api/docker/nnunet_torsion/Dockerfile", _nnunet_gated),
        ImageSpec("torsion", "swestfechtel/torsion", "morphometry",
                  "morphometry/Dockerfile", _torsion_gated),
        ImageSpec("api", "swestfechtel/morphometry_api", ".", "api/Dockerfile", _api_gated),
        ImageSpec("frontend", "swestfechtel/morphometry_frontend", "frontend",
                  "frontend/Dockerfile", _frontend_gated),
    )
}


def changed_images(repo: Path, base_ref: str | None, head_ref: str = "HEAD") -> dict[str, list[str]]:
    """Report which images a commit range changes, and through which files.

    Gated sets are computed from the sources *at* ``head_ref`` and *at* ``base_ref`` (each
    exported to a temporary directory, never read from the work tree) and unioned, so deleting
    or un-importing a closure module still counts as changing the image.

    :param repo: path of the git repository.
    :param base_ref: the last release (tag or sha), or ``None`` if there has been none.
    :param head_ref: the commit being checked, default ``HEAD``.
    :return: image key → sorted list of changed gated files, containing only images with at
        least one change (in :data:`IMAGES` order). With ``base_ref=None`` every image maps to
        ``["<initial release>"]``.
    :raises versioning.closure.UnresolvedImport: if a closure cannot be computed at either ref.
    :raises subprocess.CalledProcessError: if a ref does not resolve.
    """
    if base_ref is None:
        return {key: [INITIAL_RELEASE] for key in IMAGES}
    diff = _git(repo, "diff", "--name-only", "--no-renames", "-z", base_ref, head_ref)
    changed = {p for p in diff.split("\0") if p}
    if not changed:
        return {}
    gated: dict[str, set[str]] = {key: set() for key in IMAGES}
    with tempfile.TemporaryDirectory() as tmp:
        for index, ref in enumerate((head_ref, base_ref)):
            tree = Path(tmp) / str(index)
            export_ref(repo, ref, tree)
            for key, spec in IMAGES.items():
                gated[key] |= spec.gated(tree)
    return {key: sorted(changed & files) for key, files in gated.items() if changed & files}
