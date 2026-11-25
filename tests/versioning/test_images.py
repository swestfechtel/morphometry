"""Tests for the per-image gated paths and ``changed_images`` (``versioning.images``)."""
from pathlib import Path

import pytest

from tests.versioning.conftest import run_git
from versioning.images import IMAGES, changed_images

MINI_FILES = {
    # A miniature morphometry package laid out at the real entry-point path.
    "morphometry/__init__.py": "",
    "morphometry/Dockerfile": "FROM python\n",
    "morphometry/requirements.txt": "numpy==1.26.4\n",
    "morphometry/docker/compute_torsion.py": "from morphometry.femur import f\n",
    "morphometry/femur.py": "from morphometry import shared\n\ndef f():\n    return 1\n",
    "morphometry/shared.py": "X = 1\n",
    "morphometry/unused.py": "Y = 2\n",
    # The API imports the shared module, so it gates both torsion and api.
    "api/__init__.py": "",
    "api/main.py": "from morphometry.shared import X\nfrom .helpers import h\n",
    "api/helpers.py": "def h():\n    return 0\n",
    "api/tests.py": "# manual smoke tests\n",
    "api/tests/test_x.py": "def test(): pass\n",
    "api/notebook.ipynb": "{}\n",
    "api/docker/nnunet_torsion/Dockerfile": "FROM nnunet\n",
    "api/docker/nnunet_torsion/checkpoints.sha256": "abc  hip/fold_0.pth\n",
    "frontend/package.json": "{}\n",
    "frontend/app/x.ts": "export const x = 1;\n",
    "frontend/.env.example": "NEXT_PUBLIC_MODEL_API=x\n",
    "docs/readme.md": "docs\n",
}


@pytest.fixture
def mini_repo(git_repo: Path) -> Path:
    """The ``git_repo`` fixture with a miniature morphometry/api/frontend layout committed on top.

    :param git_repo: the throwaway repository from ``conftest.py``.
    :return: path of the repository root.
    """
    for rel, text in MINI_FILES.items():
        path = git_repo / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    run_git(git_repo, "add", "-A")
    run_git(git_repo, "commit", "-q", "-m", "mini layout")
    return git_repo


def _commit_edit(repo: Path, rel: str, text: str = "# edited\n") -> None:
    """Append ``text`` to ``repo/rel`` (creating it if needed) and commit it.

    :param repo: repository root.
    :param rel: repo-relative path of the file to edit.
    :param text: text appended to the file.
    """
    path = repo / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(text)
    run_git(repo, "add", "-A")
    run_git(repo, "commit", "-q", "-m", f"edit {rel}")


def test_images_order_and_repos():
    """The release builds in ``IMAGES`` order: nnunet and torsion before the api that pins them."""
    assert list(IMAGES) == ["nnunet", "torsion", "api", "frontend"]
    assert IMAGES["torsion"].repo == "swestfechtel/torsion"
    assert IMAGES["nnunet"].repo == "swestfechtel/nnunet_torsion"
    assert IMAGES["api"].repo == "swestfechtel/morphometry_api"
    assert IMAGES["frontend"].repo == "swestfechtel/morphometry_frontend"
    assert all(spec.key == key for key, spec in IMAGES.items())


def test_gated_sets(mini_repo):
    """Each image's gated set is exactly the spec's "what affects an image" on the mini layout."""
    assert IMAGES["torsion"].gated(mini_repo) == {
        "morphometry/__init__.py",
        "morphometry/Dockerfile",
        "morphometry/requirements.txt",
        "morphometry/docker/compute_torsion.py",
        "morphometry/femur.py",
        "morphometry/shared.py",
    }
    assert IMAGES["api"].gated(mini_repo) == {
        "api/__init__.py",
        "api/main.py",
        "api/helpers.py",
        "morphometry/__init__.py",
        "morphometry/shared.py",
    }
    assert IMAGES["frontend"].gated(mini_repo) == {"frontend/package.json", "frontend/app/x.ts"}
    assert IMAGES["nnunet"].gated(mini_repo) == {
        "api/docker/nnunet_torsion/Dockerfile",
        "api/docker/nnunet_torsion/checkpoints.sha256",
    }


def test_gated_sets_ignore_untracked_files(mini_repo):
    """Untracked files (e.g. checkpoints, node_modules) never enter a gated set."""
    (mini_repo / "api/docker/nnunet_torsion/checkpoints").mkdir()
    (mini_repo / "api/docker/nnunet_torsion/checkpoints/fold_0.pth").write_text("w")
    (mini_repo / "frontend/new.ts").write_text("x")

    assert "frontend/new.ts" not in IMAGES["frontend"].gated(mini_repo)
    assert IMAGES["nnunet"].gated(mini_repo) == {
        "api/docker/nnunet_torsion/Dockerfile",
        "api/docker/nnunet_torsion/checkpoints.sha256",
    }


def test_changed_images_unused_module(mini_repo):
    """Research code outside every closure changes no image."""
    base = run_git(mini_repo, "rev-parse", "HEAD")
    _commit_edit(mini_repo, "morphometry/unused.py")
    _commit_edit(mini_repo, "docs/readme.md")

    assert changed_images(mini_repo, base) == {}


def test_changed_images_shared_closure_file(mini_repo):
    """A module in both closures changes torsion and api, and nothing else."""
    base = run_git(mini_repo, "rev-parse", "HEAD")
    _commit_edit(mini_repo, "morphometry/shared.py")

    assert changed_images(mini_repo, base) == {
        "torsion": ["morphometry/shared.py"],
        "api": ["morphometry/shared.py"],
    }


def test_changed_images_frontend_only(mini_repo):
    """A frontend source edit changes the frontend image only."""
    base = run_git(mini_repo, "rev-parse", "HEAD")
    _commit_edit(mini_repo, "frontend/app/x.ts")

    assert changed_images(mini_repo, base) == {"frontend": ["frontend/app/x.ts"]}


def test_changed_images_deleted_closure_module_counts(mini_repo):
    """Deleting a module reachable only at ``base_ref`` still counts (gated sets are unioned)."""
    base = run_git(mini_repo, "rev-parse", "HEAD")
    (mini_repo / "morphometry/femur.py").write_text("def f():\n    return 1\n")
    (mini_repo / "morphometry/shared.py").unlink()
    (mini_repo / "api/main.py").write_text("from .helpers import h\n")
    run_git(mini_repo, "add", "-A")
    run_git(mini_repo, "commit", "-q", "-m", "drop shared")

    assert changed_images(mini_repo, base) == {
        "torsion": ["morphometry/femur.py", "morphometry/shared.py"],
        "api": ["api/main.py", "morphometry/shared.py"],
    }


def test_changed_images_reads_sources_at_each_ref(mini_repo):
    """An uncommitted working-tree import must not leak into the gated set at ``head_ref``."""
    base = run_git(mini_repo, "rev-parse", "HEAD")
    _commit_edit(mini_repo, "morphometry/unused.py")
    (mini_repo / "morphometry/femur.py").write_text("from morphometry import unused\n")

    assert changed_images(mini_repo, base) == {}


def test_changed_images_initial_release(mini_repo):
    """No previous release tag: every image is a first build."""
    assert changed_images(mini_repo, None) == {key: ["<initial release>"] for key in IMAGES}
