"""Tests for ``scripts/release.py`` with a recording fake runner: no docker, no network, no push."""
import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

from tests.versioning.conftest import run_git
from tests.versioning.test_release_plan import RELEASE_FILES, _write_all
from versioning.checkpoints import write_manifest
from versioning.images import IMAGES

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "release.py"
_spec = importlib.util.spec_from_file_location("release_script", _SCRIPT)
release = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(release)

DIGESTS = {"nnunet": "1", "torsion": "2", "api": "3", "frontend": "4"}


class FakeRunner:
    """Records every command; answers the read-only ones like an empty registry would.

    :param head_sha: sha reported for ``HEAD``.
    :param existing: ``repo:tag`` refs that exist on the fake registry.
    :param origin_main: sha reported as ``origin``'s ``main``; default ``head_sha``.
    """

    def __init__(self, head_sha: str, existing: set[str] = frozenset(), origin_main: str | None = None):
        self.head_sha = head_sha
        self.origin_main = origin_main or head_sha
        self.existing = set(existing)
        self.calls: list[list[str]] = []

    def _digest_for(self, ref: str) -> str:
        """Fake digest of ``repo:tag`` / ``repo@digest``: one hex char per image, repeated.

        :param ref: image reference.
        :return: ``sha256:…``.
        """
        repo = ref.split("@")[0].rsplit(":", 1)[0]
        key = next(k for k, s in IMAGES.items() if s.repo == repo)
        return "sha256:" + DIGESTS[key] * 64

    def __call__(self, cmd, cwd, capture=True):
        self.calls.append(list(cmd))
        ok = lambda out="": subprocess.CompletedProcess(cmd, 0, out, "")  # noqa: E731
        if cmd[:3] == ["docker", "manifest", "inspect"]:
            if cmd[-1] in self.existing:
                return ok("{}")
            return subprocess.CompletedProcess(cmd, 1, "", f"no such manifest: docker.io/{cmd[-1]}")
        if cmd[:4] == ["docker", "buildx", "imagetools", "inspect"]:
            if "{{json .Manifest}}" in cmd:
                return ok(json.dumps({"digest": self._digest_for(cmd[4])}))
            return ok("null")
        if cmd[:4] == ["docker", "buildx", "imagetools", "create"]:
            self.existing.add(cmd[5])
            return ok()
        if cmd[:2] == ["docker", "push"]:
            self.existing.add(cmd[2])
            return ok()
        if cmd[:2] == ["git", "ls-remote"]:
            return ok(f"{self.origin_main}\trefs/heads/main\n" if cmd[-1] == "refs/heads/main" else "")
        if cmd[:2] == ["git", "symbolic-ref"]:
            return ok("main\n")
        if cmd[:3] == ["git", "rev-parse", "-q"]:
            return subprocess.CompletedProcess(cmd, 1, "", "")
        if cmd[:2] == ["git", "rev-parse"]:
            return ok(self.head_sha + "\n")
        if cmd[:2] == ["git", "status"]:
            return ok("")
        return ok()

    def mutating(self) -> list[list[str]]:
        """Commands that would change docker, registry or git state.

        :return: the recorded mutating commands, in order.
        """
        reads = (["docker", "manifest", "inspect"], ["docker", "buildx", "imagetools", "inspect"])
        git_reads = ("ls-remote", "symbolic-ref", "rev-parse", "status")
        return [c for c in self.calls
                if not any(c[:len(r)] == r for r in reads) and not (c[0] == "git" and c[1] in git_reads)]


@pytest.fixture
def initial(git_repo: Path) -> Path:
    """A repository ready for its first release, with a checkpoint manifest and checkpoints.

    :param git_repo: the throwaway repository from ``conftest.py``.
    :return: path of the repository root.
    """
    _write_all(git_repo, RELEASE_FILES)
    ckpt = git_repo / release.CHECKPOINT_DIR  # the nnUNet build context, gitignored as in the real repo
    ckpt.mkdir(parents=True)
    (ckpt / "hip.pth").write_bytes(b"weights")
    write_manifest(ckpt, git_repo / release.CHECKPOINT_MANIFEST)  # replaces MINI_FILES' placeholder
    _write_all(git_repo, {".gitignore": "api/docker/nnunet_torsion/checkpoints/\n"})
    return git_repo


def _sha(repo: Path) -> str:
    """Full sha of ``HEAD``.

    :param repo: repository root.
    :return: the sha.
    """
    return run_git(repo, "rev-parse", "HEAD")


def test_build_command_api_refuses_empty_refs():
    """An empty MORPH_NNUNET_IMAGE/MORPH_TORSION_IMAGE would override the manifest in the image."""
    good = {"nnunet": "swestfechtel/nnunet_torsion@sha256:" + "1" * 64}
    with pytest.raises(ValueError, match="torsion"):
        release.build_command(IMAGES["api"], "1.0.0", "f" * 40, "2026-11-25T00:00:00Z", "x", good)
    with pytest.raises(ValueError, match="torsion"):
        release.build_command(IMAGES["api"], "1.0.0", "f" * 40, "t", "x", {**good, "torsion": ""})
    with pytest.raises(ValueError, match="nnunet"):
        release.build_command(IMAGES["api"], "1.0.0", "f" * 40, "t", "x",
                              {"nnunet": "swestfechtel/torsion@sha256:" + "1" * 64,
                               "torsion": "swestfechtel/torsion@sha256:" + "2" * 64})


def test_build_command_labels_and_args():
    refs = {"nnunet": "swestfechtel/nnunet_torsion@sha256:" + "1" * 64,
            "torsion": "swestfechtel/torsion@sha256:" + "2" * 64}
    cmd = release.build_command(IMAGES["api"], "1.2.0", "f" * 40, "2026-11-25T10:00:00Z", "- api: x.", refs)
    assert cmd[:6] == ["docker", "build", "-f", "api/Dockerfile", "-t", "swestfechtel/morphometry_api:1.2.0"]
    assert cmd[-1] == "."
    pairs = list(zip(cmd[::1], cmd[1::1]))
    for label in ("org.opencontainers.image.version=1.2.0", "org.opencontainers.image.revision=" + "f" * 40,
                  "org.opencontainers.image.created=2026-11-25T10:00:00Z", "morphometry.changelog=- api: x."):
        assert ("--label", label) in pairs
    for arg in ("MORPH_VERSION=1.2.0", "MORPH_GIT_COMMIT=" + "f" * 40,
                "MORPH_NNUNET_IMAGE=" + refs["nnunet"], "MORPH_TORSION_IMAGE=" + refs["torsion"]):
        assert ("--build-arg", arg) in pairs
    torsion = release.build_command(IMAGES["torsion"], "1.2.0", "f" * 40, "t", "c", {})
    assert "--build-arg" not in torsion and torsion[-1] == "morphometry"
    frontend = release.build_command(IMAGES["frontend"], "1.2.0", "f" * 40, "t", "c", {})
    assert ("--build-arg", "MORPH_VERSION=1.2.0") in list(zip(frontend, frontend[1:]))


def test_dry_run_only_reads(initial, capsys):
    runner = FakeRunner(_sha(initial))
    assert release.main(["--dry-run", "--repo", str(initial)], run=runner) == 0
    out = capsys.readouterr().out
    assert out.count(" build ") == 4 and "1.0.0" in out
    assert runner.mutating() == []


def test_dry_run_refusal_is_reported(initial, capsys):
    runner = FakeRunner(_sha(initial), origin_main="0" * 40)
    assert release.main(["--dry-run", "--repo", str(initial)], run=runner) == 1
    assert "origin/main" in capsys.readouterr().out
    assert runner.mutating() == []


def test_execute_initial_release(initial):
    head = _sha(initial)
    runner = FakeRunner(head)
    assert release.main(["--repo", str(initial)], run=runner) == 0
    mutating = runner.mutating()
    builds = [c for c in mutating if c[:2] == ["docker", "build"]]
    assert [c[5] for c in builds] == [f"{IMAGES[k].repo}:1.0.0" for k in IMAGES]
    api_build = builds[2]
    assert "MORPH_NNUNET_IMAGE=swestfechtel/nnunet_torsion@sha256:" + "1" * 64 in api_build
    assert "MORPH_TORSION_IMAGE=swestfechtel/torsion@sha256:" + "2" * 64 in api_build
    assert [c for c in mutating if c[:2] == ["docker", "push"]] == [
        ["docker", "push", f"{IMAGES[k].repo}:1.0.0"] for k in IMAGES]
    latest = [c for c in mutating if c[:4] == ["docker", "buildx", "imagetools", "create"]]
    assert [c[5] for c in latest] == [f"{IMAGES[k].repo}:latest" for k in IMAGES]
    git_calls = [c for c in mutating if c[0] == "git"]
    assert git_calls[1] == ["git", "commit", "-m", "release: v1.0.0"]
    assert git_calls[2][:4] == ["git", "tag", "-a", "--cleanup=verbatim"] and "torsion: first." in git_calls[2][-1]
    assert git_calls[3] == ["git", "push", "origin", "main", "v1.0.0"]
    # :latest moves only after the release is recorded and pushed.
    assert mutating.index(latest[0]) > mutating.index(git_calls[3])
    entry = json.loads((initial / "release/images.json").read_text())["1.0.0"]
    assert entry["api"] == "swestfechtel/morphometry_api@sha256:" + "3" * 64
    assert entry["git_commit"] == head and set(entry) == {*IMAGES, "git_commit", "date"}


def test_execute_refuses_bad_checkpoints(initial, tmp_path, capsys):
    """A mismatch in the build-context checkpoints refuses, even if an intact copy exists elsewhere."""
    intact = tmp_path / "intact"
    intact.mkdir()
    (intact / "hip.pth").write_bytes(b"weights")
    (initial / release.CHECKPOINT_DIR / "hip.pth").write_bytes(b"other")
    runner = FakeRunner(_sha(initial))
    with pytest.raises(SystemExit):  # no way to point verification at another directory
        release.main(["--repo", str(initial), "--checkpoints-dir", str(intact)], run=runner)
    assert release.main(["--repo", str(initial)], run=runner) == 1
    assert "mismatch" in capsys.readouterr().out
    assert runner.mutating() == []


def test_execute_refuses_tag_that_appeared(initial):
    """A :<version> pushed between planning and publishing is never overwritten."""
    head = _sha(initial)
    runner = FakeRunner(head)
    plan = [release.ImageAction(k, "build", None, "x") for k in IMAGES]
    runner.existing.add("swestfechtel/torsion:1.0.0")
    with pytest.raises(RuntimeError, match="torsion:1.0.0 appeared"):
        release.execute(initial, "1.0.0", head, plan, runner)
    pushes = [c for c in runner.mutating() if c[:2] == ["docker", "push"]]
    assert pushes == [["docker", "push", "swestfechtel/nnunet_torsion:1.0.0"]]


def test_execute_retag_is_registry_side(initial):
    head = _sha(initial)
    runner = FakeRunner(head)
    src = "swestfechtel/nnunet_torsion@sha256:" + "1" * 64
    plan = [release.ImageAction("nnunet", "retag", src, "unchanged"),
            *(release.ImageAction(k, "skip", None, "x") for k in ("torsion", "api", "frontend"))]
    release.execute(initial, "1.0.0", head, plan, runner)
    mutating = runner.mutating()
    assert mutating[0] == ["docker", "buildx", "imagetools", "create", "-t", "swestfechtel/nnunet_torsion:1.0.0", src]
    assert not any(c[:2] in (["docker", "build"], ["docker", "push"]) for c in mutating)


def test_hub_tag_state_fails_safe():
    """A registry error other than "no such manifest" must not be read as "absent"."""
    def run(cmd, cwd, capture=True):
        return subprocess.CompletedProcess(cmd, 1, "", "error: unauthorized: authentication required")
    with pytest.raises(RuntimeError, match="cannot tell"):
        release.hub_tag_state(run, "swestfechtel/torsion:1.0.0", Path("."))


def test_hub_tag_state_unlabelled_and_labelled():
    runner = FakeRunner("f" * 40, existing={"swestfechtel/torsion:1.0.0"})
    revision, digest = release.hub_tag_state(runner, "swestfechtel/torsion:1.0.0", Path("."))
    assert revision == release.UNLABELLED and digest == "sha256:" + "2" * 64

    def labelled(cmd, cwd, capture=True):
        if "{{json .Image.Config.Labels}}" in cmd:
            return subprocess.CompletedProcess(cmd, 0, json.dumps({release.REVISION_LABEL: "abc"}), "")
        return runner(cmd, cwd, capture)
    assert release.hub_tag_state(labelled, "swestfechtel/torsion:1.0.0", Path("."))[0] == "abc"


def test_execute_rechecks_tag_between_build_and_push(initial):
    """A :<version> pushed by someone else during the (hours-long) build is not overwritten."""
    head = _sha(initial)
    plan = [release.ImageAction(k, "build", None, "x") for k in IMAGES]

    class AppearsDuringBuild(FakeRunner):
        def __call__(self, cmd, cwd, capture=True):
            if cmd[:2] == ["docker", "build"] and cmd[5] == "swestfechtel/nnunet_torsion:1.0.0":
                self.existing.add(cmd[5])
            return super().__call__(cmd, cwd, capture)

    runner = AppearsDuringBuild(head)
    with pytest.raises(RuntimeError, match="nnunet_torsion:1.0.0 appeared"):
        release.execute(initial, "1.0.0", head, plan, runner)
    assert not any(c[:2] == ["docker", "push"] for c in runner.calls)


def test_hub_tag_state_credential_helper_error_is_not_absent():
    """"executable file not found" (a missing credential helper) must not read as an absent tag."""
    def run(cmd, cwd, capture=True):
        return subprocess.CompletedProcess(
            cmd, 1, "", 'error getting credentials - err: exec: "docker-credential-desktop": '
                        "executable file not found in $PATH")
    with pytest.raises(RuntimeError, match="cannot tell"):
        release.hub_tag_state(run, "swestfechtel/torsion:1.0.0", Path("."))
