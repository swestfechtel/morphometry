"""Build, tag and push the four images of one release — the only path by which images are published.

Usage (from a clean ``main`` that equals ``origin/main``)::

    python scripts/release.py --dry-run        # gather state, print the plan, change nothing
    python scripts/release.py                  # execute it

The version is the committed ``VERSION``. All decisions (refusals, and ``build`` / ``retag`` /
``skip`` per image) are made by :func:`versioning.release_plan.plan_release`; this script only
gathers the state it needs and executes the plan:

1. Gather: Docker Hub ``:<version>`` of each image (``docker manifest inspect -v``, then its
   revision label and digest via ``docker buildx imagetools inspect``), git tag ``v<version>``
   locally and on ``origin`` (``git ls-remote``), work-tree cleanliness, branch, and whether
   ``HEAD`` equals ``origin``'s ``main``. Read-only.
2. Print the plan. Stop here with ``--dry-run``.
3. If nnUNet is rebuilt, verify the checkpoints in its build context
   (``api/docker/nnunet_torsion/checkpoints/``, the files ``docker build`` bakes in) against
   ``api/docker/nnunet_torsion/checkpoints.sha256``. There is deliberately no option to verify
   another directory.
4. Per image in ``IMAGES`` order (nnunet, torsion, api, frontend): re-check that
   ``:<version>`` is still absent, then ``build`` (``docker build`` with OCI labels, re-check
   absence again, ``docker push``) or ``retag`` (``docker buildx imagetools create``:
   registry-side, nothing downloaded); ``skip`` reuses the existing tag. Read back the pushed
   digest. The api image is built with the nnunet/torsion digests of this release, which is why
   those two come first; it is therefore built on every release, never retagged.
5. Write ``release/images.json[<version>]``, commit ``release: v<version>``, create the
   annotated tag ``v<version>`` (message: the changelog section, kept verbatim), push ``main``
   and the tag.
6. Move ``:latest`` of each image to its release digest (convenience only, last).

Standard library only. Exit status: 0 success, 1 refused or failed.
"""
import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

sys.path.append(str(Path(__file__).resolve().parents[1]))

from versioning._git import git  # noqa: E402
from versioning.changelog import section  # noqa: E402
from versioning.check import _show, head_version  # noqa: E402
from versioning.checkpoints import verify_manifest  # noqa: E402
from versioning.images import IMAGES, ImageSpec  # noqa: E402
from versioning.release_plan import (  # noqa: E402
    UNLABELLED, ImageAction, ReleaseRefused, RemoteState, parse_digest_ref, plan_release,
)

REPO = Path(__file__).resolve().parents[1]
MANIFEST = Path("release/images.json")
CHECKPOINT_MANIFEST = Path("api/docker/nnunet_torsion/checkpoints.sha256")
#: The checkpoints the nnUNet ``docker build`` bakes in (its build context); the only directory verified.
CHECKPOINT_DIR = Path("api/docker/nnunet_torsion/checkpoints")
REVISION_LABEL = "org.opencontainers.image.revision"

#: ``run(cmd, cwd, capture)`` executes one command and returns the completed process (never
#: raises on a non-zero exit). Injected so tests can record commands instead of running them.
Runner = Callable[..., subprocess.CompletedProcess]


def default_runner(cmd: list[str], cwd: Path, capture: bool = True) -> subprocess.CompletedProcess:
    """Run a command for real.

    git commands go through :func:`versioning._git.git` (always captured), so ``GIT_DIR`` & co. cannot redirect
    them to another repository.

    :param cmd: argument vector; ``cmd[0]`` is the program.
    :param cwd: working directory.
    :param capture: capture stdout/stderr as text; ``False`` streams them (long builds/pushes).
    :return: the completed process (``check=False``).
    """
    if cmd[0] == "git":  # always captured; git output is short
        return git(cwd, *cmd[1:], check=False)
    return subprocess.run(cmd, cwd=cwd, check=False, text=True, capture_output=capture)


def _checked(run: Runner, cmd: list[str], cwd: Path, capture: bool = True) -> subprocess.CompletedProcess:
    """Run a command and raise if it fails.

    :param run: the runner.
    :param cmd: argument vector.
    :param cwd: working directory.
    :param capture: see :func:`default_runner`.
    :return: the completed process.
    :raises RuntimeError: on a non-zero exit, with the command and its stderr.
    """
    result = run(cmd, cwd=cwd, capture=capture)
    if result.returncode != 0:
        raise RuntimeError(f"command failed ({result.returncode}): {' '.join(cmd)}\n{result.stderr or ''}".rstrip())
    return result


# --------------------------------------------------------------------------- gathering (read-only)

def hub_digest(run: Runner, ref: str, cwd: Path) -> str:
    """Read the manifest digest of a tag on the registry.

    :param run: the runner.
    :param ref: ``repo:tag``.
    :param cwd: working directory.
    :return: ``sha256:<64 hex>``.
    :raises RuntimeError: if the inspection fails or returns no valid digest.
    """
    out = _checked(run, ["docker", "buildx", "imagetools", "inspect", ref, "--format", "{{json .Manifest}}"], cwd)
    digest = json.loads(out.stdout or "{}").get("digest", "")
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", digest):
        raise RuntimeError(f"no valid digest for {ref}: {out.stdout!r}")
    return digest


def hub_tag_state(run: Runner, ref: str, cwd: Path) -> tuple[str | None, str | None]:
    """Find out whether ``repo:tag`` exists on the registry and whose it is.

    :param run: the runner.
    :param ref: ``repo:tag``.
    :param cwd: working directory.
    :return: ``(None, None)`` if absent; else ``(revision label or UNLABELLED, digest)``.
    :raises RuntimeError: if existence cannot be established (network, auth, …) or the labels
        cannot be read — the release must then refuse rather than guess "absent".
    """
    result = run(["docker", "manifest", "inspect", "-v", ref], cwd=cwd, capture=True)
    if result.returncode != 0:
        err = (result.stderr or "") + (result.stdout or "")
        if re.search(r"no such manifest|manifest unknown|manifest for \S+ not found", err, re.IGNORECASE):
            return None, None
        raise RuntimeError(f"cannot tell whether {ref} exists: {err.strip()}")
    labels_out = _checked(
        run, ["docker", "buildx", "imagetools", "inspect", ref, "--format", "{{json .Image.Config.Labels}}"], cwd)
    labels = json.loads(labels_out.stdout or "null") or {}
    return labels.get(REVISION_LABEL) or UNLABELLED, hub_digest(run, ref, cwd)


def gather_remote_state(repo: Path, version: str, head_sha: str, run: Runner) -> RemoteState:
    """Collect the :class:`RemoteState` the planner needs. Read-only.

    :param repo: repository root.
    :param version: version being released.
    :param head_sha: full sha of ``HEAD``.
    :param run: the runner.
    :return: the gathered state.
    :raises RuntimeError: if the registry or ``origin`` cannot be queried.
    """
    existing_tags, existing_digests = {}, {}
    for key, spec in IMAGES.items():
        existing_tags[key], existing_digests[key] = hub_tag_state(run, f"{spec.repo}:{version}", repo)
    tag = f"v{version}"
    local = run(["git", "rev-parse", "-q", "--verify", f"refs/tags/{tag}"], cwd=repo, capture=True)
    remote_tags = _checked(run, ["git", "ls-remote", "--tags", "origin", f"refs/tags/{tag}"], repo).stdout
    remote_main = _checked(run, ["git", "ls-remote", "origin", "refs/heads/main"], repo).stdout.split()
    status = _checked(run, ["git", "status", "--porcelain", "--untracked-files=normal"], repo).stdout
    branch = run(["git", "symbolic-ref", "-q", "--short", "HEAD"], cwd=repo, capture=True)
    return RemoteState(
        existing_tags=existing_tags,
        tag_exists_local=local.returncode == 0,
        tag_exists_remote=bool(remote_tags.strip()),
        clean=not status.strip(),
        on_main=branch.returncode == 0 and branch.stdout.strip() == "main",
        up_to_date=bool(remote_main) and remote_main[0] == head_sha,
        existing_digests=existing_digests,
    )


# --------------------------------------------------------------------------- execution

def utc_now() -> str:
    """Current UTC time as RFC 3339 (``2026-11-25T12:00:00Z``).

    :return: the timestamp.
    """
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def build_command(spec: ImageSpec, version: str, head_sha: str, created: str, changelog: str,
                  refs: dict[str, str]) -> list[str]:
    """The ``docker build`` command for one image (run from the repository root).

    :param spec: the image.
    :param version: version being released.
    :param head_sha: full sha of the commit being built.
    :param created: RFC 3339 UTC build time.
    :param changelog: this version's changelog section (the ``morphometry.changelog`` label).
    :param refs: image key → ``repo@sha256:…`` already resolved in this release.
    :return: the argument vector.
    :raises ValueError: for ``api``, if this release's nnunet or torsion reference is missing or
        not a ``repo@sha256:`` digest of the right repository — an empty build arg would become
        an empty environment variable overriding the release manifest inside the image.
    """
    labels = {
        "org.opencontainers.image.version": version,
        REVISION_LABEL: head_sha,
        "org.opencontainers.image.created": created,
        "morphometry.changelog": changelog,
    }
    build_args: dict[str, str] = {}
    if spec.key in ("api", "frontend"):
        build_args["MORPH_VERSION"] = version
    if spec.key == "api":
        build_args["MORPH_GIT_COMMIT"] = head_sha
        for dep, arg in (("nnunet", "MORPH_NNUNET_IMAGE"), ("torsion", "MORPH_TORSION_IMAGE")):
            ref = refs.get(dep, "")
            parsed = parse_digest_ref(ref)
            if parsed is None or parsed[0] != IMAGES[dep].repo:
                raise ValueError(f"api build needs this release's {dep} as {IMAGES[dep].repo}@sha256:…, got {ref!r}")
            build_args[arg] = ref
    cmd = ["docker", "build", "-f", spec.dockerfile, "-t", f"{spec.repo}:{version}"]
    for name, value in labels.items():
        cmd += ["--label", f"{name}={value}"]
    for name, value in build_args.items():
        cmd += ["--build-arg", f"{name}={value}"]
    return cmd + [spec.context]


def _ensure_absent(run: Runner, tag: str, repo: Path) -> None:
    """Refuse to publish ``tag`` if it exists on the registry (never overwrite a ``:<version>``).

    :param run: the runner.
    :param tag: ``repo:<version>``.
    :param repo: working directory.
    :raises RuntimeError: if the tag exists or its existence cannot be established.
    """
    if hub_tag_state(run, tag, repo) != (None, None):
        raise RuntimeError(f"{tag} appeared on Docker Hub since planning; refusing to overwrite it")


def execute(repo: Path, version: str, head_sha: str, plan: list[ImageAction], run: Runner,
            today: str | None = None, created: str | None = None) -> dict:
    """Carry out a plan: build/retag/skip, record, commit, tag, push, then move ``:latest``.

    :param repo: repository root.
    :param version: version being released.
    :param head_sha: full sha of the commit being released.
    :param plan: the planner's actions, in ``IMAGES`` order.
    :param run: the runner.
    :param today: release date ``YYYY-MM-DD``; default today (UTC).
    :param created: RFC 3339 build time for the labels; default now (UTC).
    :return: the ``release/images.json`` entry written.
    :raises ReleaseRefused: if the checkpoints do not match their manifest.
    :raises RuntimeError: if a command fails or a ``:<version>`` appeared since planning.
    :raises ValueError: see :func:`build_command`.
    """
    created = created or utc_now()
    today = today or created[:10]
    actions = {a.key: a for a in plan}
    if actions["nnunet"].action == "build":
        manifest = repo / CHECKPOINT_MANIFEST
        if not manifest.exists():
            raise ReleaseRefused([f"{CHECKPOINT_MANIFEST} is missing; nnunet cannot be rebuilt reproducibly"])
        # Always the build context: that is what `docker build` copies into the image.
        print(f"verifying checkpoints in {CHECKPOINT_DIR} …", flush=True)
        problems = verify_manifest(repo / CHECKPOINT_DIR, manifest)
        if problems:
            raise ReleaseRefused([f"checkpoints: {p}" for p in problems])
    changelog = section(_show(repo, head_sha, "CHANGELOG.md") or "", version) or ""

    refs: dict[str, str] = {}
    for key, spec in IMAGES.items():
        action = actions[key]
        tag = f"{spec.repo}:{version}"
        if action.action != "skip":
            # Never overwrite a :<version> (the anti-goal): check before the (hours-long) build
            # and again immediately before each publishing command.
            _ensure_absent(run, tag, repo)
            if action.action == "build":
                print(f"building {tag} …", flush=True)
                _checked(run, build_command(spec, version, head_sha, created, changelog, refs), repo, capture=False)
                _ensure_absent(run, tag, repo)
                _checked(run, ["docker", "push", tag], repo, capture=False)
            else:
                print(f"tagging {action.from_ref} as {tag} …", flush=True)
                _checked(run, ["docker", "buildx", "imagetools", "create", "-t", tag, action.from_ref], repo)
        refs[key] = f"{spec.repo}@{hub_digest(run, tag, repo)}"
        if action.action == "retag" and parse_digest_ref(refs[key])[1] != parse_digest_ref(action.from_ref)[1]:
            print(f"warning: {tag} resolved to {refs[key]}, not {action.from_ref}", flush=True)
        print(f"{key}: {refs[key]}", flush=True)

    entry = {**refs, "git_commit": head_sha, "date": today}
    path = repo / MANIFEST
    manifest_data = json.loads(path.read_text()) if path.exists() else {}
    manifest_data[version] = entry
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest_data, indent=2) + "\n")

    tag = f"v{version}"
    _checked(run, ["git", "add", str(MANIFEST)], repo)
    _checked(run, ["git", "commit", "-m", f"release: {tag}"], repo)
    # verbatim: the changelog section has "### …" lines, which the default cleanup strips as comments.
    _checked(run, ["git", "tag", "-a", "--cleanup=verbatim", tag, "-m", changelog or tag], repo)
    _checked(run, ["git", "push", "origin", "main", tag], repo)

    # Convenience only, after the release is recorded and pushed (a failure here leaves no
    # :latest pointing at an unrecorded release; re-run these commands by hand).
    for key, spec in IMAGES.items():
        _checked(run, ["docker", "buildx", "imagetools", "create", "-t", f"{spec.repo}:latest", refs[key]], repo)
    return entry


def format_plan(version: str, plan: list[ImageAction]) -> str:
    """Render a plan for printing.

    :param version: version being released.
    :param plan: the planner's actions.
    :return: one line per image: key, action, target tag, source (for retag) and reason.
    """
    lines = [f"release plan for {version}:"]
    for a in plan:
        source = f" <- {a.from_ref}" if a.from_ref else ""
        lines.append(f"  {a.key:<9}{a.action:<7}{IMAGES[a.key].repo}:{version}{source}  ({a.reason})")
    return "\n".join(lines)


def main(argv: list[str] | None = None, run: Runner = default_runner) -> int:
    """Plan and (unless ``--dry-run``) execute a release.

    :param argv: command-line arguments (default ``sys.argv[1:]``): ``--dry-run``,
        ``--adopt KEY`` (repeatable; initial release only: take over an existing unlabelled
        ``:<version>``), ``--repo PATH`` (default: the repository containing this script).
    :param run: command runner (tests inject a recorder).
    :return: 0 on success, 1 if refused or failed.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="print the plan; build and push nothing")
    parser.add_argument("--adopt", action="append", default=[], choices=list(IMAGES),
                        help="initial release only: adopt an existing unlabelled :<version> of this image")
    parser.add_argument("--repo", type=Path, default=REPO)
    args = parser.parse_args(argv)
    repo = args.repo.resolve()

    try:
        head_sha = _checked(run, ["git", "rev-parse", "HEAD"], repo).stdout.strip()
        version = head_version(repo, head_sha)
        manifest_path = repo / MANIFEST
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        remote = gather_remote_state(repo, version, head_sha, run)
        plan = plan_release(repo, version, head_sha, remote, manifest, adopt=frozenset(args.adopt))
    except ReleaseRefused as exc:
        print(exc)
        return 1
    except (RuntimeError, ValueError) as exc:
        print(f"release refused: {exc}")
        return 1

    print(format_plan(version, plan))
    if args.dry_run:
        if any(a.key == "nnunet" and a.action == "build" for a in plan):
            print(f"(dry run: checkpoints in {CHECKPOINT_DIR} would be verified before building nnunet)")
        print("dry run: nothing built, tagged or pushed")
        return 0
    try:
        entry = execute(repo, version, head_sha, plan, run)
    except ReleaseRefused as exc:
        print(exc)
        return 1
    except (RuntimeError, ValueError) as exc:
        print(f"release failed: {exc}\n"
              "A re-run skips the images already pushed from this commit. If the failure was in the final "
              "git commit/tag/push or the :latest moves, finish those steps by hand instead (a re-run refuses "
              "once v<version> exists).")
        return 1
    print(json.dumps({version: entry}, indent=2))
    print(f"released v{version}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
