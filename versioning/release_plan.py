"""Release decisions: what to do with each image for one release (standard library only).

:func:`plan_release` is pure apart from git reads: it takes the remote state that
``scripts/release.py`` gathered (Docker Hub tags, git tags, work-tree state) and either returns
one :class:`ImageAction` per image, in :data:`versioning.images.IMAGES` order, or raises
:class:`ReleaseRefused` listing every reason the release must not happen. The script only
executes a plan; every decision lives here so it can be tested without docker.

Per image the action is:

- ``skip`` — ``:<version>`` already exists on Docker Hub and is ours: its revision label is the
  commit being released (an interrupted run resumed), or it already points at the digest a
  ``retag`` would have used, or (initial release only, with ``adopt``) it was pushed by hand
  before versioning existed and carries no revision label.
- ``build`` — the image's gated files changed since the last release tag (or there is none).
  The **api image is always built**: it bakes this release's version and the nnunet/torsion
  image references in at build time (``api/Dockerfile`` build args), so retagging the previous
  api image would make release N+1's stack launch release N's model images and report N's
  version. Only the resume ``skip`` below applies to it.
- ``retag`` — unchanged: add ``:<version>`` to the previous release's digest, read from
  ``release/images.json`` (no rebuild, no upload). Never for ``api``.

The anti-goal is a ``:<version>`` tag whose digest changes, so any other existing tag refuses.
"""
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from versioning.changelog import section
from versioning.check import _show, check, head_version
from versioning.closure import UnresolvedImport
from versioning.images import IMAGES, changed_images
from versioning.semver import last_release_tag, parse_version

#: ``RemoteState.existing_tags`` value for a tag that exists but has no revision label.
UNLABELLED = "<unlabelled>"

_DIGEST_REF = re.compile(r"^(?P<repo>[^@\s]+)@(?P<digest>sha256:[0-9a-f]{64})$")


class ReleaseRefused(Exception):
    """The release must not proceed.

    :param messages: every reason found, one human-readable message each.
    """

    def __init__(self, messages: list[str]):
        super().__init__("release refused:\n" + "\n".join(f"- {m}" for m in messages))
        self.messages = messages


@dataclass(frozen=True)
class RemoteState:
    """Everything outside the repository's committed content that the plan depends on.

    :param existing_tags: image key → state of ``<repo>:<version>`` on Docker Hub: ``None`` if
        absent, its ``org.opencontainers.image.revision`` label if present, or
        :data:`UNLABELLED` if present without that label.
    :param tag_exists_local: git tag ``v<version>`` exists locally.
    :param tag_exists_remote: git tag ``v<version>`` exists on ``origin``.
    :param clean: the work tree has no modified, staged or untracked files.
    :param on_main: ``HEAD`` is the branch ``main``.
    :param up_to_date: ``HEAD`` equals ``origin``'s ``main``.
    :param existing_digests: image key → manifest digest (``sha256:…``) of an existing
        ``:<version>``; consulted only to recognise a finished ``retag`` when resuming.
    """

    existing_tags: dict[str, str | None]
    tag_exists_local: bool
    tag_exists_remote: bool
    clean: bool
    on_main: bool
    up_to_date: bool
    existing_digests: dict[str, str | None] = field(default_factory=dict)


@dataclass(frozen=True)
class ImageAction:
    """What the release does with one image.

    :param key: image key (``nnunet``, ``torsion``, ``api``, ``frontend``).
    :param action: ``build``, ``retag`` or ``skip`` (see the module docstring).
    :param from_ref: for ``retag``, the previous release's ``repo@sha256:…``; else ``None``.
    :param reason: human-readable justification, printed with the plan.
    """

    key: str
    action: Literal["build", "retag", "skip"]
    from_ref: str | None
    reason: str


def parse_digest_ref(ref: str) -> tuple[str, str] | None:
    """Split a ``repo@sha256:<64 hex>`` reference.

    :param ref: candidate reference.
    :return: ``(repo, digest)``, or ``None`` if ``ref`` is not of that form.
    """
    match = _DIGEST_REF.match(ref or "")
    return (match.group("repo"), match.group("digest")) if match else None


def _state_refusals(version: str, remote: RemoteState) -> list[str]:
    """Refusals that depend only on the work tree and the git tags.

    :param version: version being released.
    :param remote: gathered state.
    :return: messages, possibly empty.
    """
    problems = []
    if not remote.clean:
        problems.append("working tree is not clean (commit or remove modified/untracked files)")
    if not remote.on_main:
        problems.append("not on branch main")
    if not remote.up_to_date:
        problems.append("HEAD is not origin/main (pull, or push your commits so CI checks them, first)")
    if remote.tag_exists_local:
        problems.append(f"git tag v{version} already exists locally")
    if remote.tag_exists_remote:
        problems.append(f"git tag v{version} already exists on origin")
    return problems


def _version_refusals(repo: Path, version: str, head_sha: str, last_tag: str | None) -> list[str]:
    """Refusals about the version itself, beyond what :func:`versioning.check.check` covers.

    ``check`` only demands a bump and a changelog section when a gated file changed; a release
    always needs both (the changelog section becomes the tag message and image label).

    :param repo: path of the git repository.
    :param version: version being released.
    :param head_sha: commit being released.
    :param last_tag: the last ``v*`` tag reachable from ``head_sha``, or ``None``.
    :return: messages, possibly empty.
    """
    problems = []
    committed = head_version(repo, head_sha)
    if committed != version:
        problems.append(f"releasing {version} but VERSION at {head_sha[:12]} is {committed!r}")
    try:
        parsed = parse_version(version)
    except ValueError as exc:
        return problems + [str(exc)]
    if last_tag is not None and parsed <= parse_version(last_tag[1:]):
        problems.append(f"version {version} is not greater than the last release {last_tag}")
    if section(_show(repo, head_sha, "CHANGELOG.md") or "", version) is None:
        problems.append(f"CHANGELOG.md has no section for {version}")
    return problems


def plan_release(
    repo: Path,
    version: str,
    head_sha: str,
    remote: RemoteState,
    manifest: dict,
    adopt: frozenset[str] = frozenset(),
) -> list[ImageAction]:
    """Decide, per image, whether the release builds, retags or skips it.

    The api image is always built (never retagged), because it bakes this release's version and
    model-image references; see the module docstring.

    :param repo: path of the git repository.
    :param version: version being released (must equal the committed ``VERSION``).
    :param head_sha: full sha of the commit being released.
    :param remote: Docker Hub / git / work-tree state gathered by the release script.
    :param manifest: parsed ``release/images.json`` (version → ``{image key: repo@digest, ...}``).
    :param adopt: image keys whose existing, *unlabelled* ``:<version>`` may be taken over as
        this release's image. Honoured only for the initial release (no previous ``v*`` tag),
        i.e. for images pushed by hand before versioning existed.
    :return: one :class:`ImageAction` per image, in :data:`versioning.images.IMAGES` order.
    :raises ReleaseRefused: with every reason found — ``check()`` violations; dirty / not main /
        not up to date; ``v<version>`` git tag exists; VERSION/changelog problems; an existing
        ``:<version>`` that is not ours; a ``retag`` without a usable previous manifest entry.
    """
    problems = list(check(repo, head_sha))
    problems += _state_refusals(version, remote)
    unknown_adopt = sorted(set(adopt) - set(IMAGES))
    if unknown_adopt:
        problems.append(f"unknown image key(s) to adopt: {', '.join(unknown_adopt)}")
    try:
        last_tag = last_release_tag(repo, head_sha)
    except RuntimeError as exc:
        raise ReleaseRefused(problems + [str(exc)]) from exc
    problems += _version_refusals(repo, version, head_sha, last_tag)
    try:
        changed = changed_images(repo, last_tag, head_sha)
    except (UnresolvedImport, subprocess.CalledProcessError) as exc:
        raise ReleaseRefused(problems + [f"cannot determine changed images: {exc}"]) from exc

    previous = manifest.get(last_tag[1:], {}) if last_tag else {}
    plan = []
    for key, spec in IMAGES.items():
        # 1. What the release would do if Hub had no :<version> yet.
        if key in changed:
            action = ImageAction(key, "build", None, f"changed: {', '.join(changed[key])}")
        elif key == "api":
            # Never retagged: the image bakes this release's version and model-image refs.
            action = ImageAction(key, "build", None, "always built: bakes this release's version and "
                                 "nnunet/torsion image references")
        else:
            from_ref = previous.get(key)
            parsed = parse_digest_ref(from_ref) if from_ref else None
            if from_ref is None:
                problems.append(f"{key}: unchanged since {last_tag}, but release/images.json has no "
                                f"{key} entry for {last_tag[1:]} to retag")
            elif parsed is None or parsed[0] != spec.repo:
                problems.append(f"{key}: release/images.json entry {from_ref!r} for {last_tag[1:]} "
                                f"is not a {spec.repo}@sha256:… reference")
            action = ImageAction(key, "retag", from_ref, f"unchanged since {last_tag}")

        # 2. Reconcile with an existing :<version> on Hub (never overwrite one that is not ours).
        existing = remote.existing_tags.get(key)
        tag = f"{spec.repo}:{version}"
        if existing is None:
            plan.append(action)
        elif existing == head_sha:
            plan.append(ImageAction(key, "skip", None, f"{tag} exists, built from {head_sha[:12]} (resume)"))
        elif existing == UNLABELLED and key in adopt and last_tag is None:
            plan.append(ImageAction(key, "skip", None, f"{tag} exists without a revision label; adopted"))
        elif (action.action == "retag" and action.from_ref
              and remote.existing_digests.get(key) == (parse_digest_ref(action.from_ref) or ("", ""))[1]):
            plan.append(ImageAction(key, "skip", None, f"{tag} already points at {action.from_ref} (resume)"))
        else:
            if existing == UNLABELLED:
                why = "has no revision label" + (
                    "; --adopt is only allowed for the initial release" if key in adopt
                    else "; pass --adopt " + key + " to take it over on the initial release"
                    if last_tag is None else "")
            else:
                why = f"was built from {existing}, not {head_sha[:12]}"
            problems.append(f"{key}: {tag} already exists on Docker Hub and {why}; refusing to overwrite it")
            plan.append(action)

    if problems:
        raise ReleaseRefused(problems)
    return plan
