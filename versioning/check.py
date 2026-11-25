"""The release rules enforced in CI and by the pre-push hook (standard library only).

:func:`check` returns human-readable violations of four rules:

1. ``VERSION`` is a plain ``MAJOR.MINOR.PATCH``.
2. If any image's gated files changed since the last release tag, ``VERSION`` is strictly
   greater than that tag's version and ``CHANGELOG.md`` has a section for it.
3. If a torsion-path golden file changed since the last release tag, the bump is at least MINOR
   (a golden change means measured values moved, which is a do-not-pool event).
4. Every pin in ``morphometry/requirements.txt`` and ``api/requirements.txt`` equals the root
   ``requirements.txt`` pin of the same package.

Everything is read from ``head_ref`` via git, never from the work tree.
"""
import re
import subprocess
from pathlib import Path

from versioning.changelog import section
from versioning.closure import UnresolvedImport
from versioning.images import _env_without_git_overrides, _git, changed_images
from versioning.semver import Version, last_release_tag, parse_version

# Golden files whose change means a torsion-path measurement moved. Derived by reading which
# ``tests/golden/*.json`` the torsion tests load:
#   - ``tests/characterization/test_compute_torsion_pipeline.py`` (the production oracle) loads no
#     golden at all; it compares against ``PA000001`` expectations inline.
#   - ``compute_torsion.py`` calls ``calculate_femoral_torsion`` / ``get_femoral_torsion_landmarks``
#     (lee, murphy) and the tibial equivalents, so the goldens of ``test_torsion.py``,
#     ``test_torsion_head_posterior.py`` (femoral_torsion_*), ``test_condylar_fallback.py`` and
#     ``test_ccd_tibial_equivariance.py`` / ``test_ct.py`` (tibial_torsion_*) are on the list.
#   - Included although ambiguous (module-level closure is the whole package, function-level
#     reachability is narrower; a false positive only costs an extra MINOR bump):
#     ``femoral_torsion_ct_left`` (a ``_ct`` function, not called by compute_torsion),
#     ``knee_rotation_angle_left`` (knee.py composition shares the condylar line) and
#     ``proximal_reference_line_tomczak_left`` (an unvalidated neck-axis method of the femoral
#     torsion landmark function).
#   - Excluded: hip/CCD/CEA/alpha, bone-length, CT head-radius/HKA/MAD, PMA/JLCA and every
#     ``knee*``/``cartilage*`` golden (cartilage-only tests; no torsion consumer).
GOLDEN_ALLOWLIST: tuple[str, ...] = (
    "tests/golden/femoral_torsion_PA000147_right_lee.json",
    "tests/golden/femoral_torsion_PA001023_left_lee.json",
    "tests/golden/femoral_torsion_ct_left.json",
    "tests/golden/femoral_torsion_left_lee.json",
    "tests/golden/femoral_torsion_left_murphy.json",
    "tests/golden/femoral_torsion_right_lee.json",
    "tests/golden/femoral_torsion_right_murphy.json",
    "tests/golden/knee_rotation_angle_left.json",
    "tests/golden/proximal_reference_line_tomczak_left.json",
    "tests/golden/tibial_torsion_ct_left.json",
    "tests/golden/tibial_torsion_ct_right.json",
    "tests/golden/tibial_torsion_fallback_PA000077_left.json",
    "tests/golden/tibial_torsion_fallback_PA000347_left.json",
    "tests/golden/tibial_torsion_fallback_PA000717_right.json",
    "tests/golden/tibial_torsion_left.json",
    "tests/golden/tibial_torsion_right.json",
)

_PIN_RE = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)(?:\[[^\]]*\])?\s*==\s*([^\s;#]+)")


def _show(repo: Path, ref: str, path: str) -> str | None:
    """Read a file as committed at a ref.

    :param repo: path of the git repository.
    :param ref: any commit-ish.
    :param path: repo-relative POSIX path.
    :return: the file's text, or ``None`` if it does not exist at ``ref``.
    """
    try:
        return _git(repo, "show", f"{ref}:{path}", env=_env_without_git_overrides())
    except subprocess.CalledProcessError:
        return None


def parse_pins(text: str) -> dict[str, str]:
    """Extract ``name==version`` pins from a requirements file.

    :param text: requirements file content; comments, options and unpinned lines are ignored.
    :return: normalised package name (lower case, ``_``/``.`` → ``-``) → pinned version.
    """
    pins = {}
    for line in text.splitlines():
        match = _PIN_RE.match(line.strip())
        if match:
            pins[re.sub(r"[-_.]+", "-", match.group(1)).lower()] = match.group(2)
    return pins


def _pin_violations(repo: Path, head_ref: str) -> list[str]:
    """Compare the sub-project requirement pins with the root ones.

    :param repo: path of the git repository.
    :param head_ref: commit to read the files from.
    :return: one message per package whose pin differs from the root pin.
    """
    root_text = _show(repo, head_ref, "requirements.txt")
    if root_text is None:
        return ["requirements.txt is missing"]
    root = parse_pins(root_text)
    problems = []
    for rel in ("morphometry/requirements.txt", "api/requirements.txt"):
        text = _show(repo, head_ref, rel)
        if text is None:
            continue
        for name, version in sorted(parse_pins(text).items()):
            # Packages the root file does not list (the API's own dependencies) are unconstrained.
            if name in root and root[name] != version:
                problems.append(
                    f"{rel} pins {name}=={version} but requirements.txt pins {name}=={root[name]}"
                )
    return problems


def check(repo: Path, head_ref: str = "HEAD") -> list[str]:
    """Check the release rules (see the module docstring) at ``head_ref``.

    :param repo: path of the git repository.
    :param head_ref: commit to check, default ``HEAD``.
    :return: violation messages; empty when compliant. A shallow clone whose ``v*`` tags are
        unreachable is reported as a violation (its message) instead of raising.
    """
    problems = []
    text = _show(repo, head_ref, "VERSION")
    version: Version | None = None
    if text is None:
        problems.append("VERSION file is missing")
    else:
        try:
            version = parse_version(text.strip())
        except ValueError as exc:
            problems.append(f"VERSION is malformed: {exc}")
    problems += _pin_violations(repo, head_ref)
    if version is None:
        return problems

    try:
        tag = last_release_tag(repo)
    except RuntimeError as exc:
        return problems + [str(exc)]
    last = parse_version(tag[1:]) if tag else None
    try:
        changed = changed_images(repo, tag, head_ref)
    except (UnresolvedImport, subprocess.CalledProcessError) as exc:
        return problems + [f"cannot determine changed images: {exc}"]

    if changed:
        what = "; ".join(f"{image}: {', '.join(files)}" for image, files in changed.items())
        if last is not None and version <= last:
            problems.append(f"VERSION {version} must be greater than the last release {tag} because {what} changed")
        else:
            changelog = _show(repo, head_ref, "CHANGELOG.md")
            if changelog is None or section(changelog, str(version)) is None:
                problems.append(f"CHANGELOG.md has no '## [{version}] - YYYY-MM-DD' section for VERSION {version}")

    if tag is not None:
        diff = _git(repo, "diff", "--name-only", "--no-renames", "-z", tag, head_ref, "--", *GOLDEN_ALLOWLIST,
                    env=_env_without_git_overrides())
        for golden in sorted(p for p in diff.split("\0") if p):
            if (version.major, version.minor) <= (last.major, last.minor):
                problems.append(f"golden {golden} changed since {tag}: bump at least MINOR")
    return problems
