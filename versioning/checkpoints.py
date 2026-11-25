"""SHA-256 manifest of the nnUNet checkpoints (standard library only).

The checkpoints (~32 GB) are gitignored, so the nnUNet image cannot be reproduced from git
alone. ``api/docker/nnunet_torsion/checkpoints.sha256`` is committed instead; the release script
refuses to rebuild the nnUNet image unless the local checkpoints match it.

The format is ``sha256sum``-compatible: one ``<hex digest>  <path relative to the checkpoint
directory>`` line per file, sorted by path, so ``cd checkpoints && sha256sum -c`` also works.

Usage: ``python -m versioning.checkpoints write|verify <checkpoint dir> <manifest>``.
"""
import hashlib
import sys
from pathlib import Path

#: Read size for hashing: 1 MiB keeps memory flat on multi-GB checkpoint files.
CHUNK = 1 << 20


def _sha256(path: Path) -> str:
    """Hash one file in :data:`CHUNK`-sized reads.

    :param path: file to hash.
    :return: lower-case hex SHA-256 digest.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def _files(ckpt_dir: Path, exclude: Path | None = None) -> list[str]:
    """List every regular file under a directory.

    :param ckpt_dir: checkpoint directory.
    :param exclude: a file to leave out (the manifest itself, if it lives inside ``ckpt_dir``).
    :return: sorted POSIX paths relative to ``ckpt_dir``.
    """
    excluded = exclude.resolve() if exclude is not None else None
    return sorted(
        p.relative_to(ckpt_dir).as_posix()
        for p in ckpt_dir.rglob("*")
        if p.is_file() and p.resolve() != excluded
    )


def write_manifest(ckpt_dir: Path, out: Path) -> None:
    """Hash every file under ``ckpt_dir`` and write the manifest.

    :param ckpt_dir: checkpoint directory (e.g. ``api/docker/nnunet_torsion/checkpoints``).
    :param out: manifest file to (over)write; skipped from hashing if it lies inside ``ckpt_dir``.
    """
    ckpt_dir, out = Path(ckpt_dir), Path(out)
    lines = [f"{_sha256(ckpt_dir / rel)}  {rel}\n" for rel in _files(ckpt_dir, exclude=out)]
    out.write_text("".join(lines))


def verify_manifest(ckpt_dir: Path, manifest: Path) -> list[str]:
    """Compare the files under ``ckpt_dir`` with a manifest written by :func:`write_manifest`.

    :param ckpt_dir: checkpoint directory.
    :param manifest: the committed manifest file.
    :return: one message per problem — ``mismatch: <path>`` (content differs),
        ``missing: <path>`` (listed, not on disk), ``extra: <path>`` (on disk, not listed),
        ``malformed manifest line <n>`` — sorted by path; empty when everything matches.
    """
    ckpt_dir, manifest = Path(ckpt_dir), Path(manifest)
    expected: dict[str, str] = {}
    problems = []
    for number, line in enumerate(manifest.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        digest, sep, rel = line.partition("  ")
        if not sep or len(digest) != 64 or not rel:
            problems.append(f"malformed manifest line {number}: {line!r}")
            continue
        expected[rel] = digest
    present = set(_files(ckpt_dir, exclude=manifest))
    for rel in sorted(set(expected) | present):
        if rel not in present:
            problems.append(f"missing: {rel}")
        elif rel not in expected:
            problems.append(f"extra: {rel}")
        elif _sha256(ckpt_dir / rel) != expected[rel]:
            problems.append(f"mismatch: {rel}")
    return problems


def main(argv: list[str] | None = None) -> int:
    """Command-line entry: ``write`` or ``verify`` a checkpoint manifest.

    :param argv: ``[write|verify, <checkpoint dir>, <manifest>]`` (default ``sys.argv[1:]``).
    :return: 0 on success / clean verification, 1 on verification problems, 2 on bad usage.
    """
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 3 or argv[0] not in ("write", "verify"):
        print("usage: python -m versioning.checkpoints write|verify <checkpoint dir> <manifest>")
        return 2
    command, ckpt_dir, manifest = argv[0], Path(argv[1]), Path(argv[2])
    if command == "write":
        write_manifest(ckpt_dir, manifest)
        return 0
    problems = verify_manifest(ckpt_dir, manifest)
    for problem in problems:
        print(problem)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
