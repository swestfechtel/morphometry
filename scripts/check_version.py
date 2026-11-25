"""Check the release rules (version bump, changelog, golden MINOR bump, requirement pins).

Run by CI and the pre-push hook. Exit status 1 prints each violation; 0 prints
``version check ok (<VERSION>)``. The rules live in :mod:`versioning.check`.
"""
import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from versioning.check import check  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    """Run the check and print the outcome.

    :param argv: command-line arguments (default ``sys.argv[1:]``); ``--repo PATH`` selects the
        repository (default: the repository containing this script).
    :return: 1 if any rule is violated, else 0.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args(argv)
    problems = check(args.repo)
    for problem in problems:
        print(problem)
    if problems:
        return 1
    print(f"version check ok ({(args.repo / 'VERSION').read_text().strip()})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
