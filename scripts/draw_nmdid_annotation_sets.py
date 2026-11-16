"""Draw the held-out NMDID annotation sets for choosing (set A) and quoting (set B) thresholds.

The threshold fixes of the NMDID outlier review (``docs/evidence/nmdid_outlier_review.md``: #4 the
lesser-trochanter distance bound, #6 a fixed-mm hip crop, #7 the knee crop end) must not be chosen on
the cases that motivated them, nor quoted on the cases they were chosen on. This script draws, once
and reproducibly, two disjoint random sets of cases from the cohort's ``usable`` verdicts:

* **set A** -- thresholds are chosen on these legs;
* **set B** -- results are quoted on these legs and are never used for tuning;
* **repeat** -- a few set-A cases annotated a second time, at least a week later, for intra-rater
  error (optional).

Excluded before drawing: the 84 reviewed cases (``docs/evidence/nmdid_annotation/reviewed_cases.txt``),
every six-digit case id named anywhere in ``CLAUDE.md``, ``docs/``, ``scripts/`` or ``tests/`` (a
conservative net for cases used elsewhere), and the verification-sample cases (which carry manually
corrected masks and back the femoral-head-slab decision). Cases are drawn whole (both legs), and the
annotation order shuffles A and B together so learning and fatigue effects are not confounded with the set.

Usage::

    python scripts/draw_nmdid_annotation_sets.py            # writes the default CSV
    python scripts/draw_nmdid_annotation_sets.py --n-a 20 --n-b 20 --n-repeat 5 --seed 20261116

The output CSV has one row per case: ``order, case, set, repeat``. Re-running with the same arguments
reproduces it exactly; the committed CSV is the record -- do not redraw after annotation has started.
"""
import argparse
import csv
import random
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
NMDID = Path('/home/simon/sshfs/hpcproject/workspace_simon/nmdid')
DEFAULT_OUT = REPO / 'docs' / 'evidence' / 'nmdid_annotation' / 'annotation_sets.csv'
REVIEWED = REPO / 'docs' / 'evidence' / 'nmdid_annotation' / 'reviewed_cases.txt'
CASE_ID = re.compile(r'(?<!\d)1\d{5}(?!\d)')  # six digits starting with 1, not inside a longer number


def read_reviewed(path: Path) -> set:
    """Read the reviewed case ids (whitespace-separated, ``#`` comments ignored).

    :param path: The reviewed-cases text file.
    :return: The set of six-digit case ids as strings.
    """
    ids = set()
    for line in path.read_text().splitlines():
        line = line.split('#', 1)[0]
        ids.update(tok for tok in line.split() if tok)
    return ids


def ids_named_in_repo(repo: Path) -> set:
    """Collect every six-digit NMDID-style case id mentioned in the repo's docs, scripts and tests.

    Deliberately over-inclusive: excluding a case that was merely mentioned costs one case out of
    ~1400, while including one that was used to build or check a gate would leak.
    :param repo: The repository root.
    :return: The set of ids as strings.
    """
    files = [repo / 'CLAUDE.md']
    for sub, pattern in (('docs', '*.md'), ('scripts', '*.py'), ('tests', '*.py')):
        files.extend((repo / sub).rglob(pattern))
    ids = set()
    for f in files:
        if f.name == Path(__file__).name or 'nmdid_annotation' in f.parts:
            continue  # this script's own examples, and the annotation sets themselves
        ids.update(CASE_ID.findall(f.read_text(errors='ignore')))
    return ids


def verification_ids(nmdid: Path) -> set:
    """Return the case ids of the verification sample (manually corrected masks).

    :param nmdid: The NMDID workspace directory.
    :return: The set of ids as strings.
    """
    gt = nmdid / 'verification_sample' / 'ground_truth'
    return {m.group(0) for p in gt.iterdir() if (m := CASE_ID.search(p.name))}


def usable_ids(nmdid: Path) -> list:
    """Return the ids of cases with verdict ``usable`` whose CT and segmentation both exist, sorted.

    :param nmdid: The NMDID workspace directory.
    :return: A sorted list of ids as strings (sorted so the draw does not depend on file order).
    """
    segs = {m.group(0) for p in (nmdid / 'segmentations').iterdir() if (m := CASE_ID.search(p.name))}
    cts = {m.group(0) for p in (nmdid / 'nifti').iterdir() if (m := CASE_ID.search(p.name))}
    with open(nmdid / 'cohort_status.csv') as fh:
        usable = {row['case'].removeprefix('case-') for row in csv.DictReader(fh) if row['verdict'] == 'usable'}
    return sorted(usable & segs & cts)


def draw(pool: list, n_a: int, n_b: int, n_repeat: int, seed: int) -> list:
    """Draw sets A and B and the repeat subset of A, and a shuffled annotation order.

    :param pool: The eligible case ids, sorted.
    :param n_a: Number of set-A cases.
    :param n_b: Number of set-B cases.
    :param n_repeat: Number of set-A cases to annotate a second time.
    :param seed: The random seed.
    :return: Rows ``{'order', 'case', 'set', 'repeat'}`` in annotation order.
    """
    rng = random.Random(seed)
    chosen = rng.sample(pool, n_a + n_b)
    set_a, set_b = chosen[:n_a], chosen[n_a:]
    repeat = set(rng.sample(set_a, n_repeat))
    rows = [{'case': c, 'set': 'A', 'repeat': int(c in repeat)} for c in set_a] + \
           [{'case': c, 'set': 'B', 'repeat': 0} for c in set_b]
    rng.shuffle(rows)
    for i, row in enumerate(rows, 1):
        row['order'] = i
    return rows


def main():
    """Parse arguments, build the eligible pool, draw and write the CSV."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--n-a', type=int, default=20, help='set-A cases (2 legs each)')
    parser.add_argument('--n-b', type=int, default=20, help='set-B cases (2 legs each)')
    parser.add_argument('--n-repeat', type=int, default=5, help='set-A cases to annotate twice')
    parser.add_argument('--seed', type=int, default=20261116)
    parser.add_argument('-o', '--output', type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    excluded = read_reviewed(REVIEWED) | ids_named_in_repo(REPO) | verification_ids(NMDID)
    usable = usable_ids(NMDID)
    pool = [c for c in usable if c not in excluded]
    print(f'usable with CT + segmentation: {len(usable)}; excluded ids: {len(excluded)}; '
          f'eligible: {len(pool)}')

    rows = draw(pool, args.n_a, args.n_b, args.n_repeat, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=['order', 'case', 'set', 'repeat'])
        writer.writeheader()
        writer.writerows(rows)
    print(f'wrote {len(rows)} cases ({args.n_a} A, {args.n_b} B, {args.n_repeat} repeats) to {args.output}')


if __name__ == '__main__':
    main()
