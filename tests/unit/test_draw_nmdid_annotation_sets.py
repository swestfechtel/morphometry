"""Unit tests for the pure half of ``scripts/draw_nmdid_annotation_sets.py`` (no cluster data needed)."""
import importlib.util
import pathlib

_SCRIPT = pathlib.Path(__file__).resolve().parents[2] / 'scripts' / 'draw_nmdid_annotation_sets.py'
_spec = importlib.util.spec_from_file_location('script_draw_nmdid_annotation_sets', _SCRIPT)
draw_sets = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(draw_sets)

POOL = [str(100000 + i) for i in range(500)]


def test_sets_are_disjoint_sized_and_repeats_come_from_set_a():
    """A and B never share a case, have the requested sizes, and repeats are a subset of A."""
    rows = draw_sets.draw(POOL, 20, 20, 5, seed=1)
    a = {r['case'] for r in rows if r['set'] == 'A'}
    b = {r['case'] for r in rows if r['set'] == 'B'}
    repeat = {r['case'] for r in rows if r['repeat']}
    assert len(a) == 20 and len(b) == 20 and not a & b
    assert len(repeat) == 5 and repeat <= a
    assert [r['order'] for r in rows] == list(range(1, 41))


def test_draw_is_reproducible_and_seed_dependent():
    """Same seed, same draw; a different seed draws differently."""
    assert draw_sets.draw(POOL, 20, 20, 5, seed=7) == draw_sets.draw(POOL, 20, 20, 5, seed=7)
    assert draw_sets.draw(POOL, 20, 20, 5, seed=7) != draw_sets.draw(POOL, 20, 20, 5, seed=8)


def test_case_id_pattern_matches_inside_file_names_but_not_longer_numbers():
    """The id regex must match ``case-100010_seg`` (a ``\\b`` pattern does not) and skip 7-digit numbers."""
    assert draw_sets.CASE_ID.findall('case-100010_seg.nii.gz') == ['100010']
    assert draw_sets.CASE_ID.findall('1234567 and 199926') == ['199926']


def test_reviewed_list_holds_the_84_cases(tmp_path):
    """The committed reviewed-cases file parses to exactly 84 distinct ids."""
    reviewed = draw_sets.read_reviewed(draw_sets.REVIEWED)
    assert len(reviewed) == 84
