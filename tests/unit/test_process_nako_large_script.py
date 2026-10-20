"""Unit tests for the CLI plumbing of ``scripts/process_nako_large.py`` (no image data needed).

The measurements themselves are covered by the characterization tests; these pin what the cohort
run depends on around them: the per-chunk CSV keeps the exact layout of the original
``results_chunk_*.csv`` files (downstream concatenates them into ``results.csv``), an existing
result is never overwritten silently, the worker count follows the SLURM allocation, every result
is traceable to a commit, and the chunk files merge into one table.
"""
import importlib.util
import json
import pathlib

import numpy as np
import pandas as pd
import pytest

_SCRIPT = pathlib.Path(__file__).resolve().parents[2] / 'scripts' / 'process_nako_large.py'
_spec = importlib.util.spec_from_file_location('process_nako_large', _SCRIPT)
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)

#: The header of the original results_chunk_*.csv / results.csv files, with the ``qc_flags``
#: column (the Murphy femoral-neck QC flags) appended after them -- see
#: ``tests/unit/test_murphy_qc_scripts.py`` for what goes into it.
HEADER = ('Patient,Side,CCD,AT_murphy,AA_anterior,AA_posterior,AAV,CE,Offset,Cartilage_thickness,'
          'Cartilage_thickness (mean),Cartilage_thickness (std),Cartilage_thickness (min),'
          'Cartilage_thickness (max),qc_flags')


def _result(patient: str, base: float) -> dict:
    """A fake per-patient result as ``f`` returns it; image-left values end in .1x, image-right in .2x.

    :param patient: Patient id.
    :param base: Offset added to every value so patients are distinguishable.
    :return: The result dict.
    """
    return {'patient': patient, 'ccd_left': base + 0.11, 'ccd_right': base + 0.21,
            'fat_left': base + 0.12, 'fat_right': base + 0.22,
            'aa_left': (base + 0.13, base + 0.14), 'aa_right': (base + 0.23, base + 0.24),
            'aav_left': base + 0.15, 'aav_right': base + 0.25, 'cea_left': base + 0.16, 'cea_right': base + 0.26,
            'offset_left': base + 0.17, 'offset_right': base + 0.27,
            'cartilage_thickness_left': (1.11, 1.12, 1.13, 1.14),
            'cartilage_thickness_right': (2.11, 2.12, 2.13, np.nan)}


def test_table_keeps_the_original_layout_and_side_mapping(tmp_path):
    """Header, row order and the image-left = patient-right mapping match the original chunk files."""
    df = mod.build_table([_result('p1', 10), _result('p2', 20)], ['p1', 'p2'])
    out = tmp_path / 'chunk.csv'
    df.to_csv(out)
    lines = out.read_text().splitlines()
    assert lines[0] == HEADER
    assert [line.split(',')[:2] for line in lines[1:]] == [['p1', 'right'], ['p1', 'left'], ['p2', 'right'],
                                                            ['p2', 'left']]
    table = pd.read_csv(out, index_col=[0, 1])
    # image-left half (``*_left``) is the patient's right side, rounded to one decimal
    assert table.loc[('p1', 'right'), 'CCD'] == 10.1 and table.loc[('p1', 'left'), 'CCD'] == 10.2
    assert table.loc[('p2', 'right'), 'AA_posterior'] == 20.1 and table.loc[('p2', 'left'), 'AA_anterior'] == 20.2
    assert table.loc[('p1', 'left'), 'Cartilage_thickness (mean)'] == 2.1
    assert np.isnan(table.loc[('p1', 'left'), 'Cartilage_thickness (max)'])
    assert table['Cartilage_thickness'].isna().all()  # kept empty, as in the original files


def test_existing_output_is_refused_without_overwrite(tmp_path):
    """An existing chunk CSV is an error unless --overwrite is given."""
    out = tmp_path / 'results_chunk_3.csv'
    mod.check_writable(out, overwrite=False)  # absent: fine
    out.write_text('old')
    with pytest.raises(FileExistsError):
        mod.check_writable(out, overwrite=False)
    mod.check_writable(out, overwrite=True)


def test_default_workers_follow_the_slurm_allocation(monkeypatch):
    """SLURM_CPUS_PER_TASK wins over the visible CPUs."""
    monkeypatch.setenv('SLURM_CPUS_PER_TASK', '7')
    assert mod.default_workers() == 7
    monkeypatch.delenv('SLURM_CPUS_PER_TASK')
    assert mod.default_workers() >= 1


def test_meta_records_the_commit_and_arguments(tmp_path):
    """The meta file names the commit, the arguments and the number of cases."""
    meta = tmp_path / 'results_chunk_0.meta.json'
    mod.write_meta(meta, {'chunk': 0, 'input_root': '/x'}, n_cases=5)
    data = json.loads(meta.read_text())
    assert len(data['commit']) == 40 and isinstance(data['dirty'], bool)
    assert data['args'] == {'chunk': 0, 'input_root': '/x'} and data['n_cases'] == 5
    assert 'date' in data


def test_merge_concatenates_the_chunks_in_chunk_order(tmp_path):
    """--merge writes one table with one header, chunks in numeric order (chunk_10 after chunk_2)."""
    for chunk in range(11):
        mod.build_table([_result(f'p{chunk}', chunk)], [f'p{chunk}']).to_csv(tmp_path / f'results_chunk_{chunk}.csv')
    merged = tmp_path / 'results.csv'
    mod.merge_chunks(tmp_path, merged, overwrite=False, expected_chunks=11)
    lines = merged.read_text().splitlines()
    assert lines[0] == HEADER and len(lines) == 1 + 2 * 11
    assert [line.split(',')[0] for line in lines[1::2]] == [f'p{c}' for c in range(11)]
    with pytest.raises(FileExistsError):
        mod.merge_chunks(tmp_path, merged, overwrite=False, expected_chunks=11)

# --- robustness: load failures, incremental rows, resume, broken pool, merge checks ------------------

import os
import sys
from concurrent.futures.process import BrokenProcessPool

sys.modules.setdefault('process_nako_large', mod)  # so worker processes can unpickle mod's functions


def _fake_worker(patient):
    """A picklable stand-in for ``f``: a deterministic result per patient, no image I/O.

    :param patient: Path of the case file.
    :return: A result dict as ``f`` returns it.
    """
    name = patient.name.split('.')[0]
    return _result(name, float(name.lstrip('p')))


def _dying_worker(patient):
    """A worker whose process dies hard (as on an OOM kill or a VTK segfault) on patient ``p2``.

    :param patient: Path of the case file.
    :return: A result dict for every other patient.
    """
    if patient.name.startswith('p2.'):
        os._exit(1)
    return _fake_worker(patient)


def _cases(tmp_path, names):
    """Create empty case files ``<name>.nii.gz`` in a chunk directory.

    :param tmp_path: pytest temporary directory.
    :param names: Patient ids.
    :return: The case paths, sorted.
    """
    d = tmp_path / 'in'
    d.mkdir(exist_ok=True)
    for n in names:
        (d / f'{n}.nii.gz').write_bytes(b'')
    return sorted(d.glob('*.nii.gz'))


def test_an_unreadable_case_becomes_all_nan_rows(tmp_path, capsys):
    """A case that cannot be loaded yields an all-NaN result named after the patient, not an exception."""
    broken = tmp_path / 'zzz_broken.nii.gz'
    broken.write_bytes(b'not a nifti')
    result = mod.f(broken)
    assert result['patient'] == 'zzz_broken'
    row = mod.build_table([result], ['zzz_broken'])
    # every measurement is NaN; qc_flags is an (empty) string, not NaN, so it is checked apart
    assert row.drop(columns=['Cartilage_thickness', 'qc_flags']).isna().all().all() and len(row) == 2
    assert (row['qc_flags'] == '').all()
    assert 'zzz_broken' in capsys.readouterr().out


def test_run_chunk_writes_the_sorted_table_and_meta(tmp_path):
    """The finished chunk equals build_table over all patients in sorted order; the partial file is gone."""
    cases = _cases(tmp_path, ['p3', 'p1', 'p2'])
    out = tmp_path / 'out'
    mod.run_chunk(cases, out, chunk=4, workers=2, overwrite=False, args={'chunk': 4}, worker=_fake_worker)
    expected = tmp_path / 'expected.csv'
    mod.build_table([_fake_worker(c) for c in cases], ['p1', 'p2', 'p3']).to_csv(expected)
    assert (out / 'results_chunk_4.csv').read_text() == expected.read_text()
    assert not (out / 'results_chunk_4.partial.csv').exists()
    assert json.loads((out / 'results_chunk_4.meta.json').read_text())['n_cases'] == 3


def test_run_chunk_resumes_from_the_partial_file(tmp_path):
    """Patients already in the partial file are not recomputed (their rows are kept as written)."""
    cases = _cases(tmp_path, ['p1', 'p2'])
    out = tmp_path / 'out'
    out.mkdir()
    mod.build_table([_result('p1', 99)], ['p1']).to_csv(out / 'results_chunk_0.partial.csv')
    mod.run_chunk(cases, out, chunk=0, workers=1, overwrite=False, args={}, worker=_fake_worker)
    table = pd.read_csv(out / 'results_chunk_0.csv', index_col=[0, 1])
    assert table.loc[('p1', 'right'), 'CCD'] == 99.1  # kept, not recomputed (would be 1.1)
    assert table.loc[('p2', 'right'), 'CCD'] == 2.1


def test_a_finished_chunk_is_refused_and_overwrite_starts_fresh(tmp_path):
    """An existing chunk CSV is refused; --overwrite discards it and any partial file and recomputes."""
    cases = _cases(tmp_path, ['p1'])
    out = tmp_path / 'out'
    mod.run_chunk(cases, out, chunk=0, workers=1, overwrite=False, args={}, worker=_fake_worker)
    with pytest.raises(FileExistsError):
        mod.run_chunk(cases, out, chunk=0, workers=1, overwrite=False, args={}, worker=_fake_worker)
    mod.build_table([_result('p1', 99)], ['p1']).to_csv(out / 'results_chunk_0.partial.csv')
    mod.run_chunk(cases, out, chunk=0, workers=1, overwrite=True, args={}, worker=_fake_worker)
    assert pd.read_csv(out / 'results_chunk_0.csv', index_col=[0, 1]).loc[('p1', 'right'), 'CCD'] == 1.1


def test_a_dead_worker_fails_fast_and_keeps_finished_rows(tmp_path):
    """A hard worker death raises instead of hanging; rows flushed before it survive for the resume."""
    cases = _cases(tmp_path, ['p1', 'p2', 'p3'])
    out = tmp_path / 'out'
    with pytest.raises(BrokenProcessPool):
        mod.run_chunk(cases, out, chunk=0, workers=1, overwrite=False, args={}, worker=_dying_worker)
    assert not (out / 'results_chunk_0.csv').exists()
    partial = pd.read_csv(out / 'results_chunk_0.partial.csv', index_col=[0, 1])
    assert set(partial.index.get_level_values(0)) == {'p1'}  # one worker: p1 finished before p2 killed it


def _chunk(out, chunk, patients):
    """Write a finished chunk CSV for the given patients.

    :param out: Output directory.
    :param chunk: Chunk number.
    :param patients: Patient ids.
    """
    mod.build_table([_result(p, chunk) for p in patients], patients).to_csv(out / f'results_chunk_{chunk}.csv')


def test_merge_refuses_missing_chunks(tmp_path):
    """Merging fewer chunks than expected errors and names the missing ones."""
    _chunk(tmp_path, 0, ['a'])
    _chunk(tmp_path, 2, ['b'])
    with pytest.raises(FileNotFoundError, match=r'\[1, 3\]'):
        mod.merge_chunks(tmp_path, tmp_path / 'results.csv', overwrite=False, expected_chunks=4)
    assert not (tmp_path / 'results.csv').exists()


def test_merge_refuses_unfinished_chunks(tmp_path):
    """A chunk that has only a partial file is reported as unfinished."""
    _chunk(tmp_path, 0, ['a'])
    mod.build_table([_result('b', 1)], ['b']).to_csv(tmp_path / 'results_chunk_1.partial.csv')
    with pytest.raises(FileNotFoundError, match='unfinished.*\\[1\\]'):
        mod.merge_chunks(tmp_path, tmp_path / 'results.csv', overwrite=False, expected_chunks=2)


def test_merge_refuses_duplicate_rows(tmp_path):
    """The same (Patient, Side) in two chunks is an error."""
    _chunk(tmp_path, 0, ['a'])
    _chunk(tmp_path, 1, ['a'])
    with pytest.raises(ValueError, match='duplicate'):
        mod.merge_chunks(tmp_path, tmp_path / 'results.csv', overwrite=False, expected_chunks=2)


def test_a_case_torn_by_a_kill_is_recomputed(tmp_path):
    """A partial file cut mid-row (job killed while writing) drops the torn case and resumes cleanly."""
    cases = _cases(tmp_path, ['p1', 'p2'])
    out = tmp_path / 'out'
    out.mkdir()
    partial = out / 'results_chunk_0.partial.csv'
    mod.build_table([_result('p1', 99), _result('p2', 98)], ['p1', 'p2']).to_csv(partial)
    text = partial.read_text()
    partial.write_text(text[:text.rfind('p2,left') + 9])  # p2's second row torn off mid-line
    mod.run_chunk(cases, out, chunk=0, workers=1, overwrite=False, args={}, worker=_fake_worker)
    table = pd.read_csv(out / 'results_chunk_0.csv', index_col=[0, 1])
    assert table.loc[('p1', 'right'), 'CCD'] == 99.1 and table.loc[('p2', 'left'), 'CCD'] == 2.2
    assert len(table) == 4


# --- poison cases, load-failure accounting, empty partials, merge range ------------------------------

def _poison_worker(patient):
    """A worker whose process dies on ``p2`` every time (a deterministic OOM / segfault).

    :param patient: Path of the case file.
    :return: A result dict for every other patient.
    """
    if patient.name.startswith('p2.'):
        os._exit(1)
    return _fake_worker(patient)


def _flaky_load_worker(patient):
    """A worker that fails to load ``p2`` the first time it is asked (a transient I/O hiccup).

    :param patient: Path of the case file.
    :return: A load-failure result on the first call for p2, a normal result otherwise.
    """
    marker = patient.parent / 'p2.tried'
    if patient.name.startswith('p2.') and not marker.exists():
        marker.write_text('')
        return mod._nan_result('p2', load_failed=True)
    return _fake_worker(patient)


def _unloadable_worker(patient):
    """A worker that can load only ``p1`` (as on a broken mount).

    :param patient: Path of the case file.
    :return: A normal result for p1, a load failure for everything else.
    """
    name = patient.name.split('.')[0]
    return _fake_worker(patient) if name == 'p1' else mod._nan_result(name, load_failed=True)


def test_a_poison_case_is_isolated_and_written_as_nan(tmp_path, capsys):
    """A case that kills its worker every time is recorded as a suspect, run alone on the next run and,
    when it breaks alone, written as all-NaN rows and listed in the meta file."""
    cases = _cases(tmp_path, ['p1', 'p2', 'p3'])
    out = tmp_path / 'out'
    with pytest.raises(BrokenProcessPool):
        mod.run_chunk(cases, out, chunk=0, workers=2, overwrite=False, args={}, worker=_poison_worker)
    suspects = json.loads((out / 'results_chunk_0.suspects.json').read_text())
    assert 'p2' in suspects

    mod.run_chunk(cases, out, chunk=0, workers=2, overwrite=False, args={}, worker=_poison_worker)
    table = pd.read_csv(out / 'results_chunk_0.csv', index_col=[0, 1])
    assert table.loc['p2'].drop(columns='Cartilage_thickness').isna().all().all()
    assert table.loc[('p1', 'right'), 'CCD'] == 1.1 and table.loc[('p3', 'left'), 'CCD'] == 3.2
    meta = json.loads((out / 'results_chunk_0.meta.json').read_text())
    assert meta['crashed'] == ['p2'] and meta['load_failed'] == []
    assert not (out / 'results_chunk_0.suspects.json').exists()
    assert 'p2' in capsys.readouterr().out


def test_a_load_failure_is_retried_before_finalising(tmp_path):
    """A case that fails to load is not flushed but retried once at the end of the run."""
    cases = _cases(tmp_path, ['p1', 'p2'])
    out = tmp_path / 'out'
    mod.run_chunk(cases, out, chunk=0, workers=1, overwrite=False, args={}, worker=_flaky_load_worker)
    table = pd.read_csv(out / 'results_chunk_0.csv', index_col=[0, 1])
    assert table.loc[('p2', 'right'), 'CCD'] == 2.1
    assert json.loads((out / 'results_chunk_0.meta.json').read_text())['load_failed'] == []


def test_persistent_load_failures_are_recorded_in_the_meta_file(tmp_path):
    """Below the threshold, a case that still fails to load becomes NaN rows and is listed in the meta."""
    cases = _cases(tmp_path, ['p1', 'p2'])
    out = tmp_path / 'out'
    mod.run_chunk(cases, out, chunk=0, workers=1, overwrite=False, args={}, worker=_unloadable_worker,
                  max_load_fail_frac=0.5)
    meta = json.loads((out / 'results_chunk_0.meta.json').read_text())
    assert meta['load_failed'] == ['p2'] and meta['n_load_failed'] == 1
    assert pd.read_csv(out / 'results_chunk_0.csv', index_col=[0, 1]).loc['p2', 'CCD'].isna().all()


def test_too_many_load_failures_abort_without_finalising(tmp_path):
    """Above the threshold (a broken mount), the run fails, keeps what loaded and writes no chunk table."""
    cases = _cases(tmp_path, ['p1', 'p2', 'p3'])
    out = tmp_path / 'out'
    with pytest.raises(RuntimeError, match='failed to load'):
        mod.run_chunk(cases, out, chunk=0, workers=1, overwrite=False, args={}, worker=_unloadable_worker,
                      max_load_fail_frac=0.01)
    assert not (out / 'results_chunk_0.csv').exists()
    partial = pd.read_csv(out / 'results_chunk_0.partial.csv', index_col=[0, 1])
    assert set(partial.index.get_level_values(0)) == {'p1'}  # load failures are never flushed


@pytest.mark.parametrize('content', ['', 'Patient,Side,CCD,AT_mur'], ids=['empty', 'torn_header'])
def test_an_empty_or_torn_partial_starts_afresh(tmp_path, content):
    """A partial file with no complete row (0 bytes, or killed while writing the header) is discarded."""
    cases = _cases(tmp_path, ['p1'])
    out = tmp_path / 'out'
    out.mkdir()
    (out / 'results_chunk_0.partial.csv').write_text(content)
    mod.run_chunk(cases, out, chunk=0, workers=1, overwrite=False, args={}, worker=_fake_worker)
    assert pd.read_csv(out / 'results_chunk_0.csv', index_col=[0, 1]).loc[('p1', 'right'), 'CCD'] == 1.1


def test_merge_refuses_chunks_outside_the_expected_range(tmp_path):
    """A chunk numbered >= expected_chunks is an error, not silently merged."""
    for chunk in range(3):
        _chunk(tmp_path, chunk, [f'c{chunk}'])
    with pytest.raises(ValueError, match=r'\[2\]'):
        mod.merge_chunks(tmp_path, tmp_path / 'results.csv', overwrite=False, expected_chunks=2)
