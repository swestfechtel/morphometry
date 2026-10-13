#!/usr/bin/env bash
# SLURM job array for scripts/process_nako_large.py: one task per NaKo chunk (0-19).
#
# Each task measures every case of <DATA>/chunk_<N>_segmentations (~1520 isotropic hip
# MRI segmentations) on the CPUs of its node and writes results_chunk_<N>.csv plus
# results_chunk_<N>.meta.json (git commit, date, arguments) into OUT. OUT is a new
# directory per commit (rerun_<short sha>), so earlier result tables stay intact. Rows
# are flushed per case to results_chunk_<N>.partial.csv, so a task that is killed
# (time limit, OOM, a crashed worker) can simply be resubmitted and resumes where it
# stopped; a finished chunk CSV is refused rather than overwritten (pass --overwrite,
# or delete it, to redo that chunk).
#
# Failure handling, all recorded in the meta file and the task log:
#  - a case that fails to load is retried once at the end of the task, then written as
#    all-NaN rows; if more than 1% of the chunk fails to load (a broken mount, say), the
#    task exits nonzero WITHOUT finalising -- fix the cause and resubmit, it resumes;
#  - a worker that dies hard (OOM, segfault) fails the task at once; the cases in flight
#    are saved to results_chunk_<N>.suspects.json. Resubmit: suspects run one at a time
#    first, and one that kills its worker even alone is written as all-NaN rows.
#
# Every task reads the code and the commit hash from the checkout when it STARTS, so do
# not pull, switch branches or edit files in REPO until the whole array has finished.
# A dirty checkout is refused outright: the results must be traceable to a commit.
#
# Sizing: a case costs ~40 CPU-seconds and a worker peaks below 1 GB, so cores, not
# memory, set the pace: 1520 cases x 40 s / 32 workers ~ 32 min per task. The 4 h limit
# is a wide margin for slow nodes and hpcwork I/O.
#
#   cd ~/Work/morphometry && mkdir -p logs
#   sbatch scripts/process_nako_large.sbatch
#   # resubmit a killed/failed chunk (e.g. 7); it resumes from its partial file:
#   sbatch --array=7 scripts/process_nako_large.sbatch
#   # then, once all 20 chunks have finished (OUT as printed in the task logs):
#   .venv/bin/python scripts/process_nako_large.py --output-dir ~/Data/nako/rerun_<sha> --merge
#
#SBATCH --job-name=nako-morph
#SBATCH --array=0-19
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --partition truhnlab
#SBATCH --account truhnlab
#SBATCH --time=04:00:00
#SBATCH --output=logs/nako-morph-%A_%a.out
#SBATCH --error=logs/nako-morph-%A_%a.out

set -euo pipefail

REPO=/home/sw521914/Work/morphometry
DATA=/hpcwork/p0021834/workspace_simon/nako
if [ -n "$(git -C "${REPO}" status --porcelain --untracked-files=no)" ]; then
    echo "refusing to run: ${REPO} has uncommitted changes" >&2
    exit 1
fi
SHA=$(git -C "${REPO}" rev-parse --short HEAD)
OUT="${HOME}/Data/nako/rerun_${SHA}"

mkdir -p "${OUT}" "${REPO}/logs"
echo "chunk ${SLURM_ARRAY_TASK_ID} at commit ${SHA} -> ${OUT}"

# Each case is already one worker process; keep the numeric libraries single-threaded
# so the workers do not oversubscribe the node.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

"${REPO}/.venv/bin/python" "${REPO}/scripts/process_nako_large.py" \
    --chunk "${SLURM_ARRAY_TASK_ID}" \
    --input-root "${DATA}" \
    --output-dir "${OUT}"
