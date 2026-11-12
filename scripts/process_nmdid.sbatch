#!/usr/bin/env bash
# SLURM job array for scripts/process_nmdid.py.
#
# Each array task takes an interleaved slice of the cohort (SLURM_ARRAY_TASK_ID /
# SLURM_ARRAY_TASK_COUNT are read by the script) and fans it out over the CPUs of
# its node. Every task writes its own results/shard_XXX.csv, so tasks never
# contend for a file and a failed or pre-empted task can simply be requeued --
# re-running resumes from the rows it already flushed.
#
# Sizing: a worker holds the float64 volume and several working copies and peaks at
# about 4 GB, so memory -- not cores -- caps the workers per node; keep --mem at
# roughly 5 GB per --cpus-per-task. One case costs ~4 CPU-minutes, so the 1474-case
# cohort is ~100 CPU-hours: 16 tasks x 16 workers finish it in well under an hour.
#
#   sbatch scripts/process_nmdid.sbatch
#   # then, once the array has finished:
#   python scripts/process_nmdid.py -o results/nmdid_<sha> --merge
#
#SBATCH --job-name=nmdid-morph
#SBATCH --array=0-15
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=04:00:00
#SBATCH --output=logs/nmdid-morph-%A_%a.out
#SBATCH --error=logs/nmdid-morph-%A_%a.out

set -euo pipefail

REPO=/home/sw521914/Work/morphometry
DATA=/hpcwork/p0021834/workspace_simon/nmdid

# The table must be traceable to the code that produced it: refuse a dirty checkout and key the
# output directory by commit, so rows from different commits can never land in -- or be resumed
# into -- the same shards. (The checkout must not change until the whole array has finished.)
if [ -n "$(git -C "${REPO}" status --porcelain --untracked-files=no)" ]; then
    echo "refusing to run on a dirty checkout of ${REPO}; commit or stash first" >&2
    exit 1
fi
SHA=$(git -C "${REPO}" rev-parse --short HEAD)
OUT="${REPO}/results/nmdid_${SHA}"

mkdir -p "${OUT}" "${REPO}/logs"

# Each case is already one worker process; keep the numeric libraries single-threaded
# so the workers do not oversubscribe the node.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

"${REPO}/.venv/bin/python" "${REPO}/scripts/process_nmdid.py" \
    --input "${DATA}/segmentations" \
    --output "${OUT}" \
    --cohort-status "${DATA}/cohort_status.csv" \
    --verdict usable review
