#!/usr/bin/env bash
# SLURM job array for scripts/subchondral_nako.py.
#
# Each array task takes an interleaved slice of the cohort (SLURM_ARRAY_TASK_ID /
# SLURM_ARRAY_TASK_COUNT are read by the script) and fans it out over the CPUs of
# its node, writing its own summary_shard_XXX.csv / rays_shard_XXX.csv.gz. Separate
# files mean tasks never contend, and both are flushed per case, so a failed or
# pre-empted task can simply be requeued -- re-running resumes from what it wrote.
#
#   sbatch scripts/subchondral_nako.sbatch
#   # then, once the array has finished:
#   python scripts/subchondral_nako.py -o results/nako_subchondral --merge
#
# Sizing, measured rather than guessed: ~11 CPU-seconds and ~0.9 GB peak per case
# (30407 cases, so ~95 CPU-hours and ~11 million rays). 8 tasks x 32 CPUs is 256
# cores -- about 25 minutes, and a quarter of the partition's 1152 cores, which
# leaves room on a shared lab partition. Raise --array for a bigger slice of it.
#
#SBATCH --job-name=nako-subchondral
#SBATCH --account=truhnlab
#SBATCH --partition=truhnlab
#SBATCH --array=0-7
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/nako-subchondral-%A_%a.out
#SBATCH --error=logs/nako-subchondral-%A_%a.out

set -euo pipefail

REPO=/home/sw521914/Work/morphometry
DATA=/hpcwork/p0021834/workspace_simon/nako
OUT="${REPO}/results/nako_subchondral"

mkdir -p "${OUT}" "${REPO}/logs"

# Each case is already one worker process; keep the numeric libraries single-threaded
# so the workers do not oversubscribe the node.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

"${REPO}/.venv/bin/python" "${REPO}/scripts/subchondral_nako.py" \
    --input "${DATA}" \
    --output "${OUT}"
