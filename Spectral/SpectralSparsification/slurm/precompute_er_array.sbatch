#!/usr/bin/env bash
#SBATCH --job-name=edsparse-er
#SBATCH --array=0-18%1
#SBATCH --cpus-per-task=8
#SBATCH --time=7-00:00:00
#SBATCH --output=/people/dass304/dass304/Julia/SpectralSparsification/logs/slurm/%A_%a.out
#SBATCH --error=/people/dass304/dass304/Julia/SpectralSparsification/logs/slurm/%A_%a.err

set -euo pipefail

module load python/miniconda25.5.1
source /share/apps/python/miniconda25.5.1/etc/profile.d/conda.sh
conda activate py312

project_dir="/people/dass304/dass304/Julia/SpectralSparsification"
python_bin="/people/dass304/.conda/envs/py312/bin/python"
datasets=(
  cora citeseer pubmed amazon-computer amazon-photo
  coauthor-cs coauthor-physics wikics squirrel chameleon
  roman-empire amazon-ratings minesweeper questions reddit
  ogbn-products ogbn-arxiv ogbn-proteins pokec
)
dataset="${datasets[$SLURM_ARRAY_TASK_ID]}"
memory_gb="${ER_MEMORY_GB:-30}"

cd "$project_dir"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS=1
export JULIA_DEPOT_PATH="$project_dir/.julia_depot:/people/dass304/.julia"
export JULIA_PKG_PRECOMPILE_AUTO=0
export JULIA_NUM_PRECOMPILE_TASKS=1

printf '[%s] SLURM_START job=%s array=%s dataset=%s cpus=%s memory_gb=%s host=%s\n' \
  "$(date --iso-8601=seconds)" "$SLURM_JOB_ID" "$SLURM_ARRAY_TASK_ID" \
  "$dataset" "$SLURM_CPUS_PER_TASK" "$memory_gb" "$(hostname)"

# The array runs at most one element at a time, so the first task can safely
# build the project-local Julia cache and all later tasks reuse it.
if [[ ! -f "$project_dir/.julia_depot/.ready" ]]; then
  bash "$project_dir/scripts/prepare_julia.sh"
fi

"$python_bin" -u precompute_er.py \
  --dataset "$dataset" \
  --backend auto \
  --threads "$SLURM_CPUS_PER_TASK" \
  --memory-gb "$memory_gb" \
  --heartbeat-seconds 60 \
  --status-file "results/slurm_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.csv"

printf '[%s] SLURM_DONE dataset=%s\n' "$(date --iso-8601=seconds)" "$dataset"
