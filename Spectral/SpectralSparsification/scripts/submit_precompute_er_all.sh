#!/usr/bin/env bash

set -euo pipefail

project_dir="/people/dass304/dass304/Julia/SpectralSparsification"
memory="${ER_SLURM_MEMORY:-32G}"
dependency="${ER_SLURM_DEPENDENCY:-}"

mkdir -p "$project_dir/logs/slurm" "$project_dir/results"
bash "$project_dir/scripts/build_tgt.sh"

sbatch_options=(--parsable --mem="$memory")
if [[ -n "$dependency" ]]; then
  sbatch_options+=(--dependency="afterany:$dependency")
fi
job_id="$(sbatch "${sbatch_options[@]}" "$project_dir/slurm/precompute_er_array.sbatch")"
printf 'Submitted ER array job %s (19 datasets, at most one running at a time).\n' "$job_id"
if [[ -n "$dependency" ]]; then
  printf 'It will start after Slurm job %s finishes.\n' "$dependency"
fi
printf 'Queue: squeue -j %s\n' "$job_id"
printf 'Logs: tail -f %s/logs/slurm/%s_0.out\n' "$project_dir" "$job_id"
printf 'All logs: ls -lh %s/logs/slurm/%s_*\n' "$project_dir" "$job_id"
