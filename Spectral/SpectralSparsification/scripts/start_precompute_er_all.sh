#!/usr/bin/env bash

set -euo pipefail

project_dir="/people/dass304/dass304/Julia/SpectralSparsification"
python_bin="${PY:-/people/dass304/.conda/envs/py312/bin/python}"
threads="${ER_THREADS:-8}"
log_dir="$project_dir/logs"
timestamp="$(date +%Y%m%d_%H%M%S)"
log_file="$log_dir/precompute_er_all_$timestamp.log"
pid_file="$log_dir/precompute_er_all.pid"

mkdir -p "$log_dir"

if [[ ! -x "$project_dir/build/large_graph/tgt_effective_resistance" ]]; then
  bash "$project_dir/scripts/build_tgt.sh"
fi

export JULIA_DEPOT_PATH="$project_dir/.julia_depot:/people/dass304/.julia"
export JULIA_PKG_PRECOMPILE_AUTO=0
export JULIA_NUM_PRECOMPILE_TASKS=1
export OPENBLAS_NUM_THREADS=1

if [[ -f "$pid_file" ]]; then
  previous_pid="$(<"$pid_file")"
  if [[ -n "$previous_pid" ]] && kill -0 "$previous_pid" 2>/dev/null; then
    printf 'ER precomputation is already running with PID %s.\n' "$previous_pid" >&2
    printf 'Monitor with: tail -f %s/latest.log\n' "$log_dir" >&2
    exit 1
  fi
fi

nohup "$python_bin" -u "$project_dir/precompute_er.py" \
  --all \
  --threads "$threads" \
  --progress-every 1 \
  --heartbeat-seconds 60 \
  --keep-going \
  "$@" \
  >"$log_file" 2>&1 </dev/null &

process_id=$!
printf '%s\n' "$process_id" >"$pid_file"
ln -sfn "$log_file" "$log_dir/latest.log"

printf 'Started ER precomputation with PID %s and %s available CPU threads.\n' "$process_id" "$threads"
printf 'Log: %s\n' "$log_file"
printf 'Status: %s/results/er_precompute_status.csv\n' "$project_dir"
printf 'Monitor with: tail -f %s/latest.log\n' "$log_dir"
