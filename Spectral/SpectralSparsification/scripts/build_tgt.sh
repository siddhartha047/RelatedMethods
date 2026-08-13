#!/usr/bin/env bash

set -euo pipefail

project_dir="/people/dass304/dass304/Julia/SpectralSparsification"
build_dir="$project_dir/build/large_graph"

cmake -S "$project_dir/large_graph" -B "$build_dir" -DCMAKE_BUILD_TYPE=Release
cmake --build "$build_dir" --parallel "${SLURM_CPUS_PER_TASK:-${ER_THREADS:-8}}"

printf 'TGT+ executable: %s/tgt_effective_resistance\n' "$build_dir"
