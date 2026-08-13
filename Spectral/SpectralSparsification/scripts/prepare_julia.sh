#!/usr/bin/env bash

set -euo pipefail

project_dir="/people/dass304/dass304/Julia/SpectralSparsification"
private_depot="$project_dir/.julia_depot"
shared_depot="/people/dass304/.julia"

mkdir -p "$private_depot"
export JULIA_DEPOT_PATH="$private_depot:$shared_depot"
export JULIA_PKG_PRECOMPILE_AUTO=0
export JULIA_NUM_PRECOMPILE_TASKS=1
export OPENBLAS_NUM_THREADS=1

julia --startup-file=no --history-file=no --project="$project_dir" --threads=1 \
  -e 'using Pkg; Pkg.instantiate(); Pkg.precompile(); using SpectralSparsification; println("Julia/Laplacians environment ready")'

touch "$private_depot/.ready"
