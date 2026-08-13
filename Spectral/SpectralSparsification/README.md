# SpectralSparsification

This project precomputes one effective-resistance (ER) value for every edge in
the 19 EDSparse datasets.  It does not modify EDSparse's raw or processed data.
Artifacts are added under each dataset's `spectral_sparsification/` directory.

The current task is ER precomputation only.  The cached canonical ER can later
be consumed by the spectral-sparsification stage in this project.

## Corrected two-backend implementation

The failed batch sent every graph to one Julia process.  All 19 Julia children
in that run exited with signal 11, and the same in-memory path was being used
for both Cora and graphs with tens of millions of canonical edges.

The corrected `--backend auto` dispatcher uses:

- **Laplacians.jl** for graphs with at most 200,000 nodes and 2,000,000
  canonical edges.  It implements the Johnson--Lindenstrauss projection plus
  approximate-Cholesky Laplacian solves derived from
  `SpectralSampling/compute_V_julia.jl`.
- **TGT+ C++/OpenMP** for larger unweighted graphs.  It uses compact CSR,
  matrix-free ARPACK eigenvectors, streaming binary input, deterministic
  random walks, bounded per-thread workspaces, and restartable source blocks.
  It is based on Algorithms 1, 3, and 4 of Zhang et al.,
  [KDD 2023](https://arxiv.org/abs/2305.16086), but does not use the upstream
  repository's NetworkX, ASCII eigenvectors, `vector<vector<...>>`, or
  per-edge maps.

The TGT+ backend computes the largest connected component with TGT+.  Any
additional edge-bearing components are computed exactly and independently.
This matters for `ogbn-products`, whose giant component contains 2,385,902
nodes while its other 4,236 components contain only 52,709 edges total.
Isolated nodes need no edge values.

See [DESIGN.md](DESIGN.md) for formulas, assumptions, memory layout, and
validation details.

## Paste this to submit all 19 datasets on Slurm

This requests 8 CPUs and 32 GiB per task.  The array limit `%1` runs one
dataset at a time, avoiding concurrent copies of the large OGB/Reddit graphs.
Within each dataset all 8 CPUs are used.  `sbatch` returns immediately, so the
work runs in the background on Slurm.

```bash
module load python/miniconda25.5.1
source /share/apps/python/miniconda25.5.1/etc/profile.d/conda.sh
conda activate py312

cd /people/dass304/dass304/Julia/SpectralSparsification
export ER_THREADS=8
export ER_SLURM_MEMORY=32G
export ER_MEMORY_GB=30
bash scripts/submit_precompute_er_all.sh
```

If an older array is still running, queue a cache-aware retry behind it instead
of running both arrays concurrently:

```bash
export ER_SLURM_DEPENDENCY=OLD_ARRAY_JOB_ID
bash scripts/submit_precompute_er_all.sh
```

The submit command prints the Slurm array job ID.  If it prints `123456`, use:

```bash
squeue -j 123456
tail -F logs/slurm/123456_*.out logs/slurm/123456_*.err
```

To see a compact result for each finished array element:

```bash
for file in results/slurm_123456_*.csv; do echo "--- $file"; cat "$file"; done
```

The logs contain `HEARTBEAT` every 60 seconds plus `CSR_READY`,
`MEMORY_ESTIMATE`, `EIGEN_*`, `TAU_PROGRESS`, `TGT_PROGRESS`,
`DATASET_DONE`, and `DATASET_FAILED`.  The C++ messages include current RSS.
On the first submission, array element 0 also builds the project-local Julia
compiled cache on the compute node; this one-time preparation is therefore in
the background and later array elements reuse it.

If the cluster requires a partition or account, add those normal site options
to the `sbatch` line in `scripts/submit_precompute_er_all.sh`.

## Why 32 GiB is safe by default

Before its large allocations, TGT+ prints a conservative peak estimate that
includes CSR, the ARPACK `n × ncv` basis, `omega` node eigenvectors, both
oriented result arrays, and all 8 traversal workspaces.  `--memory-gb 30`
refuses to start when that estimate would not fit inside the 32 GiB Slurm
request.  Increase both `ER_SLURM_MEMORY` and `ER_MEMORY_GB` together if you
change `--omega` or the thread count.

The four default TGT+ datasets are Reddit, OGBN-Products, OGBN-Proteins, and
Pokec.  Dataset jobs are separate Slurm array elements, so failure or timeout
of one does not discard completed work from the others.

## Restart behavior

Re-submit the same command after a timeout or node failure.

- A complete directed artifact with matching graph and edge-order SHA-256
  fingerprints is a cache hit.
- `input_graph.tgtbin` is reused after its header and fingerprints validate.
- ARPACK preprocessing is reused for the same graph, `omega`, and tolerance.
- TGT+ commits completed source blocks to `tgt_work/oriented_er.bin`; a retry
  resumes at the last committed block.
- Temporary final NPZ files are published with an atomic rename.

Do not use `--force-er` for an ordinary retry; it intentionally replaces the
source checkpoint.

## Run one dataset interactively

First build the standalone backend and prepare a private Julia compiled cache:

```bash
module load python/miniconda25.5.1
source /share/apps/python/miniconda25.5.1/etc/profile.d/conda.sh
conda activate py312
cd /people/dass304/dass304/Julia/SpectralSparsification

bash scripts/build_tgt.sh
bash scripts/prepare_julia.sh
```

Then run, for example, Cora or OGBN-Products:

```bash
PY=/people/dass304/.conda/envs/py312/bin/python
"$PY" -u precompute_er.py --dataset cora --threads 8 --memory-gb 30
"$PY" -u precompute_er.py --dataset ogbn-products --threads 8 --memory-gb 30
```

Both backends receive all 8 requested CPUs.  Laplacians.jl's projection and
edge-alignment loops are threaded (the approximate-Cholesky solver has serial
sections); TGT+ parallelizes source vertices with OpenMP.  The corrected Julia
assembly uses bounded static thread slots, and only graphs below the dispatcher
limits enter that path.  To test the standalone backend on a small graph,
explicitly add `--backend tgt --omega 16`.

## Output and edge alignment

For each dataset:

```text
<dataset>/spectral_sparsification/
├── input_graph.npz                         # existing canonical export, read-only input
├── input_graph.tgtbin                      # streaming TGT cache when needed
├── effective_resistance.npz                # one value per canonical edge
├── effective_resistance.toml
├── effective_resistance_directed.npz       # one value per original PyG edge
├── effective_resistance_directed.toml
└── tgt_work/
    ├── eigen_omega_128.bin
    └── oriented_er.bin
```

`effective_resistance_directed.npz["resistance"]` has exactly the same length
and order as EDSparse's `data.edge_index`.  Reciprocal or duplicate entries are
retained and receive their canonical undirected edge's ER.  A self-loop, if
present, receives 0.  Thus Cora has exactly 10,556 directed ER values even
though it has 5,278 canonical undirected edges.

Load it in EDSparse with:

```python
import sys

sys.path.insert(0, "/people/dass304/dass304/Julia/SpectralSparsification/python")
from edsparse_adapter import (
    attach_directed_effective_resistance,
    effective_resistance_artifact,
)

data = attach_directed_effective_resistance(
    data,
    effective_resistance_artifact(data_root, dataset),
)
```

This adapter checks both the number of values and a SHA-256 fingerprint of
`num_nodes` plus every `edge_index` column in order.  It therefore rejects an
array with the right length but the wrong edge ordering.  If a pipeline has
already normalized, deduplicated, or reordered its topology, use the canonical
artifact with `align_effective_resistance(data.edge_index, canonical_path)`;
that function aligns values by undirected endpoint identity.

Scaffold's `EffectiveResistanceSparsifier` uses the canonical artifact rather
than the directed artifact.  It validates the complete set of undirected node
pairs before sampling.  This matters for datasets such as Reddit: EDSparse has
114,615,892 directed entries while Scaffold represents the same topology as
57,307,946 unique undirected edges.

Recheck all 19 artifacts against both live loaders (read-only):

```bash
PY=/people/dass304/.conda/envs/py312/bin/python
"$PY" -u tools/verify_edsparse_alignment.py
```

Inspect a large artifact without loading the full vector:

```bash
PY=/people/dass304/.conda/envs/py312/bin/python
"$PY" tools/inspect_er.py /path/to/effective_resistance_directed.npz --limit 10
```

## Manual all-dataset background run without Slurm

Slurm is preferred for the large graphs.  On an allocated compute node, this
launcher runs all datasets sequentially under `nohup`:

```bash
export PY=/people/dass304/.conda/envs/py312/bin/python
export ER_THREADS=8
bash scripts/start_precompute_er_all.sh --memory-gb 30
tail -F logs/latest.log
```

## Accuracy parameters

Large-graph defaults are `epsilon=0.05`, `delta=0.01`, `omega=128`, and
`gamma=10`.  All are included in checkpoint/artifact metadata.  Lower
`epsilon` is more accurate and can require substantially more random walks.

The implementation fixes two unsafe choices in the public research code's
CalTau routine: it does not reset a non-positive `epsilon - delta_t`, and it
returns the odd traversal index specified by Algorithm 1.  On Karate with
`omega=16` and `epsilon=0.05`, the standalone output had mean absolute error
0.00803 and maximum absolute error 0.02544 against the dense pseudoinverse;
all 78 edges were within the requested 0.05 absolute error.

On AESC's supplied Facebook graph (4,039 nodes, 88,234 canonical edges), the
rewritten backend was also compared to the repository's `gt_sec.txt` values
with `omega=128`, `epsilon=0.05`, and `delta=1/n`:

| Metric | Result |
|---|---:|
| Mean absolute difference | 0.0004983 |
| Maximum absolute difference | 0.0320480 |
| Edges within 0.05 | 100% |
| Pearson correlation | 0.999691 |
| Spearman correlation | 0.999381 |

This reference run completed its matrix-free eigensolve in 0.74 seconds and
the TGT+ source phase in 1.10 seconds with 8 threads on the current node.

## Tests

```bash
PY=/people/dass304/.conda/envs/py312/bin/python
"$PY" -m pytest -q tests
bash scripts/build_tgt.sh
```
