# Effective-resistance precomputation design

## Why the previous batch failed

The old driver sent every graph to the same Julia/Laplacians.jl process.  The
batch log shows that all 19 Julia children exited with signal 11 before they
produced an ER artifact.  It also attempted to use the same in-memory path for
Cora and for graphs with millions of nodes and tens of millions of canonical
edges.  That is not a safe Slurm design.

The replacement has two explicit backends:

- `laplacians`: the existing JL plus approximate-Cholesky implementation for
  small, weighted, general undirected graphs.  Projection assembly, edge
  accumulation, and directed alignment use bounded static Julia thread slots;
  the solver itself contains serial sections.
- `tgt`: a standalone C++17/OpenMP implementation of the TGT+ all-edge
  spanning-centrality algorithm for large, unweighted graphs.  For an
  unweighted graph, spanning centrality is effective resistance.

`auto` selects Laplacians.jl only when both the node and edge limits are met.
The selected backend and its reason are printed before any computation.

## Large-graph data path

EDSparse's already-generated `input_graph.npz` remains the source of truth.
It is read-only.  A streaming converter writes a cache next to it called
`input_graph.tgtbin`; at most one chunk of one NumPy member is resident during
conversion.  The binary contains canonical `uint32` endpoints and the
directed-to-canonical edge mapping.  The C++ process memory maps these arrays
and constructs one compact CSR:

```text
row offsets: uint64[n + 1]
neighbors:   uint32[2m]
edge ids:    uint32[2m]
```

The converter rejects non-unit conductances because TGT+'s result is
spanning centrality only in the unweighted case.  It never changes raw or
processed dataset files.

## TGT+ computation

The implementation follows Algorithms 1, 3, and 4 of Zhang et al., KDD 2023:

1. Construct the symmetric normalized adjacency operator
   `D^(-1/2) A D^(-1/2)` without forming a second sparse matrix.
2. Use ARPACK reverse communication to compute the `omega` eigenpairs with
   largest absolute eigenvalue.  Eigenvectors are stored node-major as
   `float32`; this is a restartable preprocessing cache.
3. Compute the odd edge-specific truncation length from Equations 6--8.
4. For each source node, perform sparse deterministic transition-probability
   pushes until random walks are cheaper, then estimate the remaining terms
   with the Hoeffding sample count and CalChi bound.
5. Store the two oriented contributions independently and add them once both
   endpoints have been processed.

Each OpenMP worker owns its probability vectors, sparse active-node lists,
markers, and deterministic RNG.  No `vector<vector<...>>`, NetworkX graph,
dense Laplacian, per-edge map, or per-edge Python object is used.

The backend verifies the assumptions needed by these formulas: the graph is
simple, unweighted, and undirected.  TGT+ is applied to the largest
edge-bearing component so its eigensystem has one stationary mode.  Remaining
components are small in the EDSparse collection and are solved exactly and
independently with a dense symmetric eigensolver.  A violated input assumption
is a hard error rather than an unlabelled approximation.

## Memory and restart behavior

Before allocation, the executable prints a conservative peak-memory estimate
and refuses to start if `--memory-gb` is lower.  The important terms are CSR,
ARPACK's `n * ncv` basis, `omega` node eigenvectors, two oriented edge-result
arrays, and per-thread traversal workspaces.  Graphs run sequentially, so only
one dataset's state is resident.

The large backend writes intermediate files in the dataset's
`spectral_sparsification/tgt_work/` directory.  Eigen preprocessing is reused
when the graph fingerprint and parameters match.  Source-node progress is
checkpointed at fixed blocks; a Slurm retry resumes at the last completed
block.  Final `.npz` files are written atomically and include one directed ER
value for every original EDSparse `edge_index` column.

## Accuracy and validation

`epsilon`, `delta`, `omega`, `gamma`, and the seed are artifact metadata and
cache keys.  The default large-graph accuracy is `epsilon=0.05`, matching the
paper's scalability setting, with `omega=128` and `gamma=10`.

The test path compares TGT results on small connected graphs against dense
Laplacian pseudoinverse values and checks that canonical and directed lengths,
fingerprints, finite values, edge bounds, and the Kirchhoff leverage sum are
consistent.  Large production jobs also perform the inexpensive structural
and range checks before publishing their final artifact.
