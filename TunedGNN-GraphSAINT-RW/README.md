# TunedGNN-GraphSAINT-RW

The tunedGNN medium-graph GCN — `MPNNs(..., gnn='gcn')` from
`../tunedGNN-main/medium_graph/model.py`, subclassed, not reimplemented —
trained on subgraphs produced by PyG's `GraphSAINTRandomWalkSampler`.

The sampler, its defaults, the optimizer, the evaluation protocol, and the
result reporting are all reused from `../GraphSAINT-RW/graphsaint_rw.py`, so the
only deliberate difference from the GraphSAINT-RW baseline is the backbone.

## Sampler settings (inherited, unchanged)

- Small graphs (<50k nodes): batch size 500, walk length 2, 3 steps, sampling
  normalization on.
- Reddit-scale (<500k nodes): batch size 6000, walk length 4, 30 steps.
- Million-node graphs: batch size 20000, walk length 4, 30 steps.

## Reusing a saved normalization

`data.edge_index` is left exactly as the GraphSAINT-RW baseline sees it, so the
sampler cache fingerprint — dataset, node count, edge count, batch size, walk
length, step count, coverage — is identical. Pointing `--sampler_cache_dir` at
the GraphSAINT cache therefore loads that dataset's existing `node_norm` /
`edge_norm` instead of recomputing them. The bundled wrapper
(`ICML_SPARSIFICATION/scripts/methods/tunedgnn_graphsaint/`) already points at
`${CACHE_ROOT}/graphsaint/sampler` for this reason.

## Where the GCN normalization comes from

This is the one thing that cannot be inherited. `GCNConv(normalize=True)`
re-derives `D^-1/2 A~ D^-1/2` from whatever edge weights it is handed, so
feeding it GraphSAINT's `edge_norm` would both cancel the sampling-bias
correction and normalize against *subgraph* degrees instead of full-graph ones.

Instead, `TunedGNNGCN.prepare_data` precomputes the full-graph coefficients
once:

- `gcn_norm` (edge-level) — `deg^-1/2[row] * deg^-1/2[col]` with the implicit
  GCN self-loop counted in `deg`.
- `gcn_self_norm` (node-level) — `1 / deg`.

Both are ordinary `Data` attributes, so the GraphSAINT and `NeighborLoader`
collates slice them alongside the graph they already index. Each conv then runs
with `normalize=False` and:

- **normalized sampled training** — `gcn_norm[e] * edge_norm[e]` on the sampled
  edges, plus one self-loop per sampled node at `gcn_self_norm[v]`. A self-loop
  is present exactly when its node is, so its GraphSAINT `alpha` is 1 and it
  takes no correction.
- **evaluation** — `gcn_norm` / `gcn_self_norm` unmodified, which reproduces
  stock `GCNConv(normalize=True)` on the full graph to float32 precision.
- **layer-wise `NeighborLoader` inference** (graphs too large for full-graph
  eval) — the same full-graph coefficients for the sampled edges, so seed-node
  outputs match full-graph evaluation exactly even though the sampled
  neighbors' own degrees are truncated.

## Measured on Cora

tunedGNN split (140/500/1000), preset `hidden=512, layers=3, dropout=0.7,
lr=1e-3, wd=5e-4`, 500 epochs, 3 runs, CPU. Reported figure is the
validation-selected test accuracy.

| backbone | sampler | test acc |
| --- | --- | --- |
| GraphConv (GraphSAINT-RW baseline) | RW, bias norm **on** | 79.73 ± 0.61 |
| GraphConv (GraphSAINT-RW baseline) | RW, bias norm **off** | 76.83 ± 0.67 |
| tunedGNN GCN (this method) | RW, bias norm **on** | 79.13 ± 1.35 |
| tunedGNN GCN (this method) | RW, bias norm **off** | **80.23 ± 1.08** |
| tunedGNN GCN | full graph (reference) | 81.80 ± 1.14 |

Two things to read off this. The backbones respond to GraphSAINT's
sampling-bias reweighting in opposite directions: it helps GraphConv and hurts
the GCN. `edge_norm` reaches 7.7 on Cora, so multiplying it into the symmetric
GCN coefficients moves every sampled edge far from the `D^-1/2 A~ D^-1/2` value
the model is evaluated at — unbiased per layer, but high variance, and three
layers compose. GraphConv is less exposed because its separate `W1 x_i` self
transform carries signal that never passes through `edge_norm`.

The default stays `--use_normalization auto` (i.e. on for small graphs) so that
this method and the GraphSAINT-RW baseline sample identically and the
comparison isolates the backbone. Pass `--use_normalization false` to get the
80.23 row.

## Loss scale

The shared backend defaults to `--normalized_loss_scale graphsaint_sum`, the
PyG example's `sum / num_nodes` scale. `train_mean` instead rescales to the
per-train-node mean that the tunedGNN presets pin `weight_decay` against —
principled, but measured as a wash on Cora (79.1 → 79.3 here, 79.7 → 79.0 for
the GraphSAINT-RW baseline), so it is opt-in.

## Example

```bash
python tunedgnn_graphsaint_rw.py \
  --dataset Cora \
  --data_root /path/to/data \
  --hidden_channels 512 \
  --num_layers 3 \
  --dropout 0.7 \
  --lr 0.001 \
  --weight_decay 0.0005 \
  --epochs 500 \
  --runs 5 \
  --seed 123
```

Or through the harness, which supplies the resolved tunedGNN preset contract:

```bash
python EDSparse/scripts/Configuration.py command \
  --method tunedgnn-graphsaint-rw --dataset cora
```
