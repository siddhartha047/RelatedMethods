"""GraphSAINT-RW node classification baseline.

Follows the canonical PyG example ``examples/graph_saint.py`` verbatim in
sampler + model, adapted to:
  - Datasets loaded through ``ICML_SPARSIFICATION.scripts.baseline_dataset_bridge``
    (Cora / Reddit / OGB-products / OGB-arxiv / OGB-proteins / Pokec).
  - Full-input-graph training via random-walk sampled subgraphs.
  - Layer-wise NeighborLoader inference for graphs where full-graph eval OOMs
    (e.g. OGB-products, 2.4 M nodes → 46 GiB message tensor on an 80 GiB GPU).

Reference: https://github.com/pyg-team/pytorch_geometric/blob/master/examples/graph_saint.py
"""

import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch_geometric.loader import GraphSAINTRandomWalkSampler, NeighborLoader
from torch_geometric.nn import GraphConv
from torch_geometric.utils import coalesce, degree, is_undirected, remove_self_loops, to_undirected

SUPPORT_GRAPH_ROOT = Path(os.environ.get("SUPPORT_GRAPH_ROOT", Path(__file__).resolve().parents[2])).resolve()
if str(SUPPORT_GRAPH_ROOT) not in sys.path:
    sys.path.insert(0, str(SUPPORT_GRAPH_ROOT))
from EDSparseDataset import load_pyg_data
from ICML_SPARSIFICATION.scripts.baseline_result_utils import (
    append_baseline_result,
    macro_f1_percent,
    multilabel_roc_auc_f1_percent,
)
from ICML_SPARSIFICATION.utils.defaults import DEFAULT_CACHE_DIR, DEFAULT_DATA_DIR


def parse_bool(value):
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "yes", "y"}:
        return True
    if lowered in {"0", "false", "no", "n"}:
        return False
    raise ValueError(f"Invalid boolean value: {value}")


def fix_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def graphsaint_sampler_cache_dir(cache_root: str, dataset_name: str, data, args) -> str:
    """Return a cache directory unique to this graph and sampler setup.

    PyG names its normalization file only from ``sample_coverage``. Reusing one
    save directory across datasets therefore loads tensors with the wrong node
    and edge counts. Keep a small human-readable dataset level plus a stable
    fingerprint of every setting that changes the normalization statistics.
    """
    if not cache_root:
        return ""
    canonical = str(dataset_name).strip().lower().replace("_", "-")
    metadata = {
        "schema": 3,
        "dataset": canonical,
        "num_nodes": int(data.num_nodes),
        "num_edges": int(data.edge_index.size(1)),
        "batch_size": int(args.batch_size),
        "walk_length": int(args.walk_length),
        "num_steps": int(args.num_steps),
        "sample_coverage": int(args.sample_coverage),
    }
    encoded = json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")
    fingerprint = hashlib.sha256(encoded).hexdigest()[:16]
    cache_dir = Path(cache_root).expanduser() / canonical / fingerprint
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_dir / "manifest.json"
    manifest_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")

    norm_path = cache_dir / f"graphsaint_random_walk_sampler_{args.sample_coverage}.pt"
    if norm_path.exists():
        try:
            node_norm, edge_norm = torch.load(norm_path, map_location="cpu", weights_only=False)
            valid = (
                int(node_norm.numel()) == int(data.num_nodes)
                and int(edge_norm.numel()) == int(data.edge_index.size(1))
            )
        except Exception:
            valid = False
        if not valid:
            print(f"[Sampler cache] removing incompatible normalization cache: {norm_path}", flush=True)
            norm_path.unlink(missing_ok=True)
    return str(cache_dir)


def build_sgs_split(num_nodes: int, labels=None, seed: int = 1):
    if labels is not None:
        labels = np.asarray(labels).reshape(-1)
        indices = np.where(labels != -1)[0]
    else:
        indices = np.arange(num_nodes)
    train_idx, temp_idx = train_test_split(indices, test_size=0.8, random_state=seed)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=seed)
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True
    return train_mask, val_mask, test_mask


def load_sgs_dataset(dataset_name: str, data_root: str):
    data, dataset = load_pyg_data(data_root, dataset_name)
    if not is_undirected(data.edge_index):
        data.edge_index = to_undirected(data.edge_index)
    if data.y.dim() > 1:
        if data.y.size(-1) == 1:
            data.y = data.y.view(-1)
        else:
            # Preserve OGBN-Proteins' 112 independent binary targets.
            data.y = data.y.to(torch.float)
    else:
        data.y = data.y.view(-1)
    if data.y.dim() == 1:
        data.y = data.y.to(torch.long)
    return dataset, data


class SaintNet(torch.nn.Module):
    """3-layer GraphConv with concat + linear head — matches PyG's graph_saint.py."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        dropout: float = 0.2,
        multilabel: bool = False,
    ):
        super().__init__()
        self.conv1 = GraphConv(in_channels, hidden_channels)
        self.conv2 = GraphConv(hidden_channels, hidden_channels)
        self.conv3 = GraphConv(hidden_channels, hidden_channels)
        self.lin = torch.nn.Linear(3 * hidden_channels, out_channels)
        self.dropout = dropout
        self.multilabel = bool(multilabel)

    def set_aggr(self, aggr: str):
        self.conv1.aggr = aggr
        self.conv2.aggr = aggr
        self.conv3.aggr = aggr

    def forward(self, x0, edge_index, edge_weight=None):
        x1 = F.relu(self.conv1(x0, edge_index, edge_weight))
        x1 = F.dropout(x1, p=self.dropout, training=self.training)
        x2 = F.relu(self.conv2(x1, edge_index, edge_weight))
        x2 = F.dropout(x2, p=self.dropout, training=self.training)
        x3 = F.relu(self.conv3(x2, edge_index, edge_weight))
        x3 = F.dropout(x3, p=self.dropout, training=self.training)
        x = torch.cat([x1, x2, x3], dim=-1)
        x = self.lin(x)
        return x if self.multilabel else x.log_softmax(dim=-1)

    @torch.no_grad()
    def inference(self, x_all: torch.Tensor, subgraph_loader: NeighborLoader, device: torch.device):
        """Layer-wise NeighborLoader inference for graphs too big for full-graph eval.

        Streams each conv over all nodes with a 1-hop NeighborLoader, storing
        intermediate activations on CPU. Mirrors the pattern used in PyG's
        ``examples/reddit.py``. The final linear head is applied per-batch to
        avoid materializing the 3H-wide concat over all nodes at once.
        """
        self.eval()
        self.set_aggr("mean")

        def _run_layer(x_all_cpu, conv, apply_relu):
            xs = torch.empty((x_all_cpu.size(0), conv.out_channels))
            for batch in subgraph_loader:
                n_id_cpu = batch.n_id.cpu()
                xb = x_all_cpu[n_id_cpu].to(device)
                edge_index = batch.edge_index.to(device)
                h = conv(xb, edge_index)
                if apply_relu:
                    h = F.relu(h)
                bs = int(batch.batch_size)
                xs[n_id_cpu[:bs]] = h[:bs].cpu()
            return xs

        h1 = _run_layer(x_all, self.conv1, apply_relu=True)
        h2 = _run_layer(h1, self.conv2, apply_relu=True)
        h3 = _run_layer(h2, self.conv3, apply_relu=True)
        num_nodes = h3.size(0)
        out = torch.empty((num_nodes, self.lin.out_features))
        step = 200_000
        for start in range(0, num_nodes, step):
            end = min(start + step, num_nodes)
            block = torch.cat(
                [h1[start:end], h2[start:end], h3[start:end]], dim=-1
            ).to(device)
            block_output = self.lin(block)
            if not self.multilabel:
                block_output = block_output.log_softmax(dim=-1)
            out[start:end] = block_output.cpu()
        return out


def macro_f1(pred: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> float:
    mask = mask.bool()
    labels = labels[mask]
    preds = pred[mask]
    if labels.numel() == 0:
        return 0.0
    return macro_f1_percent(labels, preds) / 100.0


@torch.no_grad()
def evaluate(
    model,
    data,
    use_normalization,
    device,
    full_graph_eval,
    subgraph_loader,
    multilabel,
):
    # Follows PyG's examples/graph_saint.py test(): plain
    #   correct[mask].sum() / mask.sum()
    # accuracy on the full graph. Both eval paths (full-graph vs layer-wise
    # NeighborLoader) return log-softmax logits; argmax is the prediction.
    model.eval()
    model.set_aggr("mean")
    if full_graph_eval:
        logits = model(data.x.to(device), data.edge_index.to(device))
    else:
        logits = model.inference(data.x, subgraph_loader, device)
    if multilabel:
        labels = data.y.cpu()
        logits = logits.cpu()

        def _metric(mask):
            mask = mask.cpu().bool()
            metric, f1 = multilabel_roc_auc_f1_percent(
                labels[mask], logits[mask]
            )
            return metric / 100.0, f1 / 100.0

        train_metric, train_f1_macro = _metric(data.train_mask)
        val_metric, _ = _metric(data.val_mask)
        test_metric, test_f1_macro = _metric(data.test_mask)
        return (
            train_metric,
            val_metric,
            test_metric,
            train_f1_macro,
            test_f1_macro,
        )

    pred = logits.argmax(dim=-1)
    labels = data.y.to(pred.device)
    correct = pred.eq(labels)

    def _acc(mask):
        mask = mask.to(pred.device).bool()
        valid = mask & (labels >= 0)
        denom = int(valid.sum().item())
        if denom == 0:
            return 0.0
        return float(correct[valid].sum().item()) / denom

    train_acc = _acc(data.train_mask)
    val_acc = _acc(data.val_mask)
    test_acc = _acc(data.test_mask)
    train_f1_macro = macro_f1(pred, labels, data.train_mask)
    test_f1_macro = macro_f1(pred, labels, data.test_mask)
    return train_acc, val_acc, test_acc, train_f1_macro, test_f1_macro


def _multilabel_loss(logits, labels, node_mask, node_norm=None):
    valid = torch.isfinite(labels) & (labels >= 0)
    losses = F.binary_cross_entropy_with_logits(
        logits,
        torch.nan_to_num(labels, nan=0.0).float(),
        reduction="none",
    )
    valid_count = valid.sum(dim=-1)
    per_node = (losses * valid).sum(dim=-1) / valid_count.clamp(min=1)
    selected = node_mask.bool() & (valid_count > 0)
    if not bool(selected.any()):
        return logits.sum() * 0.0
    if node_norm is not None:
        return (per_node * node_norm)[selected].sum()
    return per_node[selected].mean()


def train_one_run(
    args,
    dataset_name,
    data,
    num_features,
    num_classes,
    run_seed,
    run_id,
    multilabel,
):
    fix_seed(run_seed)
    metric_name = "ROC-AUC" if multilabel else "Accuracy"

    model = SaintNet(
        in_channels=num_features,
        hidden_channels=args.hidden_channels,
        out_channels=num_classes,
        dropout=args.dropout,
        multilabel=multilabel,
    ).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    saint_kwargs = dict(
        batch_size=args.batch_size,
        walk_length=args.walk_length,
        num_steps=args.num_steps,
        sample_coverage=args.sample_coverage,
        num_workers=args.sampler_num_workers,
    )
    sampler_cache_dir = graphsaint_sampler_cache_dir(
        args.sampler_cache_dir, dataset_name, data, args
    )
    if sampler_cache_dir:
        saint_kwargs["save_dir"] = sampler_cache_dir
    print(
        f"[Sampler init] batch_size={args.batch_size} walk_length={args.walk_length} "
        f"num_steps={args.num_steps} sample_coverage={args.sample_coverage} "
        f"save_dir={sampler_cache_dir or None}",
        flush=True,
    )
    loader = GraphSAINTRandomWalkSampler(data, **saint_kwargs)

    full_graph_eval = (
        int(data.num_nodes) <= args.full_graph_eval_max_nodes
        and int(data.edge_index.size(1)) <= args.full_graph_eval_max_edges
    )
    subgraph_loader = None
    if not full_graph_eval:
        subgraph_loader = NeighborLoader(
            data,
            num_neighbors=[-1],
            batch_size=args.eval_batch_size,
            shuffle=False,
            num_workers=args.sampler_num_workers,
        )
    print(
        f"[Eval mode] full_graph_eval={full_graph_eval} "
        f"nodes={int(data.num_nodes)} edges={int(data.edge_index.size(1))}",
        flush=True,
    )

    best_val = float("-inf")
    chosen_test = 0.0
    chosen_train = 0.0
    chosen_test_macro = 0.0
    chosen_train_macro = 0.0
    chosen_epoch = 0
    train_f1 = val_f1 = test_f1 = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        model.set_aggr("add" if args.use_normalization else "mean")
        total_loss = 0.0
        total_examples = 0

        for batch in loader:
            batch = batch.to(args.device)
            optimizer.zero_grad()

            train_mask_batch = batch.train_mask.bool()
            n_train = int(train_mask_batch.sum())
            if n_train == 0:
                continue

            if args.use_normalization:
                edge_weight = batch.edge_norm * batch.edge_weight
                out = model(batch.x, batch.edge_index, edge_weight)
                if multilabel:
                    loss = _multilabel_loss(
                        out,
                        batch.y,
                        train_mask_batch,
                        node_norm=batch.node_norm,
                    )
                else:
                    loss = F.nll_loss(out, batch.y, reduction="none")
                    loss = (loss * batch.node_norm)[train_mask_batch].sum()
            else:
                out = model(batch.x, batch.edge_index)
                if multilabel:
                    loss = _multilabel_loss(
                        out, batch.y, train_mask_batch
                    )
                else:
                    loss = F.nll_loss(
                        out[train_mask_batch], batch.y[train_mask_batch]
                    )

            loss.backward()
            optimizer.step()
            # Match the official PyG GraphSAINT example. In particular, do not
            # divide the node-normalized loss by the number of labeled nodes a
            # second time merely for reporting.
            total_loss += float(loss.item()) * int(batch.num_nodes)
            total_examples += int(batch.num_nodes)

        if epoch == 1 or epoch % args.eval_step == 0 or epoch == args.epochs:
            train_f1, val_f1, test_f1, train_f1_macro, test_f1_macro = evaluate(
                model,
                data,
                args.use_normalization,
                args.device,
                full_graph_eval,
                subgraph_loader,
                multilabel,
            )
            avg_loss = total_loss / max(total_examples, 1)
            if val_f1 > best_val:
                best_val = val_f1
                chosen_test = test_f1
                chosen_train = train_f1
                chosen_test_macro = test_f1_macro
                chosen_train_macro = train_f1_macro
                chosen_epoch = epoch
            print(
                f"Epoch: {epoch:03d}, Loss: {avg_loss:.4f}, "
                f"Train: {train_f1 * 100:.2f}%, Val: {val_f1 * 100:.2f}%, "
                f"Test: {test_f1 * 100:.2f}%, Test F1 Macro: {test_f1_macro * 100:.2f}%",
                flush=True,
            )

    print(
        f"run_{run_id} metric={metric_name} train_metric: {train_f1:.6f} "
        f"val_metric: {val_f1:.6f} test_metric: {test_f1:.6f} "
        f"best_val_metric: {best_val:.6f} "
        f"chosen_test_metric: {chosen_test:.6f} "
        f"chosen_test_f1_macro: {chosen_test_macro:.6f} "
        f"chosen_epoch: {chosen_epoch}"
    )
    append_baseline_result(
        method="graphsaint",
        dataset=dataset_name,
        run=run_id,
        seed=run_seed,
        epochs=args.epochs,
        train_acc=100 * chosen_train,
        valid_acc=100 * best_val,
        test_acc=100 * chosen_test,
        train_f1_macro=100 * chosen_train_macro,
        test_f1_macro=100 * chosen_test_macro,
        chosen_epoch=chosen_epoch,
    )
    return chosen_test


def _default_batch_size(num_nodes: int) -> int:
    # Rough tier defaults tuned for graph coverage and GPU memory:
    #  small graphs (< 50k nodes): 500
    #  reddit-scale (< 500k):      6000  (matches PyG example)
    #  OGB-products / pokec:       20000
    if num_nodes < 50_000:
        return 500
    if num_nodes < 500_000:
        return 6000
    return 20_000


def resolve_sampler_settings(args, num_nodes: int):
    """Resolve graph-scale GraphSAINT defaults while preserving CLI overrides.

    GraphSAINT defines an epoch by graph coverage. A Cora batch with 2,000
    random-walk roots already covers most of its 2,708 nodes, so the previous
    universal 30-step setting performed many updates before the first
    validation. Small graphs instead use smaller subgraphs, three coverage
    steps, and the paper's node/edge sampling-bias normalization.
    """

    is_small_graph = int(num_nodes) < 50_000
    if int(args.batch_size) <= 0:
        args.batch_size = _default_batch_size(int(num_nodes))
    if int(args.walk_length) <= 0:
        args.walk_length = 2 if is_small_graph else 4
    if int(args.num_steps) <= 0:
        args.num_steps = 3 if is_small_graph else 30

    normalization = str(args.use_normalization).strip().lower()
    if normalization == "auto":
        args.use_normalization = is_small_graph
    else:
        args.use_normalization = parse_bool(args.use_normalization)

    if int(args.sample_coverage) < 0:
        args.sample_coverage = 100 if args.use_normalization else 0

    if int(args.batch_size) < 1:
        raise ValueError("batch_size must be positive after default resolution")
    if int(args.walk_length) < 1:
        raise ValueError("walk_length must be positive after default resolution")
    if int(args.num_steps) < 1:
        raise ValueError("num_steps must be positive after default resolution")
    if int(args.sample_coverage) < 0:
        raise ValueError("sample_coverage must be non-negative")
    return args


def main():
    parser = argparse.ArgumentParser(description="GraphSAINT-RW baseline (PyG official sampler)")
    parser.add_argument("--dataset", required=True)
    # Resolve to the SAME tree main.py / scaffold_fast use (DEFAULT_DATA_DIR in
    # utils/defaults.py). That way we always load the same cached Reddit /
    # ogbn_products / etc. and don't re-download per node.
    parser.add_argument("--data_root", type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--hidden_channels", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=0,
                        help="0 = graph-scale default (500 / 6000 / 20000).")
    parser.add_argument("--walk_length", type=int, default=0,
                        help="0 = graph-scale default (2 small, 4 large).")
    parser.add_argument("--num_steps", type=int, default=0,
                        help="0 = graph-coverage default (3 small, 30 large).")
    parser.add_argument("--sample_coverage", type=int, default=-1,
                        help="-1 = 100 with bias normalization, otherwise 0.")
    parser.add_argument("--sampler_num_workers", type=int, default=0)
    parser.add_argument("--eval_step", type=int, default=1)
    parser.add_argument("--eval_batch_size", type=int, default=4096)
    parser.add_argument("--full_graph_eval_max_nodes", type=int, default=50_000,
                        help="Use full-graph eval below this many nodes; else layer-wise NeighborLoader eval.")
    parser.add_argument("--full_graph_eval_max_edges", type=int, default=2_000_000,
                        help="Also fall back to layer-wise NeighborLoader eval above this many edges.")
    parser.add_argument("--sampler_cache_dir", type=str,
                        default=os.environ.get(
                            "GRAPHSAINT_SAMPLER_CACHE",
                            str(Path(DEFAULT_CACHE_DIR) / "graphsaint" / "sampler"),
                        ))
    parser.add_argument(
        "--use_normalization",
        type=str,
        default="auto",
        help=(
            "true/false/auto. Auto enables GraphSAINT node/edge bias "
            "normalization on small graphs and disables it on large graphs."
        ),
    )
    args = parser.parse_args()

    if args.cpu or not torch.cuda.is_available():
        args.device = torch.device("cpu")
    else:
        args.device = torch.device(f"cuda:{args.device}")

    _dataset, data = load_sgs_dataset(args.dataset, args.data_root)
    data.edge_index, _ = remove_self_loops(data.edge_index)
    data.edge_index = coalesce(data.edge_index, num_nodes=data.num_nodes)
    num_edges = int(data.edge_index.size(1))

    # PyG graph_saint.py convention: edge_weight = 1 / in_degree(col)
    _row, col = data.edge_index
    deg = degree(col, data.num_nodes, dtype=torch.float)
    deg = torch.clamp(deg, min=1.0)
    data.edge_weight = 1.0 / deg[col]

    multilabel = data.y.dim() > 1 and data.y.size(-1) > 1
    if multilabel:
        num_classes = int(data.y.size(-1))
    else:
        labeled = data.y[data.y != -1]
        num_classes = int(labeled.max().item()) + 1 if labeled.numel() else 0
    num_features = int(data.x.size(1))

    resolve_sampler_settings(args, int(data.num_nodes))

    print(
        f"dataset {args.dataset} | num nodes {data.num_nodes} | "
        f"num edge {num_edges} | num node feats {num_features} | "
        f"num outputs {num_classes} | "
        f"metric {'ROC-AUC' if multilabel else 'Accuracy'}"
    )
    print(
        f"[Resolved sampler] batch_size={args.batch_size} "
        f"walk_length={args.walk_length} num_steps={args.num_steps} "
        f"sample_coverage={args.sample_coverage} "
        f"use_normalization={args.use_normalization}"
    )

    run_scores = []
    for run_idx in range(1, args.runs + 1):
        run_seed = args.seed + run_idx - 1
        score = train_one_run(
            args,
            args.dataset,
            data,
            num_features,
            num_classes,
            run_seed,
            run_idx,
            multilabel,
        )
        run_scores.append(score)

    run_scores = np.asarray(run_scores, dtype=np.float64)
    mean = float(run_scores.mean()) if len(run_scores) else 0.0
    std = float(run_scores.std(ddof=1)) if len(run_scores) > 1 else 0.0
    metric_key = "roc_auc" if multilabel else "accuracy"
    print(
        f"all_runs chosen_test_{metric_key}_mean: {mean:.6f} "
        f"chosen_test_{metric_key}_std: {std:.6f}"
    )


if __name__ == "__main__":
    main()
