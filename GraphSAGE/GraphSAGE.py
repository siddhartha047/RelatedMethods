"""GraphSAGE mini-batch node classification baseline.

Mirrors the canonical PyG example ``examples/reddit.py``:
  - ``NeighborLoader`` neighbor sampling for training
  - ``SAGEConv`` stack (default 2 layers with fanout ``[25, 10]``, matching
    the PyG Reddit example) and layer-wise ``NeighborLoader`` inference so
    full-graph eval never allocates a multi-hop message tensor.

Reference:
  https://github.com/pyg-team/pytorch_geometric/blob/master/examples/reddit.py
"""

import argparse
import copy
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import coalesce, is_undirected, remove_self_loops, to_undirected

SUPPORT_GRAPH_ROOT = Path(os.environ.get("SUPPORT_GRAPH_ROOT", Path(__file__).resolve().parents[2])).resolve()
if str(SUPPORT_GRAPH_ROOT) not in sys.path:
    sys.path.insert(0, str(SUPPORT_GRAPH_ROOT))
from EDSparseDataset import load_pyg_data
from ICML_SPARSIFICATION.scripts.baseline_result_utils import append_baseline_result, macro_f1_percent
from ICML_SPARSIFICATION.utils.defaults import DEFAULT_DATA_DIR


class SAGE(torch.nn.Module):
    """Stack of SAGEConv layers matching PyG's Reddit example.

    Layer-wise inference performs one conv at a time over all nodes with a
    1-hop NeighborLoader, holding intermediate representations on CPU. This
    is what lets Reddit / OGB-products fit on a single GPU during eval.
    """

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int,
                 num_layers: int = 3, dropout: float = 0.5):
        super().__init__()
        self.dropout = dropout
        self.convs = torch.nn.ModuleList()
        if num_layers == 1:
            self.convs.append(SAGEConv(in_channels, out_channels))
        else:
            self.convs.append(SAGEConv(in_channels, hidden_channels))
            for _ in range(num_layers - 2):
                self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.convs.append(SAGEConv(hidden_channels, out_channels))

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = x.relu_()
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    @torch.no_grad()
    def inference(self, x_all: torch.Tensor, subgraph_loader: NeighborLoader, device: torch.device):
        # Compute representations layer-by-layer using *all* edges, matching
        # PyG's examples/reddit.py inference() loop.
        for i, conv in enumerate(self.convs):
            xs = []
            for batch in subgraph_loader:
                x = x_all[batch.n_id.to(x_all.device)].to(device)
                x = conv(x, batch.edge_index.to(device))
                if i < len(self.convs) - 1:
                    x = x.relu_()
                xs.append(x[: batch.batch_size].cpu())
            x_all = torch.cat(xs, dim=0)
        return x_all


def fix_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _prepare_data(data):
    if data.y.dim() > 1 and data.y.size(-1) == 1:
        data.y = data.y.view(-1)
    if data.y.dim() > 1:
        data.y = data.y.argmax(dim=-1)
    data.y = data.y.to(torch.long)

    data.edge_index, _ = remove_self_loops(data.edge_index)
    if not is_undirected(data.edge_index):
        data.edge_index = to_undirected(data.edge_index, num_nodes=data.num_nodes)
    data.edge_index = coalesce(data.edge_index, num_nodes=data.num_nodes)
    return data


def train_one_epoch(model, loader, optimizer, device):
    # Features/labels already live on ``device`` (see ``data.to(device, "x", "y")``
    # in main). PyG's examples/reddit.py only moves ``edge_index`` per batch —
    # doing a full ``batch.to(device)`` would round-trip x/y through H2D each
    # step for no reason.
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    for batch in loader:
        optimizer.zero_grad()
        y = batch.y[: batch.batch_size]
        y_hat = model(batch.x, batch.edge_index.to(device))[: batch.batch_size]
        loss = F.cross_entropy(y_hat, y)
        loss.backward()
        optimizer.step()
        total_loss += float(loss) * batch.batch_size
        total_correct += int((y_hat.argmax(dim=-1) == y).sum())
        total_examples += int(batch.batch_size)
    return total_loss / max(total_examples, 1), total_correct / max(total_examples, 1)


@torch.no_grad()
def evaluate(model, data, subgraph_loader, device):
    model.eval()
    y_hat = model.inference(data.x, subgraph_loader, device).argmax(dim=-1)
    y = data.y.to(y_hat.device)
    accs = []
    f1s = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        mask = mask.to(y_hat.device).bool()
        accs.append(int((y_hat[mask] == y[mask]).sum()) / int(mask.sum().item() or 1))
        f1s.append(macro_f1_percent(y[mask], y_hat[mask]) / 100.0)
    return accs, f1s


def main():
    parser = argparse.ArgumentParser(description="GraphSAGE neighbor-sampling baseline (PyG)")
    parser.add_argument("--dataset", default="Cora")
    # Resolve to the SAME tree main.py / scaffold_fast use (DEFAULT_DATA_DIR in
    # utils/defaults.py). That way we always load the same cached Reddit /
    # ogbn_products / etc. and don't re-download per node.
    parser.add_argument("--data_root", default=DEFAULT_DATA_DIR)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden_channels", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.5)
    # PyG's examples/reddit.py uses lr=0.01. 0.003 (a common OGB baseline
    # default) still converges but ~3x slower.
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--eval_batch_size", type=int, default=4096)
    parser.add_argument("--num_neighbors", default="25,10")
    parser.add_argument("--num_workers", type=int, default=6)
    parser.add_argument("--persistent_workers", type=str, default="true")
    parser.add_argument("--eval_step", type=int, default=1)
    parser.add_argument("--display_step", type=int, default=1)
    args = parser.parse_args()

    fix_seed(args.seed)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else f"cuda:{args.device}")

    data, _dataset = load_pyg_data(args.data_root, args.dataset)
    data = _prepare_data(data)

    fanout = [int(x) for x in args.num_neighbors.split(",") if x.strip()]
    if len(fanout) < args.num_layers:
        fanout = fanout + [fanout[-1]] * (args.num_layers - len(fanout))
    elif len(fanout) > args.num_layers:
        fanout = fanout[: args.num_layers]

    labeled = data.y[data.y >= 0]
    num_classes = int(labeled.max().item()) + 1 if labeled.numel() else 0

    print(
        f"dataset {args.dataset} | num nodes {data.num_nodes} | num edge {data.edge_index.size(1)} | "
        f"num node feats {data.num_features} | num classes {num_classes} | fanout {fanout}",
        flush=True,
    )

    # Move features/labels to GPU up-front — same as PyG's reddit.py.
    data = data.to(device, "x", "y")
    persistent = args.num_workers > 0 and str(args.persistent_workers).lower() in {"1", "true", "yes", "y"}
    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
    }
    if persistent:
        loader_kwargs["persistent_workers"] = True

    eval_loader_kwargs = {
        "batch_size": args.eval_batch_size,
        "num_workers": args.num_workers,
    }
    if persistent:
        eval_loader_kwargs["persistent_workers"] = True

    all_run_test = []
    for run_idx in range(1, args.runs + 1):
        run_seed = args.seed + run_idx - 1
        fix_seed(run_seed)

        train_loader = NeighborLoader(
            data,
            input_nodes=data.train_mask,
            num_neighbors=fanout,
            shuffle=True,
            **loader_kwargs,
        )

        # Eval subgraph loader: strip features / labels from the copy so PyG
        # doesn't allocate a huge per-worker copy.
        subgraph_loader = NeighborLoader(
            copy.copy(data),
            input_nodes=None,
            num_neighbors=[-1],
            shuffle=False,
            **eval_loader_kwargs,
        )
        del subgraph_loader.data.x
        if hasattr(subgraph_loader.data, "y"):
            del subgraph_loader.data.y
        subgraph_loader.data.num_nodes = data.num_nodes
        subgraph_loader.data.n_id = torch.arange(data.num_nodes)

        model = SAGE(
            in_channels=data.num_features,
            hidden_channels=args.hidden_channels,
            out_channels=num_classes,
            num_layers=args.num_layers,
            dropout=args.dropout,
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        best_val = float("-inf")
        best = None
        times = []
        for epoch in range(1, args.epochs + 1):
            t0 = time.time()
            loss, approx_train = train_one_epoch(model, train_loader, optimizer, device)
            if epoch == 1 or epoch % args.eval_step == 0 or epoch == args.epochs:
                (train_acc, val_acc, test_acc), (train_f1, val_f1, test_f1) = evaluate(
                    model, data, subgraph_loader, device
                )
                if val_acc > best_val:
                    best_val = val_acc
                    best = (epoch, train_acc, val_acc, test_acc, train_f1, test_f1)
                if epoch == 1 or epoch % args.display_step == 0 or epoch == args.epochs:
                    print(
                        f"Run {run_idx} Epoch {epoch:03d} | Loss {loss:.4f} | ApproxTrain {approx_train:.4f} | "
                        f"Train {train_acc:.4f} | Val {val_acc:.4f} | Test {test_acc:.4f} | "
                        f"Test F1 Macro {test_f1:.4f}",
                        flush=True,
                    )
            times.append(time.time() - t0)

        if best is not None:
            epoch, train_acc, val_acc, test_acc, train_f1, test_f1 = best
            print(
                f"Run {run_idx} best epoch {epoch:03d} | "
                f"Final Test Accuracy {100 * test_acc:.2f} | Final Test F1 Macro {100 * test_f1:.2f} | "
                f"Median epoch {float(np.median(times)):.2f}s"
            )
            append_baseline_result(
                method="graphsage",
                dataset=args.dataset,
                run=run_idx,
                seed=run_seed,
                epochs=args.epochs,
                train_acc=100 * train_acc,
                valid_acc=100 * val_acc,
                test_acc=100 * test_acc,
                train_f1_macro=100 * train_f1,
                test_f1_macro=100 * test_f1,
                chosen_epoch=epoch,
            )
            all_run_test.append(test_acc)

    if all_run_test:
        arr = np.asarray(all_run_test, dtype=np.float64)
        mean = float(arr.mean())
        std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
        print(f"all_runs chosen_test_acc_mean: {mean:.6f} chosen_test_acc_std: {std:.6f}")


if __name__ == "__main__":
    main()
