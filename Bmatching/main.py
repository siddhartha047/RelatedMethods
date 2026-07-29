#!/usr/bin/env python3
"""Load an EDSparse dataset and demonstrate b-matching sparsification."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from bmatching_adapter import solve_bmatching


HERE = Path(__file__).resolve().parent
SUPPORT_GRAPH_ROOT = HERE.parents[1]
EDSPARSE_ROOT = SUPPORT_GRAPH_ROOT / "EDSparse"
if str(EDSPARSE_ROOT) not in sys.path:
    sys.path.insert(0, str(EDSPARSE_ROOT))

from edsparse.core.graph import canonicalize_undirected  # noqa: E402
from edsparse.data.connector import load_pyg_data  # noqa: E402
from edsparse.models.weights import compute_edge_scores  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="karate")
    parser.add_argument(
        "--data-root",
        default="/rcfs/scratch/dass304/EDSparse/data",
    )
    parser.add_argument("--target-ratio", type=float, default=0.5)
    parser.add_argument(
        "--weight-method",
        choices=("cosine", "euclidean", "dot", "uniform", "edge_attr"),
        default="cosine",
    )
    parser.add_argument(
        "--backend",
        choices=("auto", "legacy", "scalable"),
        default="auto",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--legacy-max-nodes", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    data, bundle = load_pyg_data(
        args.data_root,
        args.dataset,
        seed=args.seed,
        split_protocol="tunedgnn",
    )
    graph, _, _ = canonicalize_undirected(data.edge_index, int(data.num_nodes))
    scores = compute_edge_scores(data, graph, args.weight_method)
    result = solve_bmatching(
        graph.edge_index,
        scores,
        graph.num_nodes,
        target_ratio=args.target_ratio,
        backend=args.backend,
        device=args.device,
        num_workers=args.workers,
        legacy_max_nodes=args.legacy_max_nodes,
    )

    selected_edges = graph.edge_index[:, result.selected_indices]
    selected_degree = torch.bincount(
        selected_edges.reshape(-1), minlength=graph.num_nodes
    )
    capacity_ok = bool((selected_degree <= result.capacities).all())
    score_sum = float(scores[result.selected_indices].sum())
    print(
        "\nBMatching demonstration complete\n"
        f"dataset={bundle.name}\n"
        f"nodes={graph.num_nodes} original_undirected_edges={graph.num_edges}\n"
        f"requested_edges={result.requested_edges} "
        f"selected_edges={result.selected_edges} "
        f"realized_ratio={result.selected_edges / max(1, graph.num_edges):.6f}\n"
        f"backend={result.backend} legacy_solver_ran={result.legacy_solver_ran}\n"
        f"ordering_device={result.ordering_device} workers={result.workers}\n"
        f"capacity_constraints_satisfied={capacity_ok} "
        f"selected_weight_sum={score_sum:.6f}\n"
        f"elapsed_seconds={result.elapsed_seconds:.6f}",
        flush=True,
    )
    if result.message:
        print(f"note={result.message}", flush=True)


if __name__ == "__main__":
    main()
