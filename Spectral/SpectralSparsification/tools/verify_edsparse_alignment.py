#!/usr/bin/env python3
"""Verify ER edge order/topology against EDSparse and Scaffold loaders."""

from __future__ import annotations

import argparse
import gc
import hashlib
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from export_edsparse_graph import (  # noqa: E402
    ALL_DATASETS,
    DEFAULT_DATA_ROOT,
    DEFAULT_EDSPARSE_ROOT,
    artifact_directory,
)
from tgt_binary import _decode, _read_small, _shape  # noqa: E402


DEFAULT_SCAFFOLD_ROOT = Path(
    "/people/dass304/dass304/Support Graph/ICML_SPARSIFICATION"
)


def directed_fingerprint(num_nodes: int, edge_index: torch.Tensor) -> str:
    """Hash PyG columns in bounded memory, exactly like the exporter."""

    edges = torch.as_tensor(edge_index).detach().cpu().long().contiguous().numpy()
    if edges.ndim != 2 or edges.shape[0] != 2:
        raise ValueError(f"edge_index must have shape [2, E], got {edges.shape}")
    digest = hashlib.sha256()
    digest.update(b"edsparse-directed-edge-order-v1\0")
    digest.update(np.asarray([num_nodes], dtype="<i8").tobytes())
    chunk_size = 1_048_576
    for row in range(2):
        for first in range(0, edges.shape[1], chunk_size):
            chunk = np.asarray(edges[row, first : first + chunk_size], dtype="<i8")
            digest.update(memoryview(chunk).cast("B"))
    return digest.hexdigest()


def artifact_contract(data_root: Path, dataset: str) -> dict[str, object]:
    directory = artifact_directory(data_root, dataset)
    graph_path = directory / "input_graph.npz"
    directed_path = directory / "effective_resistance_directed.npz"
    canonical_path = directory / "effective_resistance.npz"
    with zipfile.ZipFile(graph_path, "r") as graph:
        contract = {
            "num_nodes": int(_read_small(graph, "num_nodes").reshape(-1)[0]),
            "num_canonical_edges": int(_shape(graph, "src")[0]),
            "num_directed_edges": int(
                _read_small(graph, "num_directed_edges").reshape(-1)[0]
            ),
            "directed_fingerprint": _decode(
                _read_small(graph, "directed_fingerprint")
            ),
            "graph_path": graph_path,
        }
    with zipfile.ZipFile(directed_path, "r") as directed:
        directed_shape = _shape(directed, "resistance")
        artifact_directed_fingerprint = _decode(
            _read_small(directed, "directed_fingerprint")
        )
    with zipfile.ZipFile(canonical_path, "r") as canonical:
        canonical_shape = _shape(canonical, "resistance")
    if directed_shape != (contract["num_directed_edges"],):
        raise ValueError(
            f"directed artifact shape {directed_shape} does not match export contract"
        )
    if canonical_shape != (contract["num_canonical_edges"],):
        raise ValueError(
            f"canonical artifact shape {canonical_shape} does not match export contract"
        )
    if artifact_directed_fingerprint != contract["directed_fingerprint"]:
        raise ValueError("directed artifact and input export fingerprints differ")
    return contract


def canonical_keys(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    edges = torch.as_tensor(edge_index).detach().cpu().long()
    if edges.ndim != 2 or edges.size(0) != 2:
        raise ValueError(f"edge_index must have shape [2, E], got {tuple(edges.shape)}")
    source, target = edges
    non_loop = source != target
    low = torch.minimum(source[non_loop], target[non_loop])
    high = torch.maximum(source[non_loop], target[non_loop])
    return torch.unique(low * int(num_nodes) + high, sorted=True)


def canonical_topology_matches(
    edge_index: torch.Tensor,
    contract: dict[str, object],
) -> bool:
    num_nodes = int(contract["num_nodes"])
    keys = canonical_keys(edge_index, num_nodes)
    if keys.numel() != int(contract["num_canonical_edges"]):
        return False
    with np.load(Path(contract["graph_path"]), allow_pickle=False) as graph:
        src = np.asarray(graph["src"], dtype=np.int64)
        dst = np.asarray(graph["dst"], dtype=np.int64)
    chunk_size = 1_048_576
    for first in range(0, keys.numel(), chunk_size):
        last = min(keys.numel(), first + chunk_size)
        chunk = keys[first:last]
        if not torch.equal(
            torch.div(chunk, num_nodes, rounding_mode="floor"),
            torch.from_numpy(src[first:last]),
        ) or not torch.equal(
            chunk.remainder(num_nodes),
            torch.from_numpy(dst[first:last]),
        ):
            return False
    return True


def load_edsparse(root: Path, data_root: Path, dataset: str):
    sys.path.insert(0, str(root))
    try:
        from edsparse.data import load_dataset

        return load_dataset(
            data_root,
            dataset,
            seed=42,
            split_protocol="tunedgnn",
        ).data
    finally:
        sys.path.pop(0)


def load_scaffold(root: Path, data_root: Path, dataset: str):
    sys.path.insert(0, str(root.parent))
    try:
        from ICML_SPARSIFICATION.utils.dataset import load_dataset

        loaded = load_dataset(str(data_root), dataset)
        graph, _label = loaded[0]
        return torch.as_tensor(graph["edge_index"]).long(), int(graph["num_nodes"])
    finally:
        sys.path.pop(0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", nargs="+", choices=ALL_DATASETS)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--edsparse-root", type=Path, default=DEFAULT_EDSPARSE_ROOT)
    parser.add_argument("--scaffold-root", type=Path, default=DEFAULT_SCAFFOLD_ROOT)
    parser.add_argument(
        "--source", choices=("both", "edsparse", "scaffold"), default="both"
    )
    args = parser.parse_args()
    datasets = args.dataset or list(ALL_DATASETS)
    failures = 0

    for index, dataset in enumerate(datasets, start=1):
        print(
            f"VERIFY_START index={index}/{len(datasets)} dataset={dataset}",
            flush=True,
        )
        contract = artifact_contract(args.data_root, dataset)
        if args.source in ("both", "edsparse"):
            data = load_edsparse(args.edsparse_root, args.data_root, dataset)
            edge_index = data.edge_index
            edge_count = int(edge_index.size(1))
            exact = (
                int(data.num_nodes) == int(contract["num_nodes"])
                and edge_count == int(contract["num_directed_edges"])
                and directed_fingerprint(int(data.num_nodes), edge_index)
                == contract["directed_fingerprint"]
            )
            print(
                f"VERIFY_EDSPARSE dataset={dataset} directed_edges={edge_count} "
                f"exact_order={str(exact).lower()}",
                flush=True,
            )
            failures += int(not exact)
            del data, edge_index
            gc.collect()

        if args.source in ("both", "scaffold"):
            edge_index, num_nodes = load_scaffold(
                args.scaffold_root, args.data_root, dataset
            )
            direct_exact = (
                num_nodes == int(contract["num_nodes"])
                and edge_index.size(1) == int(contract["num_directed_edges"])
                and directed_fingerprint(num_nodes, edge_index)
                == contract["directed_fingerprint"]
            )
            topology_exact = (
                num_nodes == int(contract["num_nodes"])
                and canonical_topology_matches(edge_index, contract)
            )
            print(
                f"VERIFY_SCAFFOLD dataset={dataset} "
                f"input_edges={int(edge_index.size(1))} "
                f"direct_order={str(direct_exact).lower()} "
                f"canonical_topology={str(topology_exact).lower()}",
                flush=True,
            )
            failures += int(not topology_exact)
            del edge_index
            gc.collect()

    print(
        f"VERIFY_DONE datasets={len(datasets)} failures={failures}",
        flush=True,
    )
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
