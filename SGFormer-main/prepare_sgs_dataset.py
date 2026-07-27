import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch_geometric.datasets import Amazon, Coauthor, HeterophilousGraphDataset, LINKXDataset, Planetoid
from torch_geometric.utils import is_undirected, to_undirected


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


def select_sgs_masks(data):
    if hasattr(data, "train_mask") and hasattr(data, "val_mask") and hasattr(data, "test_mask"):
        train_mask = data.train_mask
        val_mask = data.val_mask
        test_mask = data.test_mask
        if train_mask.dim() > 1 and val_mask.dim() > 1 and test_mask.dim() > 1:
            split_index = 2
            return train_mask[:, split_index], val_mask[:, split_index], test_mask[:, split_index]
        return train_mask.bool(), val_mask.bool(), test_mask.bool()

    labels = data.y.view(-1).cpu().numpy() if hasattr(data, "y") else None
    return build_sgs_split(data.num_nodes, labels=labels)


def load_sgs_dataset(dataset_name: str):
    if dataset_name == "SmallCora":
        return Planetoid(root="./data", name="Cora")
    if dataset_name in {"Photo", "Computers"}:
        return Amazon(root="./data", name=dataset_name)
    if dataset_name == "CS":
        return Coauthor(root="./data", name="CS")
    if dataset_name in {"johnshopkins55", "amherst41"}:
        return LINKXDataset(root="./data", name=dataset_name)
    if dataset_name == "Tolokers":
        return HeterophilousGraphDataset(root="./data", name="Tolokers")
    raise ValueError(f"Unsupported SGS dataset for SGFormer: {dataset_name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    save_dir = repo_root / "data_saved"
    save_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_sgs_dataset(args.dataset)
    data = dataset[0]
    if not is_undirected(data.edge_index):
        data.edge_index = to_undirected(data.edge_index)
    train_mask, val_mask, test_mask = select_sgs_masks(data)

    payload = {
        "edge_index": data.edge_index.clone().cpu(),
        "x": data.x.clone().cpu(),
        "y": data.y.clone().cpu(),
        "train_mask": train_mask.clone().cpu(),
        "val_mask": val_mask.clone().cpu(),
        "test_mask": test_mask.clone().cpu(),
    }
    out_path = save_dir / f"{args.dataset}.pt"
    torch.save(payload, out_path)

    print(
        "Prepared",
        out_path,
        "stats",
        data.num_nodes,
        int(data.edge_index.size(1)),
        int(data.x.size(1)),
        int(torch.unique(data.y[data.y != -1]).numel()),
        int(train_mask.sum()),
        int(val_mask.sum()),
        int(test_mask.sum()),
    )


if __name__ == "__main__":
    main()
