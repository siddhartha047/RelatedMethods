import argparse
import os

import numpy as np
from sklearn.model_selection import train_test_split
from torch_geometric.datasets import Amazon, Coauthor, HeterophilousGraphDataset, LINKXDataset, Planetoid
from torch_geometric.utils import is_undirected, to_undirected

from utils.data_processor import DataProcess, edgeidx2adj


def build_sgs_split(num_nodes: int, labels=None, seed: int = 1):
    if labels is not None:
        labels = np.asarray(labels).reshape(-1)
        indices = np.where(labels != -1)[0]
    else:
        indices = np.arange(num_nodes)
    train_idx, temp_idx = train_test_split(indices, test_size=0.8, random_state=seed)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=seed)
    return train_idx, val_idx, test_idx


def load_sgs_dataset(dataset_name: str):
    if dataset_name == "SmallCora":
        dataset = Planetoid(root="./data", name="Cora")
        export_name = "smallcora"
    elif dataset_name in {"Photo", "Computers"}:
        dataset = Amazon(root="./data", name=dataset_name)
        export_name = dataset_name.lower()
    elif dataset_name == "CS":
        dataset = Coauthor(root="./data", name="CS")
        export_name = "cs"
    elif dataset_name in {"johnshopkins55", "amherst41"}:
        dataset = LINKXDataset(root="./data", name=dataset_name)
        export_name = dataset_name.lower()
    elif dataset_name == "Tolokers":
        dataset = HeterophilousGraphDataset(root="./data", name="Tolokers")
        export_name = "tolokers"
    else:
        raise ValueError(f"Unsupported SGS dataset: {dataset_name}")
    return dataset, export_name


def select_sgs_indices(data):
    if hasattr(data, "train_mask") and hasattr(data, "val_mask") and hasattr(data, "test_mask"):
        train_mask = data.train_mask
        val_mask = data.val_mask
        test_mask = data.test_mask
        if train_mask.dim() > 1 and val_mask.dim() > 1 and test_mask.dim() > 1:
            split_index = 2
            train_mask = train_mask[:, split_index]
            val_mask = val_mask[:, split_index]
            test_mask = test_mask[:, split_index]
        train_idx = train_mask.nonzero(as_tuple=False).view(-1).cpu().numpy()
        val_idx = val_mask.nonzero(as_tuple=False).view(-1).cpu().numpy()
        test_idx = test_mask.nonzero(as_tuple=False).view(-1).cpu().numpy()
        return train_idx, val_idx, test_idx
    labels = data.y.view(-1).cpu().numpy() if hasattr(data, "y") else None
    return build_sgs_split(data.num_nodes, labels=labels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    args = parser.parse_args()

    dataset, export_name = load_sgs_dataset(args.dataset)
    data = dataset[0]
    if not is_undirected(data.edge_index):
        data.edge_index = to_undirected(data.edge_index)

    dp = DataProcess(export_name, path="./data")
    dp.adj_matrix = edgeidx2adj(
        data.edge_index[0].cpu().numpy(),
        data.edge_index[1].cpu().numpy(),
        data.num_nodes,
    )
    dp.attr_matrix = data.x.cpu().numpy().astype(np.float32)
    dp.labels = data.y.view(-1).cpu().numpy()
    dp.idx_train, dp.idx_val, dp.idx_test = select_sgs_indices(data)

    os.makedirs(f"./data/{export_name}", exist_ok=True)
    dp.calculate(["deg"])
    dp.output(["adjnpz", "adjl", "attribute", "deg", "labels", "attr_matrix"])

    print(f"Prepared ./data/{export_name}")
    print(
        "stats",
        dp.n,
        dp.m,
        dp.nfeat,
        dp.nclass,
        len(dp.idx_train),
        len(dp.idx_val),
        len(dp.idx_test),
    )


if __name__ == "__main__":
    main()
