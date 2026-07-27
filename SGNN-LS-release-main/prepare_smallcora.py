from pathlib import Path

import torch
from torch_geometric.datasets import Planetoid


def main():
    repo_root = Path(__file__).resolve().parent
    data_root = repo_root / "data"
    save_dir = repo_root / "data_saved"
    save_dir.mkdir(parents=True, exist_ok=True)

    dataset = Planetoid(root=str(data_root), name="Cora")
    data = dataset[0]

    payload = {
        "edge_index": data.edge_index.clone().cpu(),
        "train_mask": data.train_mask.clone().cpu().to(torch.bool),
        "val_mask": data.val_mask.clone().cpu().to(torch.bool),
        "test_mask": data.test_mask.clone().cpu().to(torch.bool),
    }
    out_path = save_dir / "SmallCora.pt"
    torch.save(payload, out_path)

    print(
        "Prepared",
        out_path,
        "stats",
        data.num_nodes,
        int(data.edge_index.size(1)),
        dataset.num_features,
        dataset.num_classes,
        int(data.train_mask.sum()),
        int(data.val_mask.sum()),
        int(data.test_mask.sum()),
    )


if __name__ == "__main__":
    main()
