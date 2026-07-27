import os

import numpy as np
from torch_geometric.datasets import Planetoid

from utils.data_processor import DataProcess, edgeidx2adj


def main():
    dataset = Planetoid(root="./data", name="Cora")
    data = dataset[0]

    dp = DataProcess("smallcora", path="./data")
    dp.adj_matrix = edgeidx2adj(
        data.edge_index[0].cpu().numpy(),
        data.edge_index[1].cpu().numpy(),
        data.num_nodes,
    )
    dp.attr_matrix = data.x.cpu().numpy().astype(np.float32)
    dp.labels = data.y.cpu().numpy()
    dp.idx_train = data.train_mask.nonzero(as_tuple=False).view(-1).cpu().numpy()
    dp.idx_val = data.val_mask.nonzero(as_tuple=False).view(-1).cpu().numpy()
    dp.idx_test = data.test_mask.nonzero(as_tuple=False).view(-1).cpu().numpy()

    os.makedirs("./data/smallcora", exist_ok=True)
    dp.calculate(['deg'])
    dp.output(['adjnpz', 'adjl', 'attribute', 'deg', 'labels', 'attr_matrix'])

    print("Prepared ./data/smallcora")
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
