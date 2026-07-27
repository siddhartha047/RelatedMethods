import os.path as osp
import os
import sys
import gc
import copy
from dotmap import DotMap
import numpy as np
import scipy.sparse as sp
import torch
import time

from torch_geometric.data import Data
from torch_geometric.datasets import StochasticBlockModelDataset
from typing import Any, Callable, List, Optional, Union, Sequence
from torch import Tensor

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from .data_processor import DataProcess, DataProcess_inductive, matstd_clip
try:
    from precompute.prop import A2Prop
except ModuleNotFoundError:
    A2Prop = None

np.set_printoptions(linewidth=160, edgeitems=5, threshold=20,
                    formatter=dict(float=lambda x: "% 9.3e" % x))
torch.set_printoptions(linewidth=160, edgeitems=5)


def dmap2dct(chnname: str, dmap: DotMap, processor: DataProcess):
    typedct = {'sgc': -2, 'gbp': -3,
               'sgc_agp': 0, 'gbp_agp': 1,
               'sgc_thr': 2, 'gbp_thr': 3,}
    ctype = chnname.split('_')[0]

    dct = {}
    dct['type'] = typedct[chnname]
    dct['hop'] = dmap.hop
    dct['dim'] = processor.nfeat
    dct['delta'] = dmap.delta if type(dmap.delta) is float else 1e-5
    dct['alpha'] = dmap.alpha if (type(dmap.alpha) is float and not (ctype == 'sgc')) else 0
    dct['rra'] = (1 - dmap.rrz) if type(dmap.rrz) is float else 0
    dct['rrb'] = dmap.rrz if type(dmap.rrz) is float else 0
    return dct


def to_list(value: Any) -> Sequence:
    if isinstance(value, Sequence) and not isinstance(value, str):
        return value
    else:
        return [value]


def build_ratio_pruned_adjacency(adj_matrix: sp.csr_matrix, keep_ratio: float):
    adj = adj_matrix.tocsr().copy()
    adj.setdiag(0)
    adj.eliminate_zeros()
    adj.data = np.ones(adj.nnz, dtype=np.float32)
    adj = adj.maximum(adj.T).tocsr()
    adj.setdiag(0)
    adj.eliminate_zeros()
    adj.data = np.ones(adj.nnz, dtype=np.float32)

    coo = adj.tocoo()
    pair_mask = coo.row < coo.col
    pair_row = coo.row[pair_mask]
    pair_col = coo.col[pair_mask]
    degree = np.asarray(adj.sum(axis=1)).ravel().clip(min=1)
    score = 1.0 / np.sqrt(degree[pair_row] * degree[pair_col])

    target_pairs = max(int(round(adj.nnz * keep_ratio / 2.0)), 1)
    target_pairs = min(target_pairs, pair_row.shape[0])
    order = np.lexsort((pair_col, pair_row, -score))
    keep = order[:target_pairs]

    rows = np.concatenate((pair_row[keep], pair_col[keep]))
    cols = np.concatenate((pair_col[keep], pair_row[keep]))
    data = np.ones(rows.shape[0], dtype=np.float32)
    sparse_adj = sp.csr_matrix((data, (rows, cols)), shape=adj.shape)
    sparse_adj.setdiag(0)
    sparse_adj.eliminate_zeros()
    sparse_adj.data = np.ones(sparse_adj.nnz, dtype=np.float32)

    stats = {
        'full_directed_edges': int(adj.nnz),
        'kept_directed_edges': int(sparse_adj.nnz),
        'kept_pairs': int(target_pairs),
    }
    return sparse_adj, stats


def normalize_symmetric(adj_matrix: sp.csr_matrix):
    degree = np.asarray(adj_matrix.sum(axis=1)).ravel().astype(np.float32)
    degree[degree <= 0] = 1.0
    deg_inv_sqrt = np.power(degree, -0.5).astype(np.float32)
    dmat = sp.diags(deg_inv_sqrt, format='csr')
    return (dmat @ adj_matrix @ dmat).tocsr()


def python_precompute_embeddings(dp: DataProcess, algo: str, algo_chn: DotMap,
                                 idx: dict, edge_keep_ratio: float):
    chn = dmap2dct(algo, DotMap(algo_chn), dp)
    base_adj = dp.adj_matrix.tocsr().astype(np.float32)
    sparse_adj, stats = build_ratio_pruned_adjacency(base_adj, edge_keep_ratio)
    norm_adj = normalize_symmetric(sparse_adj)

    feat = dp.attr_matrix.astype(np.float32, copy=True)
    start = time.time()
    if chn['alpha'] == 0:
        out = feat
        for _ in range(chn['hop']):
            out = norm_adj @ out
    else:
        prev = feat
        accum = np.zeros_like(prev, dtype=np.float32)
        for _ in range(chn['hop']):
            accum += chn['alpha'] * prev
            prev = (1 - chn['alpha']) * (norm_adj @ prev)
        out = accum + prev
    time_pre = time.time() - start
    macs_pre = float(norm_adj.nnz * out.shape[1] * max(chn['hop'], 1))
    print(
        f"[PythonPrecompute] offdiag={stats['kept_directed_edges']}/{stats['full_directed_edges']} "
        f"pairs={stats['kept_pairs']} hop={chn['hop']} alpha={chn['alpha']}",
        flush=True,
    )
    return out, macs_pre, time_pre


class ContextualStochasticBlockModelDataset(StochasticBlockModelDataset):
    def __init__(
        self,
        root: str,
        block_sizes: Union[List[int], Tensor],
        edge_probs: Union[List[List[float]], Tensor],
        std: float = None,
        means: Union[List[float], Tensor] = None,
        nodes_per_mean: Union[List[int], Tensor] = None,
        num_channels: Optional[int] = None,
        is_undirected: bool = True,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        **kwargs: Any,
    ):
        self.std = std if std is not None else 1.0/np.sqrt(num_channels)
        self.means = means
        self.nodes_per_mean = nodes_per_mean
        super(ContextualStochasticBlockModelDataset, self).__init__(root, block_sizes, edge_probs, num_channels,
                         is_undirected, transform, pre_transform, **kwargs)

    @property
    def processed_file_names(self):
        block_sizes = self.block_sizes.view(-1).tolist()
        hash1 = '-'.join([f'{x:.1f}' for x in block_sizes])

        edge_probs = self.edge_probs.view(-1).tolist()
        hash2 = '-'.join([f'{x:.1f}' for x in edge_probs])

        means = self.means.view(-1).tolist()
        hash3 = '-'.join([f'{x:.1f}' for x in means])

        return f'data_{self.num_features}_{hash1}_{hash2}_{hash3}_{self.num_graphs}.pt'

    def process(self) -> None:
        from torch_geometric.utils import stochastic_blockmodel_graph, to_scipy_sparse_matrix
        import scipy.sparse as sp

        edge_index = stochastic_blockmodel_graph(
            self.block_sizes, self.edge_probs, directed=not self.is_undirected)

        num_samples = int(self.block_sizes.sum())
        num_classes = self.block_sizes.size(0)

        x = None
        if self.num_channels is not None:
            A = to_scipy_sparse_matrix(edge_index)
            A.setdiag(A.diagonal() + 1)
            d_mat = np.sum(A, axis=0)
            dinv_mat = 1/d_mat
            dinv = dinv_mat.tolist()[0]
            Dinv = sp.sparse.diags(dinv, 0)
            n = num_samples

            # Create sample from mixture of Gaussians
            X = np.zeros((2*n, self.num_channels))
            for ct, mean_ in enumerate(self.means):
                X[self.nodes_per_mean[ct],:] = self.std * np.random.randn(len(self.nodes_per_mean[ct]), self.num_channels) + mean_

            # Append a column of ones for the coefficients of the bias
            x = np.zeros((2*n,self.num_channels+1))
            x[:,:-1] = Dinv@A@X
            x[:,-1]  = np.ones(2*n)
            X = torch.tensor(x, dtype=torch.float)

        y = torch.zeros(num_samples, dtype=torch.long)
        y[:num_samples//2] = 1

        data = Data(x=x, edge_index=edge_index, y=y)

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        torch.save(self.collate([data]), self.processed_paths[0])


def sbm_mixture_of_Gaussians(n, n_features, sizes, probs, std_, means, nodes_per_mean):
    from torch_geometric.utils import to_scipy_sparse_matrix
    import scipy.sparse as sp

    # Create matrix related to the graph
    # g = nx.stochastic_block_model(sizes, probs)
    ds = StochasticBlockModelDataset(root='./data/', block_sizes=sizes, edge_probs=probs)
    g_ = ds[0]

    # A = nx.adjacency_matrix(g)
    A = to_scipy_sparse_matrix(g_.edge_index, num_nodes=g_.num_nodes)
    A.setdiag(A.diagonal() + 1)
    d_mat = np.sum(A, axis=0)
    dinv_mat = 1/d_mat
    dinv = dinv_mat.tolist()[0]
    Dinv = sp.diags(dinv, 0)

    # Create sample from mixture of Gaussians
    X = np.zeros((2*n,n_features))
    for ct, mean_ in enumerate(means):
        X[nodes_per_mean[ct],:] = std_ * np.random.randn(len(nodes_per_mean[ct]), n_features) + mean_

    # Create data

    # Append a column of ones for the coefficients of the bias
    data = np.zeros((2*n,n_features+1))
    data[:,:-1] = Dinv@A@X
    data[:,-1]  = np.ones(2*n)
    g_.x = torch.tensor(data, dtype=torch.float)

    # labels
    y = np.zeros(2*n)
    y[0:n] = 1
    g_.y = torch.tensor(y, dtype=torch.float)

    return g_


def load_csbm(datastr: str, datapath: str="./data/",
                   inductive: bool=False, multil: bool=False,
                   seed: int=0, **kwargs):
    n = 500
    # n_features = 5*int(np.ceil(2*n/(np.log(2*n)**2))) # 10*int(np.ceil(np.log(2*n)))
    n_features = 127
    q = 0.1
    _, p, mu = datastr.split('-')
    p, mu = float(p), float(mu)

    probs = [[p, q], [q, p]]
    sizes = [n, n]
    # std_ = 1/np.sqrt(n_features)
    # means = [mu, -mu]
    std_ = mu
    means = [100, -100]
    nodes_per_mean = [list(range(n)),list(range(n,2*n))]
    # Training data
    # g = ContextualStochasticBlockModelDataset(
    #     root=datapath,
    #     block_sizes=sizes,
    #     edge_probs=probs,
    #     means=means,
    #     nodes_per_mean=nodes_per_mean,
    #     num_channels=n_features,
    # )
    g = sbm_mixture_of_Gaussians(n, n_features, sizes, probs, std_, means, nodes_per_mean)

    idx_rnd = np.random.permutation(2*n)
    adj = {'train': g.edge_index,
           'test':  g.edge_index}
    feat = {'train': torch.FloatTensor(g.x),
           'test':  torch.FloatTensor(g.x)}
    idx = {'train': torch.LongTensor(idx_rnd[:int(0.6*2*n)]),
           'val':   torch.LongTensor(idx_rnd[int(0.6*2*n):int(0.8*2*n)]),
           'test':  torch.LongTensor(idx_rnd[int(0.8*2*n):])}
    y = g.y.long()
    labels = {'train': y[idx['train']],
              'val':   y[idx['val']],
              'test':  y[idx['test']]}
    nfeat = n_features + 1
    nclass = 2
    if seed >= 15:
        print(g)
    return adj, feat, labels, idx, nfeat, nclass


def load_gencat(datastr: str, datapath: str="./data/",
                   inductive: bool=False, multil: bool=False,
                   seed: int=0, **kwargs):
    from .gen_cat import gencat, feature_extraction

    dp = DataProcess('cora', path="./data/", seed=0)
    dp.input(['adjnpz', 'labels', 'attr_matrix'])
    M, D, class_size, H, node_degree = feature_extraction(dp.adj_matrix, dp.attr_matrix, dp.labels.tolist())

    _, p, omega = datastr.split('-')
    p, omega = float(p), float(omega)
    adj, feat, labels = gencat(M, D, H, n=1000, m=5000, p=p, omega=omega)
    dp.adj_matrix = adj
    dp.attr_matrix = feat
    dp.labels = np.array(labels)

    dp.calculate(['idx_train'])
    idx = {'train': torch.LongTensor(dp.idx_train),
           'val':   torch.LongTensor(dp.idx_val),
           'test':  torch.LongTensor(dp.idx_test)}
    labels = torch.LongTensor(dp.labels.flatten())
    labels = {'train': labels[idx['train']],
              'val':   labels[idx['val']],
              'test':  labels[idx['test']]}
    dp.calculate(['edge_idx'])
    adj = {'test':  torch.from_numpy(dp.edge_idx).long(),
           'train': torch.from_numpy(dp.edge_idx).long()}
    feat = {'test': torch.FloatTensor(dp.attr_matrix),
            'train': torch.FloatTensor(dp.attr_matrix)}
    n, m = dp.n, dp.m
    nfeat, nclass = dp.nfeat, dp.nclass
    if seed >= 15:
        print(dp)
    return adj, feat, labels, idx, nfeat, nclass


# ==========
def load_edgelist(datastr: str, datapath: str="./data/",
                   inductive: bool=False, multil: bool=False,
                   seed: int=0, **kwargs):
    if datastr.startswith('csbm'):
        return load_csbm(datastr, datapath, inductive, multil, seed, **kwargs)
    elif datastr.startswith('gencat'):
        return load_gencat(datastr, datapath, inductive, multil, seed, **kwargs)

    # Inductive or transductive data processor
    dp = DataProcess(datastr, path=datapath, seed=seed)
    dp.input(['adjnpz', 'labels', 'attr_matrix'])
    if inductive:
        dpi = DataProcess_inductive(datastr, path=datapath, seed=seed)
        dpi.input(['adjnpz', 'attr_matrix'])
    else:
        dpi = dp
    # Get index
    if (datastr.startswith('cora') or datastr.startswith('citeseer') or datastr.startswith('pubmed')):
        dp.calculate(['idx_train'])
    else:
        dp.input(['idx_train', 'idx_val', 'idx_test'])
    idx = {'train': torch.LongTensor(dp.idx_train),
           'val':   torch.LongTensor(dp.idx_val),
           'test':  torch.LongTensor(dp.idx_test)}

    # Get label
    if multil:
        dp.calculate(['labels_oh'])
        dp.labels_oh[dp.labels_oh < 0] = 0
        labels = torch.LongTensor(dp.labels_oh).float()
    else:
        dp.labels[dp.labels < 0] = 0
        labels = torch.LongTensor(dp.labels.flatten())
    labels = {'train': labels[idx['train']],
              'val':   labels[idx['val']],
              'test':  labels[idx['test']]}

    # Get edge index
    # dp.adj_matrix.setdiag(0)
    # dp.adj_matrix.eliminate_zeros()
    dp.calculate(['edge_idx'])
    adj = {'test':  torch.from_numpy(dp.edge_idx).long()}
    if inductive:
        dpi.calculate(['edge_idx'])
        adj['train'] = torch.from_numpy(dpi.edge_idx).long()
    else:
        adj['train'] = adj['test']
    # Get node attributes
    feat = dp.attr_matrix
    feati = dpi.attr_matrix if inductive else feat

    # from sklearn.preprocessing import StandardScaler
    # scaler = StandardScaler(with_mean=False)
    # scaler.fit(feati)
    # feat = scaler.transform(feat)
    # feati = scaler.transform(feati)

    feat = {'test': torch.FloatTensor(feat)}
    feat['train'] = torch.FloatTensor(feati)

    # Get graph property
    n, m = dp.n, dp.m
    nfeat, nclass = dp.nfeat, dp.nclass
    if seed >= 15:
        print(dp)
    return adj, feat, labels, idx, nfeat, nclass


def load_embedding(datastr: str, algo: str, algo_chn: DotMap,
                   datapath: str="./data/",
                   inductive: bool=False, multil: bool=False,
                   seed: int=0, **kwargs):
    # Inductive or transductive data processor
    dp = DataProcess(datastr, path=datapath, seed=seed)
    dp.input(['adjnpz', 'labels', 'attr_matrix', 'deg'])
    if inductive:
        dpi = DataProcess_inductive(datastr, path=datapath, seed=seed)
        dpi.input(['adjnpz', 'attr_matrix', 'deg'])
    else:
        dpi = dp
    # Get index
    if (datastr.startswith('cora') or datastr.startswith('citeseer') or datastr.startswith('pubmed')):
        dp.calculate(['idx_train'])
    else:
        dp.input(['idx_train', 'idx_val', 'idx_test'])
    idx = {'train': torch.LongTensor(dp.idx_train),
           'val':   torch.LongTensor(dp.idx_val),
           'test':  torch.LongTensor(dp.idx_test)}

    # Get label
    if multil:
        dp.calculate(['labels_oh'])
        dp.labels_oh[dp.labels_oh < 0] = 0
        labels = torch.LongTensor(dp.labels_oh).float()
    else:
        dp.labels[dp.labels < 0] = 0
        labels = torch.LongTensor(dp.labels.flatten())
    labels = {'train': labels[idx['train']],
              'val':   labels[idx['val']],
              'test':  labels[idx['test']]}
    # Get graph property
    n, m = dp.n, dp.m
    nfeat, nclass = dp.nfeat, dp.nclass
    if seed >= 15:
        print(dp)

    edge_keep_ratio = kwargs.get('edge_keep_ratio')
    if edge_keep_ratio is not None:
        feat, macs_pre, time_pre = python_precompute_embeddings(dp, algo, algo_chn, idx, edge_keep_ratio)
    else:
        if A2Prop is None:
            raise ModuleNotFoundError(
                "precompute.prop is unavailable. Set edge_keep_ratio in the config to use the Python fallback."
            )
        py_a2prop = A2Prop()
        py_a2prop.load(os.path.join(datapath, datastr), m, n, seed)
        chn = dmap2dct(algo, DotMap(algo_chn), dp)

        feat = dp.attr_matrix.transpose().astype(np.float32, order='C')

        macs_pre, time_pre = py_a2prop.compute(1, [chn], feat)
        if not ('_'  in algo):
            if seed >= 15:
                print(f"[Pre] MACs Real: {macs_pre/1e9}, full: {chn['hop'] * m * nfeat/1e9}, ", end='')
                print(f"numA Real: {macs_pre/nfeat/1e3}, full: {chn['hop'] * m /1e3}")
        feat = feat.transpose()

    feat = matstd_clip(feat, idx['train'], with_mean=True)
    feats = {'val':  torch.FloatTensor(feat[idx['val']]),
             'test': torch.FloatTensor(feat[idx['test']])}
    feats['train'] = torch.FloatTensor(feat[idx['train']])
    del feat
    gc.collect()
    # print(feats['train'].shape)
    return feats, labels, idx, nfeat, nclass, macs_pre/1e9, time_pre
