import pdb
import time
import random
random.seed(42)
import numpy as np
import torch
import dspar.cpp_extension.sampler as sampler
from torch_geometric.utils import degree, to_undirected


def _exact_weighted_edge_sample(probabilities, budget, seed):
    """Sample an exact number of unique edge indices without replacement."""

    edge_count = probabilities.numel()
    if budget >= edge_count:
        indices = torch.arange(edge_count, device=probabilities.device)
    else:
        generator_device = (
            probabilities.device if probabilities.is_cuda else "cpu"
        )
        generator = torch.Generator(device=generator_device)
        generator.manual_seed(seed)
        indices = torch.multinomial(
            probabilities,
            budget,
            replacement=False,
            generator=generator,
        )
    counts = torch.ones(
        indices.numel(),
        dtype=probabilities.dtype,
        device=probabilities.device,
    )
    return indices, counts


def maybe_sparsfication(data, dataset, follow_by_subgraph_sampling, random=False, is_undirected=True, reweighted=True, target_ratio=None):
    N, E = data.num_nodes, data.num_edges
    if target_ratio is not None and not 0.0 < float(target_ratio) <= 1.0:
        raise ValueError("target_ratio must be in (0, 1]")
    src, dst = data.edge_index
    epsilon = 0.25
    if dataset == 'ogbn-arxiv':
        epsilon = 0.25 if not random else 0.35
    elif dataset == 'reddit2':
        epsilon = 0.3 if not random else 0.32
    elif dataset == 'ogbn-products':
        epsilon = 0.4 if not random else 0.45
    elif dataset == 'yelp':
        epsilon = 0.5 if not random else 0.6
    elif dataset == 'ogbn-proteins':
        epsilon = 0.25

    if follow_by_subgraph_sampling and dataset == 'ogbn-products':
        epsilon = 0.15 if not random else 0.2

    print(f'epsilon: {epsilon}')
    if target_ratio is None:
        Q = int(0.16 * N * np.log(N) / epsilon ** 2)
    else:
        Q = max(1, int(float(target_ratio) * E))
    print(f"Q: {Q}")
    print(f'E/Q ratio: {E/Q}')
    print(f'E/nlogn ratio: {E/N/np.log(N)}')
    print('sparsify the input graph')
    data = data.clone()
    s = time.time()
    if random:
        pe = torch.ones(size=(E,), dtype=torch.double) / E
    else:
        print('sparsify the graph by degrees')
        node_degree = degree(dst, data.num_nodes)
        di, dj = torch.nan_to_num(1. / node_degree[src]), torch.nan_to_num(1. / node_degree[dst])
        pe = (di + dj).double()
        pe = pe / torch.sum(pe)
    print(f'cal edge distribution used {time.time() - s} sec')
    # For reproducibility, we manually set the seed of graph sparsification to 42. We note that this seed is only effective for the graph sparsification, 
    # it does not impact any following process.
    seed_val = 42
    s = time.time()
    if target_ratio is None:
        p_cumsum = torch.cumsum(pe, 0)
        sampled = sampler.edge_sample(p_cumsum, Q, seed_val)
        e_indices, e_cnt = torch.unique(sampled, return_counts=True)
    else:
        # The benchmark API defines target_ratio as a retained-edge budget.
        # Sampling with replacement can undershoot it after duplicate removal,
        # so explicit targets use weighted sampling without replacement.
        e_indices, e_cnt = _exact_weighted_edge_sample(pe, Q, seed_val)
    print(f'sample edge used {time.time() - s} sec')
    new_graph = e_cnt / Q / pe[e_indices]
    new_src, new_dst = src[e_indices], dst[e_indices]
    edge_index = torch.cat([new_src.view(1, -1), new_dst.view(1, -1)], dim=0)
    edge_attr = new_graph.float()
    if is_undirected and target_ratio is None:
        data.edge_index, data.edge_attr = to_undirected(edge_index, edge_attr)
    else:
        # Do not expand explicitly budgeted samples with reverse edges: that
        # would make the graph used for training exceed the requested ratio.
        data.edge_index, data.edge_attr = edge_index, edge_attr
    if not reweighted:
        print('not reweight')
        data.edge_attr = None
    actual_edges = int(data.num_edges)
    actual_ratio = actual_edges / E
    data.dspar_original_num_edges = int(E)
    data.dspar_target_num_edges = int(Q)
    data.dspar_actual_num_edges = actual_edges
    data.dspar_actual_kept_ratio = actual_ratio
    print(
        f'before sparsification, num_edges: {E}, '
        f'after sparsification, num_edges: {actual_edges}, '
        f'ratio: {actual_ratio}'
    )
    return data
