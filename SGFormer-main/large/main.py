import argparse
import sys
import os, random
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import add_self_loops, coalesce, remove_self_loops, to_undirected
from torch_scatter import scatter

from logger import Logger, save_result
from dataset import load_dataset
from data_utils import normalize, gen_normalized_adjs, eval_acc, eval_rocauc, eval_f1, to_sparse_tensor, \
    load_fixed_splits, adj_mul, get_gpu_memory_map, count_parameters
from eval import evaluate
from parse import parse_method, parser_add_main_args

import time
import pickle

import warnings
warnings.filterwarnings('ignore')

# NOTE: for consistent data splits, see data_utils.rand_train_test_idx
def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def parse_bool_flag(value):
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered in {'1', 'true', 'yes', 'y'}:
        return True
    if lowered in {'0', 'false', 'no', 'n'}:
        return False
    raise ValueError(f'Invalid boolean flag value: {value}')


def prune_edges_exact(edge_index, num_nodes, keep_ratio, count_self_loops_in_budget=False):
    edge_index, _ = remove_self_loops(edge_index)
    edge_index = coalesce(edge_index, num_nodes=num_nodes)

    if keep_ratio >= 1.0:
        return edge_index, int(edge_index.size(1))

    row, col = edge_index.cpu()
    deg = torch.bincount(row, minlength=num_nodes).float()
    pair_scores = {}
    for src, dst in zip(row.tolist(), col.tolist()):
        if src == dst:
            continue
        key = (src, dst) if src < dst else (dst, src)
        if key not in pair_scores:
            src_deg = max(float(deg[key[0]].item()), 1.0)
            dst_deg = max(float(deg[key[1]].item()), 1.0)
            pair_scores[key] = 1.0 / math.sqrt(src_deg * dst_deg)

    original_offdiag = edge_index.size(1)
    if count_self_loops_in_budget:
        target_total = int(round((original_offdiag + num_nodes) * keep_ratio))
        target_offdiag = max(0, min(original_offdiag, target_total - num_nodes))
    else:
        target_offdiag = int(round(original_offdiag * keep_ratio))

    target_offdiag = max(0, min(original_offdiag, target_offdiag))
    target_offdiag = min(original_offdiag, 2 * int(round(target_offdiag / 2.0)))
    if target_offdiag >= original_offdiag:
        return edge_index, original_offdiag

    keep_pair_count = target_offdiag // 2
    ranked_pairs = sorted(
        pair_scores.items(),
        key=lambda item: (-item[1], item[0][0], item[0][1]),
    )
    kept_pairs = ranked_pairs[:keep_pair_count]
    if kept_pairs:
        kept_edges = []
        for (src, dst), _score in kept_pairs:
            kept_edges.append((src, dst))
            kept_edges.append((dst, src))
        kept_edge_index = torch.tensor(kept_edges, dtype=edge_index.dtype).t().contiguous()
    else:
        kept_edge_index = edge_index.new_empty((2, 0))

    kept_edge_index = coalesce(kept_edge_index, num_nodes=num_nodes)
    return kept_edge_index.to(edge_index.device), int(kept_edge_index.size(1))

### Parse args ###
parser = argparse.ArgumentParser(description='Training Pipeline for Node Classification')
parser_add_main_args(parser)
args = parser.parse_args()
args.count_self_loops_in_budget = parse_bool_flag(args.count_self_loops_in_budget)
print(args)

fix_seed(args.seed)

if args.cpu:
    device = torch.device("cpu")
else:
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

### Load and preprocess data ###
dataset = load_dataset(args.data_dir, args.dataset, args.sub_dataset)

if len(dataset.label.shape) == 1:
    dataset.label = dataset.label.unsqueeze(1)
dataset.label = dataset.label.to(device)

# get the splits for all runs
if args.rand_split:
    split_idx_lst = [dataset.get_idx_split(train_prop=args.train_prop, valid_prop=args.valid_prop)
                     for _ in range(args.runs)]
elif args.rand_split_class:
    split_idx_lst = [dataset.get_idx_split(split_type='class', label_num_per_class=args.label_num_per_class)
                     for _ in range(args.runs)]
elif hasattr(dataset, 'load_fixed_splits'):
    split_idx_lst = [dataset.load_fixed_splits()
                     for _ in range(args.runs)]
elif args.dataset in ['ogbn-proteins', 'ogbn-arxiv', 'ogbn-products']:
    split_idx_lst = [dataset.load_fixed_splits()
                     for _ in range(args.runs)]
else:
    split_idx_lst = load_fixed_splits(args.data_dir, dataset, name=args.dataset, protocol=args.protocol)

### Basic information of datasets ###
n = dataset.graph['num_nodes']
e = dataset.graph['edge_index'].shape[1]
# infer the number of classes for non one-hot and one-hot labels
c = max(dataset.label.max().item() + 1, dataset.label.shape[1])
d = dataset.graph['node_feat'].shape[1]

print(f"dataset {args.dataset} | num nodes {n} | num edge {e} | num node feats {d} | num classes {c}")

# whether or not to symmetrize
if not args.directed and args.dataset != 'ogbn-proteins':
    dataset.graph['edge_index'] = to_undirected(dataset.graph['edge_index'])

dataset.graph['edge_index'], _ = remove_self_loops(dataset.graph['edge_index'])
dataset.graph['edge_index'] = coalesce(dataset.graph['edge_index'], num_nodes=n)
original_offdiag = int(dataset.graph['edge_index'].size(1))
dataset.graph['edge_index'], kept_offdiag = prune_edges_exact(
    dataset.graph['edge_index'],
    n,
    args.edge_keep_ratio,
    count_self_loops_in_budget=args.count_self_loops_in_budget,
)
dataset.graph['edge_index'], _ = add_self_loops(dataset.graph['edge_index'], num_nodes=n)
print(
    f"[EdgeBudget] original_offdiag={original_offdiag} "
    f"kept_offdiag={kept_offdiag} keep_ratio={args.edge_keep_ratio} "
    f"self_loops_added={n} total_edges={int(dataset.graph['edge_index'].size(1))}"
)

dataset.graph['edge_index'], dataset.graph['node_feat'] = \
    dataset.graph['edge_index'].to(device), dataset.graph['node_feat'].to(device)

### Load method ###
model = parse_method(args, c, d, device)

### Loss function (Single-class, Multi-class) ###
if args.dataset in ('yelp-chi', 'deezer-europe', 'twitch-e', 'fb100', 'ogbn-proteins'):
    criterion = nn.BCEWithLogitsLoss()
else:
    criterion = nn.NLLLoss()

### Performance metric (Acc, AUC, F1) ###
if args.metric == 'rocauc':
    eval_func = eval_rocauc
elif args.metric == 'f1':
    eval_func = eval_f1
else:
    eval_func = eval_acc

logger = Logger(args.runs, args)

model.train()
print('MODEL:', model)

### Training loop ###
for run in range(args.runs):
    if args.dataset in ['cora', 'citeseer', 'pubmed'] and args.protocol == 'semi':
        split_idx = split_idx_lst[0]
    else:
        split_idx = split_idx_lst[run]
    train_idx = split_idx['train'].to(device)
    model.reset_parameters()
    if args.method == 'sgformer':
        optimizer = torch.optim.Adam([
            {'params': model.params1, 'weight_decay': args.trans_weight_decay},
            {'params': model.params2, 'weight_decay': args.gnn_weight_decay}
        ],
            lr=args.lr)
    else:
        optimizer = torch.optim.Adam(
            model.parameters(), weight_decay=args.weight_decay, lr=args.lr)
    best_val = float('-inf')

    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()

        train_start = time.time()
        out = model(dataset.graph['node_feat'], dataset.graph['edge_index'])
        if args.dataset in ('yelp-chi', 'deezer-europe', 'twitch-e', 'fb100', 'ogbn-proteins'):
            if dataset.label.shape[1] == 1:
                true_label = F.one_hot(dataset.label, dataset.label.max() + 1).squeeze(1)
            else:
                true_label = dataset.label
            loss = criterion(out[train_idx], true_label.squeeze(1)[
                train_idx].to(torch.float))
        else:
            out = F.log_softmax(out, dim=1)
            loss = criterion(
                out[train_idx], dataset.label.squeeze(1)[train_idx])
        loss.backward()
        optimizer.step()

        if epoch % args.eval_step == 0:
            result = evaluate(model, dataset, split_idx, eval_func, criterion, args)
            logger.add_result(run, result[:-1])

            if epoch % args.display_step == 0:
                print_str = f'Epoch: {epoch:02d}, ' + \
                            f'Loss: {loss:.4f}, ' + \
                            f'Train: {100 * result[0]:.2f}%, ' + \
                            f'Valid: {100 * result[1]:.2f}%, ' + \
                            f'Test: {100 * result[2]:.2f}%'
                print(print_str)
    logger.print_statistics(run)

logger.print_statistics()
