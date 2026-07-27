import argparse
import copy
import math
import os
import random
import sys
import warnings
import time, subprocess

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from data_utils import class_rand_splits, eval_acc, eval_rocauc, evaluate, load_fixed_splits, class_rand_splits, to_sparse_tensor
from dataset import load_nc_dataset
from logger import Logger
from parse import parse_method, parser_add_default_args, parser_add_main_args
from torch_geometric.utils import (add_self_loops, remove_self_loops,
                                   to_undirected)

warnings.filterwarnings('ignore')

# NOTE: for consistent data splits, see data_utils.rand_train_test_idx
def get_gpu_memory_map():
    """Get the current gpu usage.
    Returns
    -------
    usage: dict
        Keys are device ids as integers.
        Values are memory usage as integers in MB.
    """
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
    # Convert lines into a dictionary
    gpu_memory = np.array([int(x) for x in result.strip().split('\n')])
    # gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
    return gpu_memory

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
    edge_index = edge_index.cpu()
    edge_index = torch.unique(edge_index, dim=1)

    if keep_ratio >= 1.0:
        return edge_index, int(edge_index.size(1))

    row, col = edge_index
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
    kept_edges = []
    for (src, dst), _score in kept_pairs:
        kept_edges.append((src, dst))
        kept_edges.append((dst, src))
    if kept_edges:
        kept_edge_index = torch.tensor(kept_edges, dtype=edge_index.dtype).t().contiguous()
        kept_edge_index = torch.unique(kept_edge_index, dim=1)
    else:
        kept_edge_index = edge_index.new_empty((2, 0))
    return kept_edge_index, int(kept_edge_index.size(1))


### Parse args ###
parser = argparse.ArgumentParser(description='General Training Pipeline')
parser_add_main_args(parser)
args = parser.parse_args()
parser_add_default_args(args)
args.count_self_loops_in_budget = parse_bool_flag(args.count_self_loops_in_budget)
print(args)

fix_seed(args.seed)

if args.cpu:
    device = torch.device("cpu")
else:
    device = torch.device("cuda:" + str(args.device)
                          ) if torch.cuda.is_available() else torch.device("cpu")

### Load and preprocess data ###
dataset = load_nc_dataset(args)

if len(dataset.label.shape) == 1:
    dataset.label = dataset.label.unsqueeze(1)

dataset_name = args.dataset

if args.rand_split:
    split_idx_lst = [dataset.get_idx_split(train_prop=args.train_prop, valid_prop=args.valid_prop)
                     for _ in range(args.runs)]
elif args.rand_split_class:
    split_idx_lst = [class_rand_splits(
        dataset.label, args.label_num_per_class, args.valid_num, args.test_num)]
else:
    split_idx_lst = load_fixed_splits(
        dataset, name=args.dataset, protocol=args.protocol)

dataset.label = dataset.label.to(device)

n = dataset.graph['num_nodes']
# infer the number of classes for non one-hot and one-hot labels
c = max(dataset.label.max().item() + 1, dataset.label.shape[1])
d = dataset.graph['node_feat'].shape[1]

_shape = dataset.graph['node_feat'].shape
print(f'features shape={_shape}')

# whether or not to symmetrize
if args.dataset not in {'deezer-europe'}:
    dataset.graph['edge_index'] = to_undirected(dataset.graph['edge_index'])

dataset.graph['edge_index'], _ = remove_self_loops(dataset.graph['edge_index'])
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
    dataset.graph['edge_index'].to(
        device), dataset.graph['node_feat'].to(device)

if args.method == 'graphormer':
    dataset.graph['x'] = dataset.graph['x'].to(device)
    dataset.graph['in_degree'] = dataset.graph['in_degree'].to(device)
    dataset.graph['out_degree'] = dataset.graph['out_degree'].to(device)
    dataset.graph['spatial_pos'] = dataset.graph['spatial_pos'].to(device)
    dataset.graph['attn_bias'] = dataset.graph['attn_bias'].to(device)

print(f"num nodes {n} | num classes {c} | num node feats {d}")

### Load method ###
model = parse_method(args.method, args, c, d, device)

# using rocauc as the eval function
if args.dataset in ('deezer-europe'):
    criterion = nn.BCEWithLogitsLoss()
else:
    criterion = nn.NLLLoss()

eval_func = eval_acc

logger = Logger(args.runs, args)

model.train()

### Training loop ###
patience = 0
if args.method == 'ours' and args.use_graph:
    optimizer = torch.optim.Adam([
        {'params': model.params1, 'weight_decay': args.ours_weight_decay},
        {'params': model.params2, 'weight_decay': args.weight_decay}
    ],
        lr=args.lr)
else:
    optimizer = torch.optim.Adam(
        model.parameters(), weight_decay=args.weight_decay, lr=args.lr)

run_time_list = []

for run in range(args.runs):
    if args.dataset in ['cora', 'citeseer', 'pubmed', 'SmallCora'] and args.protocol == 'semi':
        split_idx = split_idx_lst[0]
    else:
        split_idx = split_idx_lst[run]
    train_idx = split_idx['train'].to(device)
    model.reset_parameters()

    best_val = float('-inf')
    patience = 0
    for epoch in range(args.epochs):
        start_time = time.perf_counter()
        model.train()
        optimizer.zero_grad()
        emb = None
        if args.method == 'nodeformer':
            out, link_loss_ = model(dataset)
        else:
            out = model(dataset)
        
        if args.dataset in ('deezer-europe'):
            if dataset.label.shape[1] == 1:
                true_label = F.one_hot(
                    dataset.label, dataset.label.max() + 1).squeeze(1)
            else:
                true_label = dataset.label
            loss = criterion(out[train_idx], true_label.squeeze(1)[
                train_idx].to(torch.float))
        else:
            if args.method == 'graphormer':
                out = out.squeeze(0)
            out = F.log_softmax(out, dim=1)
            loss = criterion(
                out[train_idx], dataset.label.squeeze(1)[train_idx])
                
        if args.method == 'nodeformer':
            loss -= args.lamda * sum(link_loss_) / len(link_loss_)
        loss.backward()
        optimizer.step()
        end_time = time.perf_counter()
        run_time = 1000 * (end_time - start_time)
        run_time_list.append(run_time)

        result = evaluate(model, dataset, split_idx,
                          eval_func, criterion, args)
        logger.add_result(run, result[:-1])

        if result[1] > best_val:
            best_val = result[1]
            patience = 0
        else:
            patience += 1
            if patience >= args.patience:
                break

        if epoch % args.display_step == 0:
            print(f'Epoch: {epoch:02d}, '
                  f'Loss: {loss:.4f}, '
                  f'Train: {100 * result[0]:.2f}%, '
                  f'Valid: {100 * result[1]:.2f}%, '
                  f'Test: {100 * result[2]:.2f}%')
    logger.print_statistics(run)

run_time = sum(run_time_list) / len(run_time_list)
results = logger.print_statistics()
print(results)
out_folder = 'results'
if not os.path.exists(out_folder):
    os.mkdir(out_folder)

def make_print(method):
    print_str = ''
    if args.rand_split_class:
        print_str += f'label per class:{args.label_num_per_class}, valid:{args.valid_num},test:{args.test_num}\n'
    else:
        print_str += f'train_prop:{args.train_prop}, valid_prop:{args.valid_prop}'
    if method == 'ours':
        use_weight=' ours_use_weight' if args.ours_use_weight else ''
        print_str += f'method: {args.method} hidden: {args.hidden_channels} ours_layers:{args.ours_layers} lr:{args.lr} use_graph:{args.use_graph} aggregate:{args.aggregate} graph_weight:{args.graph_weight} alpha:{args.alpha} ours_decay:{args.ours_weight_decay} ours_dropout:{args.ours_dropout} epochs:{args.epochs} use_feat_norm:{not args.no_feat_norm} use_bn:{args.use_bn} use_residual:{args.ours_use_residual} use_act:{args.ours_use_act}{use_weight}\n'
        if not args.use_graph:
            return print_str
        if args.backbone == 'gcn':
            print_str += f'backbone:{args.backbone}, layers:{args.num_layers} hidden: {args.hidden_channels} lr:{args.lr} decay:{args.weight_decay} dropout:{args.dropout}\n'
    else:
        print_str += f'method: {args.method} hidden: {args.hidden_channels} lr:{args.lr}\n'
    return print_str


file_name = f'{args.dataset}_{args.method}'
if args.method == 'ours' and args.use_graph:
    file_name += '_' + args.backbone
file_name += '.txt'
out_path = os.path.join(out_folder, file_name)
with open(out_path, 'a+') as f:
    print_str = make_print(args.method)
    f.write(print_str)
    f.write(results)
    f.write(f' run_time: { run_time }')
    f.write('\n\n')
