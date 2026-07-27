import time
from scipy.fftpack import shift
import torch
import random
import math
import torch.nn.functional as F
import os.path as osp
import numpy as np
import torch_geometric.transforms as T
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch.autograd import Variable
from torch.nn import Parameter, Linear, ModuleList, LeakyReLU
from torch_geometric.nn import SAGEConv, GATConv, GCNConv, GCN2Conv, ChebConv, ARMAConv, APPNP
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import to_scipy_sparse_matrix,to_dense_adj,dense_to_sparse,add_remaining_self_loops
import scipy.sparse as sp
from torch_geometric.nn.inits import zeros
from torch_scatter import scatter_add
from torch_sparse import SparseTensor, matmul, fill_diag, sum as sparsesum, mul
import networkx as nx
from torch_geometric.utils.undirected import is_undirected, to_undirected


#only for message passing
class passing(MessagePassing):
    def __init__(self):
        super().__init__(aggr='add')
    
    def forward(self, x, edge_index = None, edge_weight = None, adj_t = None):
        if adj_t is not None:
            return self.propagate(edge_index=adj_t, x=x)
        else:    
            return self.propagate(edge_index=edge_index, x=x, edge_weight=edge_weight)
    
    def message(self, x_j, edge_weight):
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t, x):
        return matmul(adj_t, x, reduce=self.aggr)
#only for message passing

class LSGPR(torch.nn.Module):
    def __init__(self, dataset, args):
        super(LSGPR, self).__init__()
        data = dataset[0]
        data.edge_index = to_undirected(data.edge_index)
        
        if dataset.num_classes == 2:
            self.out_classes = 1
        else:
            self.out_classes = dataset.num_classes
        self.num_features = dataset.num_features
        self.hidden = args.hidden
        self.dropout = args.dropout
        self.dprate = args.dprate
        self.K = args.K
        self.nlayer = args.nlayer
        self.ec = args.ec
        self.edge_keep_ratio = args.edge_keep_ratio
        self.sparse_eval = args.sparse_eval
        self.count_self_loops_in_budget = args.count_self_loops_in_budget
        
        self.lins = ModuleList()
        self.lins.append(Linear(self.num_features, self.hidden))
        self.lins.append(Linear(self.hidden, self.out_classes))
        
        tmptensor = args.alpha * ((1-args.alpha) ** torch.arange(0, args.K+1))
        tmptensor[-1] = (1-args.alpha) ** args.K
        tmptensor = tmptensor.repeat(self.nlayer, 1)
        self.att = Parameter(tmptensor)

        # rws settings 
        self.rws = args.rws
        self.device = torch.device('cuda:'+str(args.device) if torch.cuda.is_available() else 'cpu')

        #graph information
        self.N = data.num_nodes
        self.edge_index = data.edge_index
        self.M = data.num_edges
        self.adj = self._build_normalized_adj(self.edge_index, add_self_loops=True)
        self.degree = sparsesum(self.adj, dim=0)
        self.sparse_adj = None
        self.realized_offdiag_edges = None
        self.realized_total_edges = None
        self.full_offdiag_edges = int((self.edge_index[0] != self.edge_index[1]).sum().item())
        if self.edge_keep_ratio is not None:
            sparse_edge_index, realized_offdiag = self._build_sparse_edge_index(
                self.edge_index, self.N, self.edge_keep_ratio, self.count_self_loops_in_budget
            )
            self.realized_offdiag_edges = realized_offdiag
            self.sparse_adj = self._build_normalized_adj(
                sparse_edge_index,
                add_self_loops=not self.count_self_loops_in_budget,
            )
            self.realized_total_edges = int(self.sparse_adj.nnz())
            print(
                f"[SparseBudget] offdiag={self.realized_offdiag_edges}/{self.full_offdiag_edges} "
                f"self_loops_added={self.realized_total_edges - self.realized_offdiag_edges} "
                f"total={self.realized_total_edges}",
                flush=True,
            )
        
        self.passer = passing()

        self.reset_parameters()
        self.move()
        
    def reset_parameters(self):
        for i in range(2):
            self.lins[i].reset_parameters()
    
    def move(self):
        self.passer = self.passer.to(self.device)
        self.degree = self.degree.to(self.device)
        self.edge_index = self.edge_index.to(self.device)
        #self.weight = self.weight.to(self.device)
        self.adj = self.adj.to(self.device)
        if self.sparse_adj is not None:
            self.sparse_adj = self.sparse_adj.to(self.device)

    def _build_normalized_adj(self, edge_index, add_self_loops):
        if add_self_loops:
            edge_index = add_remaining_self_loops(edge_index, num_nodes=self.N)[0]
        value = torch.ones(edge_index.size(1), dtype=torch.float32, device=edge_index.device)
        adj = SparseTensor(
            row=edge_index[0],
            col=edge_index[1],
            value=value,
            sparse_sizes=(self.N, self.N),
        ).coalesce()
        degree = sparsesum(adj, dim=0).clamp(min=1).to(torch.float32)
        deg_inv_sqrt = degree.pow(-0.5)
        adj = mul(adj, deg_inv_sqrt.view(-1, 1))
        adj = mul(adj, deg_inv_sqrt.view(1, -1))
        return adj

    def _build_sparse_edge_index(self, edge_index, num_nodes, keep_ratio, count_self_loops_in_budget):
        edge_index = to_undirected(edge_index)
        offdiag_mask = edge_index[0] != edge_index[1]
        offdiag_edge_index = edge_index[:, offdiag_mask]
        row, col = offdiag_edge_index
        pair_mask = row < col
        pair_row = row[pair_mask]
        pair_col = col[pair_mask]
        full_offdiag_directed = int(pair_row.numel() * 2)

        if count_self_loops_in_budget:
            target_total_edges = int(round((full_offdiag_directed + num_nodes) * keep_ratio))
            target_offdiag_directed = max(target_total_edges - num_nodes, 0)
            target_pairs = max(int(round(target_offdiag_directed / 2.0)), 1)
        else:
            target_pairs = max(int(round(full_offdiag_directed * keep_ratio / 2.0)), 1)
        target_pairs = min(target_pairs, int(pair_row.numel()))

        degree = torch.bincount(row, minlength=num_nodes).to(torch.float32)
        score = degree[pair_row].clamp(min=1).rsqrt() * degree[pair_col].clamp(min=1).rsqrt()
        keep_idx = torch.argsort(score, descending=True)[:target_pairs]
        pair_row = pair_row[keep_idx]
        pair_col = pair_col[keep_idx]

        sparse_edge_index = torch.stack(
            [
                torch.cat((pair_row, pair_col), dim=0),
                torch.cat((pair_col, pair_row), dim=0),
            ],
            dim=0,
        )
        return sparse_edge_index, int(target_pairs * 2)

    def _propagate(self, x, adj):
        aggx = x * self.att[0][0]
        x_iter = x
        for j in range(1, self.K + 1):
            x_iter = self.passer(x_iter, adj_t=adj)
            aggx += x_iter * self.att[0][j]
        return F.relu(aggx)
        
    def spar_samp(self, att):
        L = torch.zeros(0, dtype = torch.long, device = self.device)
        R = torch.zeros(0, dtype = torch.long, device = self.device)
        V = torch.zeros(0, device = self.device)
        #A = torch.arange(0, self.N, dtype = torch.long, device = self.device)
        
        num_edges = int(self.N * math.log(1.0 * self.N) * self.ec)
        genidx = torch.arange(0, num_edges, device = self.device)
        for i in range(1, self.K+1):
            idx = torch.randint(0, self.M, (num_edges, ), device = self.device)
            idl = torch.randint(0, i,      (num_edges, ), device = self.device)
            idr = i - 1 - idl
            l = self.adj.storage.row()[idx]
            r = self.adj.storage.col()[idx]
            le = self.adj.random_walk(l, i-1)[genidx, idl]
            re = self.adj.random_walk(r, i-1)[genidx, idr]
            # mini_graph = SparseTensor(
            #     row = torch.cat((le, re)), 
            #     col = torch.cat((re, le)), 
            #     sparse_sizes = (self.N, self.N)
            # ).coalesce()
            # deg = sparsesum(mini_graph, dim=0)
            val = (self.degree[le] ** -0.5) * (self.degree[re] ** -0.5) * att[i] * (self.M / num_edges)
            L = torch.cat((L, le,  re))
            R = torch.cat((R, re,  le))
            V = torch.cat((V, val, val))
        sparse = SparseTensor(
            row = L,
            col = R,
            value = V,
            sparse_sizes = (self.N, self.N)
        ).coalesce()
        return sparse
        
    def forward(self, data):
        
        x = data.x
       
        x = self.lins[0](F.dropout(x, p = self.dropout, training = self.training))
        if self.edge_keep_ratio is not None:
            prop_adj = self.sparse_adj if (self.training or self.sparse_eval) else self.adj
            x = self._propagate(x, prop_adj)
        else:
            if self.training:
                sparse = self.spar_samp(self.att[0])
                x = x * self.att[0][0] + self.passer(x, adj_t = sparse)
                x = F.relu(x)
            else:
                x = self._propagate(x, self.adj)
                
        x = self.lins[-1](x)
        if self.out_classes == 1:
            return x
        else:
            return F.log_softmax(x, dim=1)
    
