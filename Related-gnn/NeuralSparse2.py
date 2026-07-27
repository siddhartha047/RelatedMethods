#!/usr/bin/env python
# coding: utf-8

# In[489]:


#jupyter nbconvert --to script NeuralSparse2.ipynb


# In[490]:


all_dataset = [
    "Cornell",
    "Texas",
    "Wisconsin",
    "reed98",
    "amherst41",
    "penn94",
    "Roman-empire",
    "cornell5",
    "Squirrel",
    "johnshopkins55",
    "Actor",
    "Minesweeper",
    "Questions",
    "Chameleon",
    "Tolokers",
    "Amazon-ratings",
    "genius",
    "pokec",
    "arxiv-year",
    "snap-patents",
    "ogbn-proteins",
    "Cora",
    "DBLP",
    "Computers",
    "PubMed",
    "Cora_ML",
    "SmallCora",
    "CS",
    "Photo",
    "Physics",
    "CiteSeer",
    "wiki",
    "Reddit"
]


# In[491]:


import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Cornell', choices=all_dataset, help='dataset')
    parser.add_argument('--epochs', type=int, default=200,  help='epochs')
    parser.add_argument('--k', type=int, default=3,  help='neigborhood size')
    parser.add_argument('--nosparsify', action='store_false') # True
    parser.add_argument('--nolog', action='store_false') #True
    

    return parser.parse_known_args()

args,_ = parse_args()

log = args.nolog


# In[492]:


import os
import sys
import math
import copy 
import torch 
import random
DEBUG = False
import numpy as np
import torch_sparse
import pandas as pd 
import pickle as pkl
from tqdm import tqdm
import time
import networkx as nx
from dgl import DGLGraph
from scipy import linalg
from pathlib import Path
from torch import Tensor
import scipy.sparse as sp
from random import randint
from dgl import transforms
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from scipy import sparse, stats
from scipy.sparse import csgraph
from scipy.sparse import csr_matrix
from dgl import from_networkx, DGLGraph
from torch_geometric.utils import scatter
from dgl.data import citation_graph as citegrh
from torch_geometric.typing import SparseTensor
from torch_geometric.utils import to_undirected
from torch_geometric.utils import remove_self_loops
from sklearn.metrics.pairwise import cosine_similarity
# from ipynb.fs.full.Dataset import get_data_from_dataset
# from ipynb.fs.full.Dataset import get_data_from_dataset,train_val_test_mask
from ogb.nodeproppred import Evaluator, PygNodePropPredDataset
from typing import Callable, List, NamedTuple, Optional, Tuple, Union
from torch_geometric.utils import add_self_loops,add_remaining_self_loops
# from ipynb.fs.full.SpectralSparsifier import EffectiveResistance, LocalEffectiveResistance, get_sparse_adj_matrix


# In[493]:


import Notebooks.DeviceDir as DeviceDir

DIR, RESULTS_DIR = DeviceDir.get_directory()
device, NUM_PROCESSORS = DeviceDir.get_device()


# In[494]:


from ipynb.fs.full.SGSLoadDataset import LOAD_DATASET

#DATASET_NAME = "Wisconsin"
DATASET_NAME = args.dataset
data, dataset  = LOAD_DATASET(DIR, DATASET_NAME)
num_classes = max(data.y).item()+1


# In[495]:


import os
import  scipy.sparse as sp
import numpy as np
import torch
import torch
import torch.nn as nn
import sys
import pickle as pkl
import numpy as np
import os
import torch.nn.functional as F
import os
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.sparse import coo_matrix
from torch_sparse import SparseTensor
import dgl.sparse as dglsp


# In[496]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

_LAYER_UIDS = {}

def get_layer_uid(layer_name=''):
    """Helper function, assigns unique layer IDs."""
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]

def sparse_dropout(x, rate, noise_shape):
    """
    Dropout for sparse tensors.
    """
    random_tensor = 1 - rate
    random_tensor += torch.rand(noise_shape, dtype=x.dtype, device=x.device)
    dropout_mask = torch.floor(random_tensor).bool()
    pre_out = x.coalesce()  # Ensure sparse tensor is in coalesced form
    retained_values = pre_out.values() * dropout_mask.float()
    return torch.sparse_coo_tensor(pre_out.indices(), retained_values * (1./(1 - rate)), pre_out.size())

def dot(x, y, sparse=False):
    """
    Wrapper for torch.matmul (sparse vs dense).
    """
    if sparse:
        res = torch.sparse.mm(x, y)
    else:
        res = torch.matmul(x, y)
    return res

class Dense(nn.Module):
    """Dense layer in PyTorch."""
    def __init__(self, input_dim, output_dim, dropout=0.0, act=F.relu, bias=False, activation=F.relu, featureless=False):
        super(Dense, self).__init__()
        self.act = act
        self.featureless = featureless
        self.dropout = dropout
        self.bias = bias

        # Weight initialization
        self.weights_ = nn.Parameter(torch.FloatTensor(input_dim, output_dim))
        nn.init.xavier_uniform_(self.weights_)

        if self.bias:
            self.bias_param = nn.Parameter(torch.FloatTensor(output_dim))
            nn.init.zeros_(self.bias_param)
        else:
            self.bias_param = None

    def forward(self, inputs):
        x = F.dropout(inputs, self.dropout, training=self.training)

        # transform
        output = torch.matmul(x, self.weights_)

        # bias
        if self.bias_param is not None:
            output += self.bias_param

        return self.act(output)


class GraphConvolution(nn.Module):
    """Graph convolution layer in PyTorch."""
    def __init__(self, input_dim, output_dim, dropout=0.0, is_sparse_inputs=False, activation=F.relu, bias=False, featureless=False):
        super(GraphConvolution, self).__init__()
        self.dropout = dropout
        self.activation = activation
        self.is_sparse_inputs = is_sparse_inputs
        self.featureless = featureless
        self.bias = bias

        # Weight initialization
        self.weights_ = nn.Parameter(torch.FloatTensor(input_dim, output_dim))
        nn.init.xavier_uniform_(self.weights_)

        if self.bias:
            self.bias_param = nn.Parameter(torch.FloatTensor(output_dim))
            nn.init.zeros_(self.bias_param)
        else:
            self.bias_param = None

    def forward(self, inputs,training):
        x, support_ = inputs

        # Apply dropout
        if training:
            x = F.dropout(x, self.dropout)

        # Convolve
        pre_sup = dot(x, self.weights_, sparse=self.is_sparse_inputs)
        output = dot(support_, pre_sup, sparse=self.is_sparse_inputs)
#         output = dot(support_, pre_sup, sparse=False)

        # Bias
        if self.bias_param is not None:
            output += self.bias_param

        return self.activation(output)


# In[497]:


def add_noisy_edge(shape, size, nb_noising_edges):
    noise_row = np.random.choice(range(size), nb_noising_edges)
    noise_col = np.random.choice(range(size), nb_noising_edges)
    noise_data = np.ones_like(noise_row)
    noise_adj = sp.coo_matrix((noise_data, (noise_row, noise_col)), shape=shape)
    return noise_adj

def preprocess_features(features):
    # Assuming features is a NumPy array, normalize if necessary
    return torch.tensor(features, dtype=torch.float32)

def sample_mask(idx, size):
    # Create a mask for training, validation, and testing
    mask = np.zeros(size, dtype=bool)
    mask[idx] = True
    return torch.tensor(mask, dtype=torch.bool)

def load_data(dataset_str):
    # Placeholder function, should load your dataset appropriately
    pass


# In[498]:


def parse_index_file(filename):
    index = []
    with open(filename, 'r') as f:
        for line in f:
            index.append(int(line.strip()))
    return index

def sample_mask(idx, size):
    # Create a mask for training, validation, and testing
    mask = np.zeros(size, dtype=bool)
    mask[idx] = True
    return torch.tensor(mask, dtype=torch.bool)

def preprocess_features(features):
    return torch.tensor(features, dtype=torch.float32)

def load_data(dataset_str, directory):
    file_location = os.path.join(directory, "ind.{}.test.index".format(dataset_str))
    test_idx_reorder = parse_index_file(file_location)
    test_idx_range = np.sort(test_idx_reorder)
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for name in names:
        file_path = os.path.join(directory, "ind.{}.{}".format(dataset_str, name))
        with open(file_path, 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))
    x, y, tx, ty, allx, ally, graph = tuple(objects)
    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    idx_test = test_idx_range.tolist()
    idx_train = range(140)
    idx_val = range(len(ally) - 500, len(ally))

    return x, y, tx, ty, allx, ally, adj, test_idx_range,features,labels,idx_train,idx_val,idx_test


# In[499]:


import torch
import torch.nn.functional as F
from sklearn import metrics

def masked_softmax_cross_entropy(preds, labels, mask):
    """
    Softmax cross-entropy loss with masking.
    """
    loss = F.cross_entropy(preds, labels, reduction='none')  # No reduction to apply mask manually
    mask = mask.float()
    mask /= mask.mean()  # Normalize mask
    loss *= mask
    return loss.mean()

def masked_accuracy(preds, labels, mask):
    """
    Accuracy with masking.
    """
    correct_prediction = (preds.argmax(dim=1) == labels.argmax(dim=1)).float()
    mask = mask.float()
    mask /= mask.mean()  # Normalize mask
    correct_prediction *= mask
    return correct_prediction.mean()

def softmax_cross_entropy(preds, labels):
    """
    Softmax cross-entropy loss.
    """
    loss = F.cross_entropy(preds, labels)
    return loss

def sigmoid_cross_entropy(preds, labels):
    """
    Sigmoid cross-entropy loss.
    """
    labels = labels.float()
    loss = F.binary_cross_entropy_with_logits(preds, labels)
    return loss.mean()

def accuracy(preds, labels):
    """
    Accuracy.
    """
    correct_prediction = (preds.argmax(dim=1) == labels.argmax(dim=1)).float()
    return correct_prediction.mean()

def calc_f1(y_pred, y_true):
    """
    F1 score calculation.
    """
    y_pred = (y_pred > 0.5).float()  # Convert predictions to binary (0 or 1)
    y_true = y_true.float()
    return metrics.f1_score(y_true.cpu().numpy(), y_pred.cpu().numpy(), average="micro")


# In[500]:


# # import torch
# # import torch.nn as nn
# # import torch.nn.functional as F
# # import dgl
# # import dgl.sparse as dglsp


# class GumbleGCN(nn.Module):
#     def __init__(self, adj_matrix, shape, input_dim, output_dim, k, dropout=0.):
#         super(GumbleGCN, self).__init__()

#         self.adj_matrix = adj_matrix  # DGL SparseMatrix
#         self.shape = shape
#         self.input_dim = input_dim
#         self.output_dim = output_dim
#         self.k = k
#         self.hidden1 = 16
#         self.hidden2 = 8
#         self.weighted = True
#         self.weight_decay = 0.0
#         self.flag_value = 0

#         self.fc_dim = nn.Linear(2 * self.input_dim, 32)

#         # Define layers
#         self.layer1 = GraphConvolution(
#             input_dim=self.input_dim,
#             output_dim=self.hidden1,
#             activation=F.relu,
#             dropout=dropout,
#             is_sparse_inputs=True  # Enable sparse inputs
#         )

#         self.layer2 = GraphConvolution(
#             input_dim=self.hidden1,
#             output_dim=self.hidden2,
#             activation=F.relu,
#             dropout=dropout,
#             is_sparse_inputs=True # Enable sparse inputs
#         )

#         self.layer3 = Dense(
#             input_dim=self.hidden2,
#             output_dim=self.output_dim,
#             activation=lambda x: x,
#             dropout=dropout
#         )

#         # Sparse layers for adjacency matrix
#         self.fb_input = nn.Linear(self.input_dim, self.hidden1)
#         self.slayer1 = nn.Linear(2 * self.hidden1 + 1, 32)
#         self.slayer2 = nn.Linear(32, 1, bias=True)

#     def sample_gumbel(self, shape, eps=1e-20):
#         """Sample from Gumbel(0, 1)."""
#         U = torch.rand(shape, dtype=torch.float32, device=self.adj_matrix.device)
#         return -torch.log(-torch.log(U + eps) + eps)

#     def gumbel_softmax_sample(self, logits, temperature, is_train):
#         """Draw a sample from the Gumbel-Softmax distribution."""
#         r = self.sample_gumbel(logits.val.shape)  # Use .val to access non-zero values
#         if is_train:
#             values = torch.log(logits.val) + r
#         else:
#             values = torch.log(logits.val)
#         values /= temperature

#         # Create a new SparseMatrix with updated values
#         A = dglsp.spmatrix(logits.indices(), values, self.shape)

#         # Apply softmax to the SparseMatrix
#         A_softmax = dglsp.softmax(A, dim=1)

#         return A_softmax

#     def forward(self, inputs, training=None):
#         x, label, mask, temperature = inputs

#         # Ensure the adjacency matrix is coalesced
#         adj_matrix = self.adj_matrix.coalesce()

#         # Sparse adjacency operations
#         f1 = self.fb_input(x[adj_matrix.indices()[0]])
#         f2 = self.fb_input(x[adj_matrix.indices()[1]])
#         auv = adj_matrix.values().unsqueeze(-1)  # Use .values() instead of .val

#         temp = torch.cat([f1, f2, auv], dim=-1)
#         temp = F.relu(self.slayer1(temp))
#         temp = self.slayer2(temp)
#         z = temp.view(-1)

#         # Create and normalize adjacency matrix
#         A = dglsp.spmatrix(adj_matrix.indices(), z, self.shape)
#         A_softmax = dglsp.softmax(A, dim=1)
#         y = self.gumbel_softmax_sample(A_softmax, temperature, training)
        

#         # Convert sparse matrix to dense for top-k selection
#         y_dense = y.to_dense()
#         #print(y_dense.shape)
#         top_k_v, top_k_i = torch.topk(y_dense, self.k, dim=-1)
#         kth = torch.min(top_k_v, dim=-1)[0] + 1e-10
#         kth = kth.unsqueeze(-1).expand_as(y_dense)
#         mask2 = (y_dense >= kth).float()

#         if self.weighted:
#             dense_support = mask2 * y_dense
#         else:
#             raise ValueError("Unweighted mode is not supported with sparse matrices!")
                

#         # Add self-loops to sparse matrix
#         self_loops = torch.eye(self.shape[0], device=y.device)
#         dense_support = dense_support + self_loops

#         # Normalize the dense matrix
#         row_sum = dense_support.sum(dim=-1) + 1e-6
#         d_inv_sqrt = torch.pow(row_sum, -0.5)
#         d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
#         support = torch.matmul(d_mat_inv_sqrt, dense_support)
#         support = torch.matmul(support, d_mat_inv_sqrt)

        
#         # Pass through layers
#         hidden = self.layer1((x, support), training)
#         hidden = self.layer2((hidden, support), training)
#         output = self.layer3(hidden)

#         # Weight decay loss
#         loss = sum(self.weight_decay * torch.sum(param**2) for param in self.layer1.parameters())

#         # Cross-entropy loss
#         loss += masked_softmax_cross_entropy(output, label, mask)

#         # Accuracy
#         acc = masked_accuracy(output, label, mask)

#         if self.flag_value == 0:
#             num_edges_retained = mask2.sum().item()
#             total_edges = self.adj_matrix._nnz()
#             percentage_retained = (num_edges_retained / total_edges) * 100
#             print(f"Percentage of edges retained: {percentage_retained:.2f}%")
#             self.flag_value = 1

#         return loss, acc


# In[ ]:





# In[ ]:





# In[501]:


import torch
import torch.nn.functional as F

class GumbleGCN(nn.Module):
    def __init__(self, adj_matrix, shape, input_dim, output_dim, k, dropout=0.):
        super(GumbleGCN, self).__init__()

        #self.adj_matrix = adj_matrix.coalesce()  # Should be a sparse tensor now
        self.adj_matrix = adj_matrix  # Should be a sparse tensor now
        self.shape = shape
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.k = k
        self.hidden1 = 64
        self.hidden2 = 32
        self.weighted = True
        self.weight_decay = 0.0
        self.flag_value = 0
        self.dropout = nn.Dropout(0.5)
        
        self.fc_dim = nn.Linear(2*self.input_dim, 32)

        # Define layers
        self.layer1 = GraphConvolution(
            input_dim=self.input_dim,
            output_dim=self.hidden1,
            activation=F.relu,
            dropout=dropout,
            is_sparse_inputs=True
        )

        self.layer2 = GraphConvolution(
            input_dim=self.hidden1,
            output_dim=self.hidden2,
            #output_dim=self.output_dim,
            activation=F.relu,
            #activation=F.relu,
            dropout=dropout,
            is_sparse_inputs=True
        )

        self.layer3 = Dense(
            input_dim=self.hidden2,
            output_dim=self.output_dim,
            activation=lambda x: x,
            dropout=dropout
        )

        # Define additional dense layers for the sparse representation
        self.fb_input = nn.Linear(self.input_dim, self.hidden1)
        self.slayer1 = nn.Linear(2*self.hidden1+1, 32)        
        self.slayer2 = nn.Linear(32, 1, bias=True)

    def sample_gumbel(self, shape, eps=1e-20):
        """Sample from Gumbel(0, 1)"""
        U = torch.rand(shape, dtype=torch.float32, device=self.adj_matrix.device)
        return -torch.log(-torch.log(U + eps) + eps)

    def gumbel_softmax_sample(self, logits, temperature, is_train):
        if is_train:
            noise = torch.rand_like(logits.values(), dtype=logits.dtype)
            noise = -torch.log(-torch.log(noise + 1e-20) + 1e-20)
            logits = logits + noise

        # Coalesce the sparse tensor before applying softmax
        logits = logits.coalesce()  # Ensure the tensor is coalesced

        # Apply softmax on sparse tensor values
        A_softmax = F.softmax(logits.values(), dim=0)  # Apply softmax only on values

        # Return as a sparse tensor
        A_softmax_sparse = torch.sparse.FloatTensor(logits.indices(), A_softmax, logits.shape)

        return A_softmax_sparse

    def forward(self, inputs, training=None):
        x, label, mask, temperature = inputs

        # Get the feature vectors based on adjacency matrix indices
        f1 = self.dropout(F.relu(self.fb_input(x[self.adj_matrix._indices()[0]])))
        f2 = self.dropout(F.relu(self.fb_input(x[self.adj_matrix._indices()[1]])))

        # Augment features with adjacency values
        auv = self.adj_matrix._values().unsqueeze(-1)
        temp = torch.cat([f1, f2, auv], dim=-1)

        # Apply layers to the augmented features
        temp = F.relu(self.slayer1(temp))
        temp = self.slayer2(temp)
        z = temp.view(-1)

        # Construct the sparse adjacency matrix A
        A = torch.sparse.FloatTensor(self.adj_matrix.indices(), z, self.shape)
        A_softmax = self.gumbel_softmax_sample(A, temperature, training)

#         with torch.no_grad():
#             A_softmax_coalesced = A_softmax.coalesce()        
#             values, indices = A_softmax_coalesced.values(), A_softmax_coalesced.indices()        
#             n_rows = A_softmax_coalesced.size(0)
#             mask_values = torch.zeros_like(values)
#             top_k_values, top_k_indices = torch.topk(values, k, dim=-1, largest=True, sorted=True)
            
#             print(top_k_indices.shape)
            
#             mask_values.scatter_(0, top_k_indices, 1)  # Set 1 at the top-k indices positions
#             mask2 = torch.sparse.FloatTensor(indices, mask_values, A_softmax_coalesced.shape)
        
    
#        ---- FOR LOOP -- SPARSE
#         #with torch.no_grad():            
#         A_softmax_coalesced = A_softmax.coalesce()
#         values, indices = A_softmax_coalesced.values(), A_softmax_coalesced.indices()
#         n_rows = A_softmax_coalesced.size(0)
#         mask_values = torch.full_like(values, 0)  # Initialize with 1e-10

#         row_indices = indices[0]
#         unique_rows = torch.unique(row_indices)

#         for row in unique_rows:
#             row_mask = (row_indices == row)
#             row_values = values[row_mask]
#             row_indices_in_row = torch.nonzero(row_mask).squeeze()

# #                 print(row_indices_in_row)

#             if row_values.numel() > 0:

#                 if row_values.numel() == 1:
#                     mask_values[row_indices_in_row] = row_values  # Directly assign the single value                    
#                 else:                    
#                     top_k_values, top_k_indices = torch.topk(row_values, min(k, row_values.numel()), largest=True, sorted=True)                    
# #                         print(top_k_indices.shape)                    
#                     mask_values[row_indices_in_row[top_k_indices]] = top_k_values

#         mask2 = torch.sparse.FloatTensor(indices, mask_values, A_softmax_coalesced.shape)


#         ------without loop SPARSE--
#          with torch.no_grad():
           
        if args.nosparsify:
    
            # Assuming k is predefined and A_softmax is a sparse tensor
            A_softmax_coalesced = A_softmax.coalesce()
            values, indices = A_softmax_coalesced.values(), A_softmax_coalesced.indices()
            n_rows = A_softmax_coalesced.size(0)

            # Extract row indices and count occurrences per row
            row_indices = indices[0]
            row_count = torch.bincount(row_indices, minlength=n_rows)

            # Prepare an auxiliary tensor for grouping
            row_offsets = torch.cumsum(row_count, dim=0) - row_count
            max_count = row_count.max().item()
            aux_indices = torch.arange(max_count, device=values.device).unsqueeze(0).expand(n_rows, max_count)
            valid_mask = aux_indices < row_count.unsqueeze(1)

            # Compute valid indices for sorting and grouping
            global_indices = row_offsets.unsqueeze(1) + aux_indices
            global_indices = global_indices[valid_mask]

            # Sort values globally by rows
            sorted_values, sorted_order = torch.sort(values[global_indices], descending=True)
            sorted_rows = row_indices[global_indices][sorted_order]


            # Compute top-k mask dynamically for valid indices only
            row_k = torch.minimum(torch.tensor(k, device=values.device), row_count)
            top_k_mask = torch.arange(max_count, device=values.device).unsqueeze(0) < row_k.unsqueeze(1)
            top_k_mask = top_k_mask[valid_mask]  # Apply the mask to only valid positions

            # Use the corrected top-k mask to index global_indices
            top_k_global_indices = global_indices[sorted_order][top_k_mask]

            # Update mask values
            mask_values = torch.zeros_like(values)
            #mask_values[top_k_global_indices] = values[top_k_global_indices]
            mask_values[top_k_global_indices] = 1

            #print(sum(mask_values))

            # Construct final sparse tensor
            mask2 = torch.sparse_coo_tensor(indices, mask_values, A_softmax_coalesced.shape)
            
#             print("global_indices shape:", global_indices.shape)
#             print("sorted_order shape:", sorted_order.shape)
#             print("top_k_mask shape:", top_k_mask.shape)

            
        
        #print(mask2.shape)

        
        #support = A
        #support = mask2
        
        #support = A.coalesce()
        #support = A
        
        
#         values, indices = A_softmax_coalesced.values(), A_softmax_coalesced.indices()

#         # Top-k selection from sparse matrix (without converting to dense)
#         top_k_values, top_k_indices = torch.topk(values, self.k, dim=-1, largest=True, sorted=True)

#         # Masking and normalization on top-k values
#         kth = torch.min(top_k_values, dim=-1)[0] + 1e-10
#         kth = kth.unsqueeze(-1).expand_as(top_k_values)
#         mask2 = (top_k_values >= kth).float()

#         # Dense support calculation with weighted top-k values
#         if self.weighted:
#             dense_support = mask2 * top_k_values
#         else:
#             raise ValueError("Unweighted mode is not supported with sparse matrices!")

#         # Add self-loops to the sparse matrix (ensuring it's n x n)
        

        identity_indices = torch.eye(self.shape[0], dtype=torch.long).nonzero().T  # Get indices of identity matrix (non-zero elements)
        identity_values = torch.ones(identity_indices.shape[1], dtype=torch.float32)  # Values at those indices (1)
        self_loops = torch.sparse_coo_tensor(identity_indices, identity_values, self.shape).to(device)
        
        
        if args.nosparsify:
            support  = A * mask2 + self_loops 
        else:            
            support = A + self_loops 
        
#         support = A
#         support  = mask2 + self_loops 

        #self_loops = torch.eye(self.shape[0], device=device)

#         # Ensure the self-loops matrix matches the size of dense_support (n x n)
#         dense_support = dense_support + self_loops

#         # Normalize the support matrix: D^-1/2 * A * D^-1/2
#         row_sum = dense_support.sum(dim=-1) + 1e-6  # Avoid NaN
#         d_inv_sqrt = torch.pow(row_sum, -0.5)
#         d_mat_inv_sqrt = torch.diag(d_inv_sqrt)

#         # Calculate the normalized support matrix
#         ad = torch.mm(dense_support, d_mat_inv_sqrt)
#         ad_t = ad.transpose(0, 1)
#         support = torch.matmul(ad_t, d_mat_inv_sqrt)

        # Now pass the normalized support matrix through layers
        hidden = self.layer1((x, support), training)
        hidden = self.layer2((hidden, support), training)
        output = self.layer3(hidden)

        # Weight decay loss
        loss = 0
        for param in self.layer1.parameters():
            loss += self.weight_decay * torch.sum(param ** 2)

        # Cross-entropy loss
        loss += masked_softmax_cross_entropy(output, label, mask)

        # Accuracy
        acc = masked_accuracy(output, label, mask)

        return loss, acc


# In[ ]:





# ## Initialize the Model

# In[502]:


data = data.to("cpu")


# In[503]:


print(data.train_mask.sum(),data.val_mask.sum(),data.test_mask.sum())
print(data)


# In[504]:


num_classes = data.y.max().item() + 1
labels = torch.zeros(data.y.size(0), num_classes)
labels.scatter_(1, data.y.unsqueeze(1), 1)


y_train = np.zeros(labels.shape)
y_val = np.zeros(labels.shape)
y_test = np.zeros(labels.shape)

train_mask = data.train_mask
val_mask = data.val_mask
test_mask = data.test_mask

print(data.train_mask.sum(),val_mask.sum(),test_mask.sum())

y_train[data.train_mask, :] = labels[data.train_mask, :]
y_val[data.val_mask, :] = labels[data.val_mask, :]
y_test[data.test_mask, :] = labels[data.test_mask, :]

edge_index = data.edge_index
row, col = edge_index
indices = torch.stack([row, col], dim=0)
values = torch.ones(indices.size(1), dtype=torch.float32)
num_nodes = data.num_nodes
shape = torch.Size([num_nodes, num_nodes])
adj_tensor = torch.sparse_coo_tensor(indices, values, shape)
# f1 = data.x[adj_tensor._indices()[0]]
# f2 = data.x[adj_tensor._indices()[1]]

# auv = adj_tensor._values().unsqueeze(-1)
# temp = torch.cat([f1, f2, auv], dim=-1)

train_label = torch.tensor(y_train).to(device)
train_mask = train_mask.clone().detach().to(device)  # Fix for the warning
val_label = torch.tensor(y_val).to(device)
val_mask = val_mask.clone().detach().to(device)  # Fix for the warning
test_label = torch.tensor(y_test).to(device)
test_mask = test_mask.clone().detach().to(device)  # Fix for the warning
features = data.x.clone().detach().float().to(device)  # Fix for the warning and specify dtype
dropout = 0  # args.dropout
feature_tensor = data.x.clone().detach().float().to(device)


# In[505]:


k = args.k

adj_tensor = adj_tensor.coalesce()

def compute_edges():
    # Assuming k is predefined and A_softmax is a sparse tensor
        
    values, indices = adj_tensor.values(), adj_tensor.indices()
    n_rows = adj_tensor.size(0)

    # Extract row indices and count occurrences per row
    row_indices = indices[0]
    row_count = torch.bincount(row_indices, minlength=n_rows)

    # Prepare an auxiliary tensor for grouping
    row_offsets = torch.cumsum(row_count, dim=0) - row_count
    max_count = row_count.max().item()
    aux_indices = torch.arange(max_count, device=values.device).unsqueeze(0).expand(n_rows, max_count)
    valid_mask = aux_indices < row_count.unsqueeze(1)

    # Compute valid indices for sorting and grouping
    global_indices = row_offsets.unsqueeze(1) + aux_indices
    global_indices = global_indices[valid_mask]

    # Sort values globally by rows
    sorted_values, sorted_order = torch.sort(values[global_indices], descending=True)
    sorted_rows = row_indices[global_indices][sorted_order]


    # Compute top-k mask dynamically for valid indices only
    row_k = torch.minimum(torch.tensor(k, device=values.device), row_count)
    top_k_mask = torch.arange(max_count, device=values.device).unsqueeze(0) < row_k.unsqueeze(1)
    top_k_mask = top_k_mask[valid_mask]  # Apply the mask to only valid positions

    # Use the corrected top-k mask to index global_indices
    top_k_global_indices = global_indices[sorted_order][top_k_mask]

    # Update mask values
    mask_values = torch.zeros_like(values)
    #mask_values[top_k_global_indices] = values[top_k_global_indices]
    mask_values[top_k_global_indices] = 1

    sparse_num_edge = sum(mask_values).item()
    
    
    print(f'E = {data.num_edges}, S = {sparse_num_edge}, Ratio = {sparse_num_edge/data.num_edges:.4f}')
    
    return 

compute_edges()


# In[506]:


# data = data.to(device)
adj_tensor = adj_tensor.to(device)
# f1 = f1.to(device)
# f2 = f2.to(device)


# In[507]:


if log == True:
    print("GumbleGCN Model Parameters")
    print(f"Shape : {shape}")
    print(f"Input Dim : {features.shape[-1]}")
    print(f"Output Dim : {labels.shape[-1]}")
    print(f"K : {k}")
    print(f"Adjacency Sparse Tensor : {adj_tensor}")

model = GumbleGCN(adj_tensor, shape = shape, input_dim=features.shape[-1], output_dim=labels.shape[-1], k=k).to(device)

None


# In[508]:


# From config 
temp_N = 50
temp_r = 1e-3
early_stopping = 100
# Optimizer setup
optimizer = torch.optim.Adam(model.parameters(), lr=0.01) # args.learning_rate

persist = 0
best_test_acc = 0
epochs =  args.epochs
init_temp = 0.05


# In[509]:


# for epoch in range(epochs):
#     if epoch % 50 == 0:
#         decay_temp = np.exp(-1 * 1e-3 * epoch)
#         temp = max(0.05, decay_temp)
#     model.train()  # Set model to training mode
#     optimizer.zero_grad()  # Clear previous gradients
#     # Forward pass
#     loss, acc = model((features, train_label, train_mask, temp))

#     # Backward pass
#     loss.backward()  # Compute gradients
#     optimizer.step()  # Update weights

#     print(epoch, 'temp', temp, 'loss', loss.item(), 'acc', acc.item())
#     # if epoch % 1 == 0:
#     #     print(epoch, 'temp', temp, 'loss', loss.item(), 'acc', acc.item(), '\tval:', val_acc.item())


# In[510]:


EpochTimes = []

best_val_acc = 0
best_test_throughout = 0
test_at_val  = 0
last_5_loss = [] 
for epoch in range(epochs):
    
    start = time.time()
    
    if epoch % temp_N == 0:
        decay_temp = np.exp(-1 * temp_r * epoch)
        temp = max(0.05, decay_temp)

    model.train()
    optimizer.zero_grad()  # Zero out the gradients
    loss, acc = model((features, train_label, train_mask, temp)) 

    loss.backward()  
    optimizer.step() 
    
    last_5_loss.append(loss.item())
    if len(last_5_loss) > 5:
        last_5_loss.pop(0)
        
    if len(last_5_loss) == 5 and np.std(last_5_loss) < 0.001:
        print(f"Convergence achieved at Epoch: {epoch}, Loss: {loss.item():.4f}, Temp: {temp:.4f}, Acc: {acc.item():.4f}")
        break
    
    EpochTimes.append(time.time() - start)

#     if epoch%1 == 0:
#         print(f'Train: Epoch {epoch}, Training Loss: {loss.item():.4f}, Training Accuracy: {acc.item():.4f}')

    #if epoch % 25 == 0:
    model.eval()
    with torch.no_grad():  # Disable gradient computation for validation
        test_loss, test_acc = model((features, test_label, test_mask, 1.0))
        val_loss, val_acc = model((features, val_label, val_mask, 1.0))

    if val_acc >= best_val_acc:
        best_val_acc = best_val_acc
        test_at_val = test_acc
        
    if test_acc > best_test_throughout:
        best_test_throughout = test_acc        
        persist = 0

    else:
        persist += 1

#     if persist > early_stopping:
#         break
    
    if log == True and epoch % 1 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item():0.4f}, Train Acc: {acc.item():0.4f}, Val Acc: {val_acc.item():0.4f},  Test Acc: {test_acc.item():0.4f}')
    
print("Best test accuracy throughout:", best_test_throughout)
print("Mean epoch time: ", np.mean(EpochTimes))
print(f'Final test accuracy: {test_at_val:.4f}')


# In[511]:


# print(best_test_acc)


# In[512]:


#torch.save(model.state_dict(), 'easy_checkpoint.pth')  # Save model weights


# In[513]:


# #model.load_state_dict(torch.load('easy_checkpoint.pth'))
# model.eval()
# with torch.no_grad(
# ):
#     test_loss, test_acc = model((features, test_label, test_mask, 1.0))
# print(f'Test Loss: {test_loss.item()}, Test Acc: {test_acc.item()}')


# In[514]:


# a= [0.7838, 0.8378, 0.7568, 0.8108, 0.7297]
# mean_a = np.mean(a)
# std_dev_a = np.std(a)
# print(f'{mean_a:.4f} +/- {std_dev_a:.4f}')


# In[ ]:




