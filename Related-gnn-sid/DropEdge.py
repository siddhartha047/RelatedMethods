#!/usr/bin/env python
# coding: utf-8

# In[7]:


from __future__ import division
from __future__ import print_function
import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from scipy.sparse import coo_matrix
from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
# from ipynb.fs.full.Dataset import get_data_from_dataset


# In[ ]:


# jupyter nbconvert --to script DropEdge.ipynb


# In[4]:


import Notebooks.DeviceDir as DeviceDir

DIR, RESULTS_DIR = DeviceDir.get_directory()
device, NUM_PROCESSORS = DeviceDir.get_device()


# In[5]:


from ipynb.fs.full.SGSLoadDataset import LOAD_DATASET, train_val_test_mask

DATASET_NAME = "Minesweeper"
# data, dataset  = LOAD_DATASET(DIR, DATASET_NAME)
# num_classes = max(data.y).item()+1


# In[6]:


import argparse

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='SmallCora', help='dataset')
    parser.add_argument('--epochs', type=int, default=10,  help='epochs')
    parser.add_argument('--log', type=str2bool, nargs='?', const=True, default=True, help='show log')

    return parser.parse_known_args()

args,_ = parse_args()

DATASET_NAME = args.dataset


# Early stoppings dot py

# In[8]:


#!/usr/bin/env python
# coding=utf-8
import numpy as np
import torch
import random,string
import datetime
import os

# TODO: hard coding the model path here.
folder = DIR+DATASET_NAME+"tmpmodel"
if not os.path.exists(folder):
    os.mkdir(folder)

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, datasets="tmp", patience=7, fname=None, clean=False, verbose=False):
        """ 
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """

        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        timstr = datetime.datetime.now().strftime("%m%d-%H%M%S")
        if fname is None:
            fname = datasets + "-" + timstr + "-" + self._random_str() + ".pt"
        self.fname = os.path.join(folder, fname)
        self.clean = clean
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score:
            self.counter += 1
            if self.verbose:
                print("EarlyStopping counter: %d out of %d"%(self.counter, self.patience))
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def _random_str(self, randomlength=3):
        a = list(string.ascii_letters)
        random.shuffle(a)
        return ''.join(a[:randomlength])

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print('Validation loss decreased (%.6f --> %.6f).  Saving model ...'%(self.val_loss_min, val_loss))
        torch.save(model.state_dict(), self.fname)
        self.val_loss_min = val_loss

    def load_checkpoint(self):
        return  torch.load(self.fname)


# normalization .py file

# In[9]:


import numpy as np
import scipy.sparse as sp

def normalized_laplacian(adj):
   adj = sp.coo_matrix(adj)
   row_sum = np.array(adj.sum(1))
   d_inv_sqrt = np.power(row_sum, -0.5).flatten()
   d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
   d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
   return (sp.eye(adj.shape[0]) - d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)).tocoo()


def laplacian(adj):
   adj = sp.coo_matrix(adj)
   row_sum = np.array(adj.sum(1)).flatten()
   d_mat = sp.diags(row_sum)
   return (d_mat - adj).tocoo()


def gcn(adj):
   adj = sp.coo_matrix(adj)
   row_sum = np.array(adj.sum(1))
   d_inv_sqrt = np.power(row_sum, -0.5).flatten()
   d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
   d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
   return (sp.eye(adj.shape[0]) + d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)).tocoo()


def aug_normalized_adjacency(adj):
   adj = adj + sp.eye(adj.shape[0])
   adj = sp.coo_matrix(adj)
   row_sum = np.array(adj.sum(1))
   d_inv_sqrt = np.power(row_sum, -0.5).flatten()
   d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
   d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
   return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()

def bingge_norm_adjacency(adj):
   adj = adj + sp.eye(adj.shape[0])
   adj = sp.coo_matrix(adj)
   row_sum = np.array(adj.sum(1))
   d_inv_sqrt = np.power(row_sum, -0.5).flatten()
   d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
   d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
   return (d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt) +  sp.eye(adj.shape[0])).tocoo()

def normalized_adjacency(adj):
   adj = sp.coo_matrix(adj)
   row_sum = np.array(adj.sum(1))
   d_inv_sqrt = np.power(row_sum, -0.5).flatten()
   d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
   d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
   return (d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)).tocoo()

def random_walk_laplacian(adj):
   adj = sp.coo_matrix(adj)
   row_sum = np.array(adj.sum(1))
   d_inv = np.power(row_sum, -1.0).flatten()
   d_mat = sp.diags(d_inv)
   return (sp.eye(adj.shape[0]) - d_mat.dot(adj)).tocoo()


def aug_random_walk(adj):
   adj = adj + sp.eye(adj.shape[0])
   adj = sp.coo_matrix(adj)
   row_sum = np.array(adj.sum(1))
   d_inv = np.power(row_sum, -1.0).flatten()
   d_mat = sp.diags(d_inv)
   return (d_mat.dot(adj)).tocoo()

def random_walk(adj):
   adj = sp.coo_matrix(adj)
   row_sum = np.array(adj.sum(1))
   d_inv = np.power(row_sum, -1.0).flatten()
   d_mat = sp.diags(d_inv)
   return d_mat.dot(adj).tocoo()

def no_norm(adj):
   adj = sp.coo_matrix(adj)
   return adj


def i_norm(adj):
    adj = adj + sp.eye(adj.shape[0])
    adj = sp.coo_matrix(adj)
    return adj
  
def fetch_normalization(type):
   switcher = {
       'NormLap': normalized_laplacian,  # A' = I - D^-1/2 * A * D^-1/2
       'Lap': laplacian,  # A' = D - A
       'RWalkLap': random_walk_laplacian,  # A' = I - D^-1 * A
       'FirstOrderGCN': gcn,   # A' = I + D^-1/2 * A * D^-1/2
       'AugNormAdj': aug_normalized_adjacency,  # A' = (D + I)^-1/2 * ( A + I ) * (D + I)^-1/2
       'BingGeNormAdj': bingge_norm_adjacency, # A' = I + (D + I)^-1/2 * (A + I) * (D + I)^-1/2
       'NormAdj': normalized_adjacency,  # D^-1/2 * A * D^-1/2
       'RWalk': random_walk,  # A' = D^-1*A
       'AugRWalk': aug_random_walk,  # A' = (D + I)^-1*(A + I)
       'NoNorm': no_norm, # A' = A
       'INorm': i_norm,  # A' = A + I
   }
   func = switcher.get(type, lambda: "Invalid normalization technique.")
   return func

def row_normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


# utils. py file

# metric . py

# In[10]:


import numpy as np
import scipy.sparse as sp
import torch
def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def roc_auc_compute_fn(y_preds, y_targets):
    try:
        from sklearn.metrics import roc_auc_score
    except ImportError:
        raise RuntimeError("This contrib module requires sklearn to be installed.")

    y_true = y_targets.cpu().numpy()
    y_true = encode_onehot(y_true)
    y_pred = y_preds.cpu().detach().numpy()
    return roc_auc_score(y_true, y_pred)

def prec_recall_n(output, labels, topn):
    preds = output.detach().numpy()[-1]
    pass


# In[11]:


import pickle as pkl
import sys
import os
import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch

datadir = "data"

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def preprocess_citation(adj, features, normalization="FirstOrderGCN"):
    adj_normalizer = fetch_normalization(normalization)
    adj = adj_normalizer(adj)
    features = row_normalize(features)
    return adj, features

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def load_citation(dataset_str="cora", normalization="AugNormAdj", porting_to_torch=True,data_path=datadir, task_type="full"):
    """
    Load Citation Networks Datasets.
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open(os.path.join(data_path, "ind.{}.{}".format(dataset_str.lower(), names[i])), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file(os.path.join(data_path, "ind.{}.test.index".format(dataset_str)))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    G = nx.from_dict_of_lists(graph)
    adj = nx.adjacency_matrix(G)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # degree = np.asarray(G.degree)
    degree = np.sum(adj, axis=1)

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]
    
    if task_type == "full":
        print("Load full supervised task.")
        #supervised setting
        idx_test = test_idx_range.tolist()
        idx_train = range(len(ally)- 500)
        idx_val = range(len(ally) - 500, len(ally))
    elif task_type == "semi":
        print("Load semi-supervised task.")
        #semi-supervised setting
        idx_test = test_idx_range.tolist()
        idx_train = range(len(y))
        idx_val = range(len(y), len(y)+500)
    else:
        raise ValueError("Task type: %s is not supported. Available option: full and semi.")

    adj, features = preprocess_citation(adj, features, normalization)
    features = np.array(features.todense())
    labels = np.argmax(labels, axis=1)
    print(porting_to_torch)
    # porting to pytorch
    if porting_to_torch:
        print("HELLLO WORLDLLLDLSD")
        features = torch.FloatTensor(features).float()
        labels = torch.LongTensor(labels)
        # labels = torch.max(labels, dim=1)[1]
        adj = sparse_mx_to_torch_sparse_tensor(adj).float()
        idx_train = torch.LongTensor(idx_train)
        idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)
        degree = torch.LongTensor(degree)
    learning_type = "transductive"
    return adj, features, labels, idx_train, idx_val, idx_test, degree, learning_type

def sgc_precompute(features, adj, degree):
    #t = perf_counter()
    for i in range(degree):
        features = torch.spmm(adj, features)
    precompute_time = 0 #perf_counter()-t
    return features, precompute_time

def set_seed(seed, cuda):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda: torch.cuda.manual_seed(seed)


def loadRedditFromNPZ(dataset_dir=datadir):
    adj = sp.load_npz(dataset_dir+"reddit_adj.npz")
    data = np.load(dataset_dir +"reddit.npz")

    return adj, data['feats'], data['y_train'], data['y_val'], data['y_test'], data['train_index'], data['val_index'], data['test_index']


def load_reddit_data(normalization="AugNormAdj", porting_to_torch=True, data_path=datadir):
    adj, features, y_train, y_val, y_test, train_index, val_index, test_index = loadRedditFromNPZ(data_path)
    labels = np.zeros(adj.shape[0])
    labels[train_index]  = y_train
    labels[val_index]  = y_val
    labels[test_index]  = y_test
    adj = adj + adj.T + sp.eye(adj.shape[0])
    train_adj = adj[train_index, :][:, train_index]
    degree = np.sum(train_adj, axis=1)

    features = torch.FloatTensor(np.array(features))
    features = (features-features.mean(dim=0))/features.std(dim=0)
    train_features = torch.index_select(features, 0, torch.LongTensor(train_index))
    if not porting_to_torch:
        features = features.numpy()
        train_features = train_features.numpy()

    adj_normalizer = fetch_normalization(normalization)
    adj = adj_normalizer(adj)
    train_adj = adj_normalizer(train_adj)

    if porting_to_torch:
        train_adj = sparse_mx_to_torch_sparse_tensor(train_adj).float()
        labels = torch.LongTensor(labels)
        adj = sparse_mx_to_torch_sparse_tensor(adj).float()
        degree = torch.LongTensor(degree)
        train_index = torch.LongTensor(train_index)
        val_index = torch.LongTensor(val_index)
        test_index = torch.LongTensor(test_index)
    learning_type = "inductive"
    return adj, train_adj, features, train_features, labels, train_index, val_index, test_index, degree, learning_type

def data_loader(dataset, data_path=datadir, normalization="AugNormAdj", porting_to_torch=True, task_type = "full"):
    if dataset == "reddit":
        return load_reddit_data(normalization, porting_to_torch, data_path)
    else:
        data, dataset = LOAD_DATASET(DIR, dataset)
        
        if dataset == "wiki":
            data = train_val_test_mask(data, train=0.2, val=0.4, test=0.4, random_state=False)

        
        # Extract features and adjacency matrix
        features = data.x  # Features of nodes
        train_features = features
        edge_index = data.edge_index  # Edge list (PyTorch Geometric format)
        num_nodes = edge_index.max().item() + 1

        # Create adjacency matrix (COO format)
        values = torch.ones(edge_index.size(1), dtype=torch.float32)  # All edges have weight 1
        row = edge_index[0].numpy()
        col = edge_index[1].numpy()
        adj = coo_matrix((values.numpy(), (row, col)), shape=(num_nodes, num_nodes))
        train_adj = adj

        # Degree of nodes
        degree = np.array(adj.sum(axis=1)).flatten()

        # Labels
        labels = data.y.cpu().numpy()
        
        idx_train = torch.nonzero(data.train_mask).squeeze().tolist()
        idx_val = torch.nonzero(data.val_mask).squeeze().tolist()
        idx_test = torch.nonzero(data.test_mask).squeeze().tolist()

#         # Train/Validation/Test split
#         if task_type == "full":
#             print("Load full supervised task.")
#             idx_train = range(len(labels) - 500)
#             idx_val = range(len(labels) - 500, len(labels))
#             idx_test = range(len(labels))
#         elif task_type == "semi":
#             print("Load semi-supervised task.")
#             idx_train = range(len(labels) - 1000)
#             idx_val = range(len(labels) - 500, len(labels))
#             idx_test = range(len(labels))
#         else:
#             raise ValueError("Task type: %s is not supported. Available options: full and semi." % task_type)
        
        print(idx_train)
        
        # Normalize adjacency matrix and features (optional, mimic `preprocess_citation`)
        if normalization == "AugNormAdj":
            adj = adj + coo_matrix(np.eye(adj.shape[0]))
            degree = np.array(adj.sum(axis=1)).flatten()
            degree = np.maximum(degree, 1)
            d_inv_sqrt = np.power(degree, -0.5).flatten()
            d_mat_inv_sqrt = coo_matrix(np.diag(d_inv_sqrt))
            adj = d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt

        features = features.numpy()

        # Convert labels to integer class indices if not already
        if labels.ndim > 1:
            labels = np.argmax(labels, axis=1)
        learning_type = "transductive"
    # else:
    #     (adj,
    #      features,
    #      labels,
    #      idx_train,
    #      idx_val,
    #      idx_test,
    #      degree,
    #      learning_type) = load_citation(dataset, normalization, porting_to_torch, data_path, task_type)
    #     train_adj = adj
    #     train_features = features

        return adj, train_adj, features, train_features, labels, idx_train, idx_val, idx_test, degree, learning_type


# In[12]:


# 
# import torch

# data, dataset = get_data_from_dataset('Cora')  #feature = data.x , train_feature = data.x, adj = adj, train_adj = adj #labels_ = data.y.cpu().numpy() degree = np.sum(adj, axis=1)
# feature = data.x 
# train_feature = data.x 

# edge_index = data.edge_index
# num_nodes = edge_index.max().item() + 1
# values = torch.ones(edge_index.size(1), dtype=torch.float32)
# row = edge_index[0].numpy()
# col = edge_index[1].numpy()
# adj = coo_matrix((values.numpy(), (row, col)), shape=(num_nodes, num_nodes))
# train_adj = adj 
# labels_ = data.y.cpu().numpy()



# dataset_name = 'Cora'
# task_type = 'full'
# normalization = 'AugNormAdj'
# porting_to_torch = False 

# data, dataset = get_data_from_dataset(dataset_name)

# # Extract features and adjacency matrix
# features = data.x  # Features of nodes
# edge_index = data.edge_index  # Edge list (PyTorch Geometric format)
# num_nodes = edge_index.max().item() + 1

# # Create adjacency matrix (COO format)
# values = torch.ones(edge_index.size(1), dtype=torch.float32)  # All edges have weight 1
# row = edge_index[0].numpy()
# col = edge_index[1].numpy()
# adj = coo_matrix((values.numpy(), (row, col)), shape=(num_nodes, num_nodes))

# # Degree of nodes
# degree = np.array(adj.sum(axis=1)).flatten()

# # Labels
# labels = data.y.cpu().numpy()

# # Train/Validation/Test split
# if task_type == "full":
#     print("Load full supervised task.")
#     idx_train = range(len(labels) - 500)
#     idx_val = range(len(labels) - 500, len(labels))
#     idx_test = range(len(labels))
# elif task_type == "semi":
#     print("Load semi-supervised task.")
#     idx_train = range(len(labels) - 1000)
#     idx_val = range(len(labels) - 500, len(labels))
#     idx_test = range(len(labels))
# else:
#     raise ValueError("Task type: %s is not supported. Available options: full and semi." % task_type)

# # Normalize adjacency matrix and features (optional, mimic `preprocess_citation`)
# if normalization == "AugNormAdj":
#     adj = adj + coo_matrix(np.eye(adj.shape[0]))
#     degree = np.array(adj.sum(axis=1)).flatten()
#     degree = np.maximum(degree, 1)
#     d_inv_sqrt = np.power(degree, -0.5).flatten()
#     d_mat_inv_sqrt = coo_matrix(np.diag(d_inv_sqrt))
#     adj = d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt

# features = features.numpy()

# # Convert labels to integer class indices if not already
# if labels.ndim > 1:
#     labels = np.argmax(labels, axis=1)
# learning_type = "transductive"


# Sample . py 

# In[13]:


# coding=utf-8
import numpy as np
import torch
import scipy.sparse as sp

class Sampler:
    """Sampling the input graph data."""
    def __init__(self, dataset, data_path="data", task_type="full"):
        self.dataset = dataset
        self.data_path = data_path
        (self.adj,
         self.train_adj,
         self.features,
         self.train_features,
         self.labels,
         self.idx_train, 
         self.idx_val,
         self.idx_test, 
         self.degree,
         self.learning_type) = data_loader(dataset, data_path, "NoNorm", False, task_type)
        
        #convert some data to torch tensor ---- may be not the best practice here.
        self.features = torch.FloatTensor(self.features).float()
        self.train_features = torch.FloatTensor(self.train_features).float()
        # self.train_adj = self.train_adj.tocsr()

        self.labels_torch = torch.LongTensor(self.labels)
        self.idx_train_torch = torch.LongTensor(self.idx_train)
        self.idx_val_torch = torch.LongTensor(self.idx_val)
        self.idx_test_torch = torch.LongTensor(self.idx_test)

        # vertex_sampler cache
        # where return a tuple
        self.pos_train_idx = np.where(self.labels[self.idx_train] == 1)[0]
        self.neg_train_idx = np.where(self.labels[self.idx_train] == 0)[0]
        # self.pos_train_neighbor_idx = np.where
        
        self.nfeat = self.features.shape[1]
        self.nclass = int(self.labels.max().item() + 1)
        self.trainadj_cache = {}
        self.adj_cache = {}
        #print(type(self.train_adj))
        self.degree_p = None

    def _preprocess_adj(self, normalization, adj, cuda):
        adj_normalizer = fetch_normalization(normalization)
        r_adj = adj_normalizer(adj)
        r_adj = sparse_mx_to_torch_sparse_tensor(r_adj).float()
        if cuda:
            r_adj = r_adj.cuda()
        return r_adj

    def _preprocess_fea(self, fea, cuda):
        if cuda:
            return fea.cuda()
        else:
            return fea

    def stub_sampler(self, normalization, cuda):
        """
        The stub sampler. Return the original data. 
        """
        if normalization in self.trainadj_cache:
            r_adj = self.trainadj_cache[normalization]
        else:
            r_adj = self._preprocess_adj(normalization, self.train_adj, cuda)
            self.trainadj_cache[normalization] = r_adj
        fea = self._preprocess_fea(self.train_features, cuda)
        return r_adj, fea

    def randomedge_sampler(self, percent, normalization, cuda):
        """
        Randomly drop edge and preserve percent% edges.
        """
        "Opt here"
        if percent >= 1.0:
            return self.stub_sampler(normalization, cuda)
        
        nnz = self.train_adj.nnz
        perm = np.random.permutation(nnz)
        preserve_nnz = int(nnz*percent)
        perm = perm[:preserve_nnz]
        r_adj = sp.coo_matrix((self.train_adj.data[perm],
                               (self.train_adj.row[perm],
                                self.train_adj.col[perm])),
                              shape=self.train_adj.shape)
        r_adj = self._preprocess_adj(normalization, r_adj, cuda)
        fea = self._preprocess_fea(self.train_features, cuda)
        return r_adj, fea

    def vertex_sampler(self, percent, normalization, cuda):
        """
        Randomly drop vertexes.
        """
        if percent >= 1.0:
            return self.stub_sampler(normalization, cuda)
        self.learning_type = "inductive"
        pos_nnz = len(self.pos_train_idx)
        # neg_neighbor_nnz = 0.4 * percent
        neg_no_neighbor_nnz = len(self.neg_train_idx)
        pos_perm = np.random.permutation(pos_nnz)
        neg_perm = np.random.permutation(neg_no_neighbor_nnz)
        pos_perseve_nnz = int(0.9 * percent * pos_nnz)
        neg_perseve_nnz = int(0.1 * percent * neg_no_neighbor_nnz)
        # print(pos_perseve_nnz)
        # print(neg_perseve_nnz)
        pos_samples = self.pos_train_idx[pos_perm[:pos_perseve_nnz]]
        neg_samples = self.neg_train_idx[neg_perm[:neg_perseve_nnz]]
        all_samples = np.concatenate((pos_samples, neg_samples))
        r_adj = self.train_adj
        r_adj = r_adj[all_samples, :]
        r_adj = r_adj[:, all_samples]
        r_fea = self.train_features[all_samples, :]
        # print(r_fea.shape)
        # print(r_adj.shape)
        # print(len(all_samples))
        r_adj = self._preprocess_adj(normalization, r_adj, cuda)
        r_fea = self._preprocess_fea(r_fea, cuda)
        return r_adj, r_fea, all_samples

    def degree_sampler(self, percent, normalization, cuda):
        """
        Randomly drop edge wrt degree (high degree, low probility).
        """
        if percent >= 0:
            return self.stub_sampler(normalization, cuda)
        if self.degree_p is None:
            degree_adj = self.train_adj.multiply(self.degree)
            self.degree_p = degree_adj.data / (1.0 * np.sum(degree_adj.data))
        # degree_adj = degree_adj.multi degree_adj.sum()
        nnz = self.train_adj.nnz
        preserve_nnz = int(nnz * percent)
        perm = np.random.choice(nnz, preserve_nnz, replace=False, p=self.degree_p)
        r_adj = sp.coo_matrix((self.train_adj.data[perm],
                               (self.train_adj.row[perm],
                                self.train_adj.col[perm])),
                              shape=self.train_adj.shape)
        r_adj = self._preprocess_adj(normalization, r_adj, cuda)
        fea = self._preprocess_fea(self.train_features, cuda)
        return r_adj, fea


    def get_test_set(self, normalization, cuda):
        """
        Return the test set. 
        """
        if self.learning_type == "transductive":
            return self.stub_sampler(normalization, cuda)
        else:
            if normalization in self.adj_cache:
                r_adj = self.adj_cache[normalization]
            else:
                r_adj = self._preprocess_adj(normalization, self.adj, cuda)
                self.adj_cache[normalization] = r_adj
            fea = self._preprocess_fea(self.features, cuda)
            return r_adj, fea

    def get_val_set(self, normalization, cuda):
        """
        Return the validataion set. Only for the inductive task.
        Currently behave the same with get_test_set
        """
        return self.get_test_set(normalization, cuda)

    def get_label_and_idxes(self, cuda):
        """
        Return all labels and indexes.
        """
        if cuda:
            return self.labels_torch.cuda(), self.idx_train_torch.cuda(), self.idx_val_torch.cuda(), self.idx_test_torch.cuda()
        return self.labels_torch, self.idx_train_torch, self.idx_val_torch, self.idx_test_torch


# layers . py 

# In[14]:


import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch import nn
import torch.nn.functional as F

device = torch.device("cuda:0")

class GraphConvolutionBS(Module):
    """
    GCN Layer with BN, Self-loop and Res connection.
    """

    def __init__(self, in_features, out_features, activation=lambda x: x, withbn=True, withloop=True, bias=True,
                 res=False):
        """
        Initial function.
        :param in_features: the input feature dimension.
        :param out_features: the output feature dimension.
        :param activation: the activation function.
        :param withbn: using batch normalization.
        :param withloop: using self feature modeling.
        :param bias: enable bias.
        :param res: enable res connections.
        """
        super(GraphConvolutionBS, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma = activation
        self.res = res

        # Parameter setting.
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        # Is this the best practice or not?
        if withloop:
            self.self_weight = Parameter(torch.FloatTensor(in_features, out_features))
        else:
            self.register_parameter("self_weight", None)

        if withbn:
            self.bn = torch.nn.BatchNorm1d(out_features)
        else:
            self.register_parameter("bn", None)

        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.self_weight is not None:
            stdv = 1. / math.sqrt(self.self_weight.size(1))
            self.self_weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)

        # Self-loop
        if self.self_weight is not None:
            output = output + torch.mm(input, self.self_weight)

        if self.bias is not None:
            output = output + self.bias
        # BN
        if self.bn is not None:
            output = self.bn(output)
        # Res
        if self.res:
            return self.sigma(output) + input
        else:
            return self.sigma(output)

    def __repr__(self):
        return self.__class__.__name__ + ' ('                + str(self.in_features) + ' -> '                + str(self.out_features) + ')'

class GraphBaseBlock(Module):
    """
    The base block for Multi-layer GCN / ResGCN / Dense GCN 
    """

    def __init__(self, in_features, out_features, nbaselayer,
                 withbn=True, withloop=True, activation=F.relu, dropout=True,
                 aggrmethod="concat", dense=False):
        """
        The base block for constructing DeepGCN model.
        :param in_features: the input feature dimension.
        :param out_features: the hidden feature dimension.
        :param nbaselayer: the number of layers in the base block.
        :param withbn: using batch normalization in graph convolution.
        :param withloop: using self feature modeling in graph convolution.
        :param activation: the activation function, default is ReLu.
        :param dropout: the dropout ratio.
        :param aggrmethod: the aggregation function for baseblock, can be "concat" and "add". For "resgcn", the default
                           is "add", for others the default is "concat".
        :param dense: enable dense connection
        """
        super(GraphBaseBlock, self).__init__()
        self.in_features = in_features
        self.hiddendim = out_features
        self.nhiddenlayer = nbaselayer
        self.activation = activation
        self.aggrmethod = aggrmethod
        self.dense = dense
        self.dropout = dropout
        self.withbn = withbn
        self.withloop = withloop
        self.hiddenlayers = nn.ModuleList()
        self.__makehidden()

        if self.aggrmethod == "concat" and dense == False:
            self.out_features = in_features + out_features
        elif self.aggrmethod == "concat" and dense == True:
            self.out_features = in_features + out_features * nbaselayer
        elif self.aggrmethod == "add":
            if in_features != self.hiddendim:
                raise RuntimeError("The dimension of in_features and hiddendim should be matched in add model.")
            self.out_features = out_features
        elif self.aggrmethod == "nores":
            self.out_features = out_features
        else:
            raise NotImplementedError("The aggregation method only support 'concat','add' and 'nores'.")

    def __makehidden(self):
        # for i in xrange(self.nhiddenlayer):
        for i in range(self.nhiddenlayer):
            if i == 0:
                layer = GraphConvolutionBS(self.in_features, self.hiddendim, self.activation, self.withbn,
                                           self.withloop)
            else:
                layer = GraphConvolutionBS(self.hiddendim, self.hiddendim, self.activation, self.withbn, self.withloop)
            self.hiddenlayers.append(layer)

    def _doconcat(self, x, subx):
        if x is None:
            return subx
        if self.aggrmethod == "concat":
            return torch.cat((x, subx), 1)
        elif self.aggrmethod == "add":
            return x + subx
        elif self.aggrmethod == "nores":
            return x

    def forward(self, input, adj):
        x = input
        denseout = None
        # Here out is the result in all levels.
        for gc in self.hiddenlayers:
            denseout = self._doconcat(denseout, x)
            x = gc(x, adj)
            x = F.dropout(x, self.dropout, training=self.training)

        if not self.dense:
            return self._doconcat(x, input)
        return self._doconcat(x, denseout)

    def get_outdim(self):
        return self.out_features

    def __repr__(self):
        return "%s %s (%d - [%d:%d] > %d)" % (self.__class__.__name__,
                                              self.aggrmethod,
                                              self.in_features,
                                              self.hiddendim,
                                              self.nhiddenlayer,
                                              self.out_features)

class MultiLayerGCNBlock(Module):
    """
    Muti-Layer GCN with same hidden dimension.
    """

    def __init__(self, in_features, out_features, nbaselayer,
                 withbn=True, withloop=True, activation=F.relu, dropout=True,
                 aggrmethod=None, dense=None):
        """
        The multiple layer GCN block.
        :param in_features: the input feature dimension.
        :param out_features: the hidden feature dimension.
        :param nbaselayer: the number of layers in the base block.
        :param withbn: using batch normalization in graph convolution.
        :param withloop: using self feature modeling in graph convolution.
        :param activation: the activation function, default is ReLu.
        :param dropout: the dropout ratio.
        :param aggrmethod: not applied.
        :param dense: not applied.
        """
        super(MultiLayerGCNBlock, self).__init__()
        self.model = GraphBaseBlock(in_features=in_features,
                                    out_features=out_features,
                                    nbaselayer=nbaselayer,
                                    withbn=withbn,
                                    withloop=withloop,
                                    activation=activation,
                                    dropout=dropout,
                                    dense=False,
                                    aggrmethod="nores")

    def forward(self, input, adj):
        return self.model.forward(input, adj)

    def get_outdim(self):
        return self.model.get_outdim()

    def __repr__(self):
        return "%s %s (%d - [%d:%d] > %d)" % (self.__class__.__name__,
                                              self.aggrmethod,
                                              self.model.in_features,
                                              self.model.hiddendim,
                                              self.model.nhiddenlayer,
                                              self.model.out_features)

class ResGCNBlock(Module):
    """
    The multiple layer GCN with residual connection block.
    """

    def __init__(self, in_features, out_features, nbaselayer,
                 withbn=True, withloop=True, activation=F.relu, dropout=True,
                 aggrmethod=None, dense=None):
        """
        The multiple layer GCN with residual connection block.
        :param in_features: the input feature dimension.
        :param out_features: the hidden feature dimension.
        :param nbaselayer: the number of layers in the base block.
        :param withbn: using batch normalization in graph convolution.
        :param withloop: using self feature modeling in graph convolution.
        :param activation: the activation function, default is ReLu.
        :param dropout: the dropout ratio.
        :param aggrmethod: not applied.
        :param dense: not applied.
        """
        super(ResGCNBlock, self).__init__()
        self.model = GraphBaseBlock(in_features=in_features,
                                    out_features=out_features,
                                    nbaselayer=nbaselayer,
                                    withbn=withbn,
                                    withloop=withloop,
                                    activation=activation,
                                    dropout=dropout,
                                    dense=False,
                                    aggrmethod="add")

    def forward(self, input, adj):
        return self.model.forward(input, adj)

    def get_outdim(self):
        return self.model.get_outdim()

    def __repr__(self):
        return "%s %s (%d - [%d:%d] > %d)" % (self.__class__.__name__,
                                              self.aggrmethod,
                                              self.model.in_features,
                                              self.model.hiddendim,
                                              self.model.nhiddenlayer,
                                              self.model.out_features)

class DenseGCNBlock(Module):
    """
    The multiple layer GCN with dense connection block.
    """

    def __init__(self, in_features, out_features, nbaselayer,
                 withbn=True, withloop=True, activation=F.relu, dropout=True,
                 aggrmethod="concat", dense=True):
        """
        The multiple layer GCN with dense connection block.
        :param in_features: the input feature dimension.
        :param out_features: the hidden feature dimension.
        :param nbaselayer: the number of layers in the base block.
        :param withbn: using batch normalization in graph convolution.
        :param withloop: using self feature modeling in graph convolution.
        :param activation: the activation function, default is ReLu.
        :param dropout: the dropout ratio.
        :param aggrmethod: the aggregation function for the output. For denseblock, default is "concat".
        :param dense: default is True, cannot be changed.
        """
        super(DenseGCNBlock, self).__init__()
        self.model = GraphBaseBlock(in_features=in_features,
                                    out_features=out_features,
                                    nbaselayer=nbaselayer,
                                    withbn=withbn,
                                    withloop=withloop,
                                    activation=activation,
                                    dropout=dropout,
                                    dense=True,
                                    aggrmethod=aggrmethod)

    def forward(self, input, adj):
        return self.model.forward(input, adj)

    def get_outdim(self):
        return self.model.get_outdim()

    def __repr__(self):
        return "%s %s (%d - [%d:%d] > %d)" % (self.__class__.__name__,
                                              self.aggrmethod,
                                              self.model.in_features,
                                              self.model.hiddendim,
                                              self.model.nhiddenlayer,
                                              self.model.out_features)

class InecptionGCNBlock(Module):
    """
    The multiple layer GCN with inception connection block.
    """

    def __init__(self, in_features, out_features, nbaselayer,
                 withbn=True, withloop=True, activation=F.relu, dropout=True,
                 aggrmethod="concat", dense=False):
        """
        The multiple layer GCN with inception connection block.
        :param in_features: the input feature dimension.
        :param out_features: the hidden feature dimension.
        :param nbaselayer: the number of layers in the base block.
        :param withbn: using batch normalization in graph convolution.
        :param withloop: using self feature modeling in graph convolution.
        :param activation: the activation function, default is ReLu.
        :param dropout: the dropout ratio.
        :param aggrmethod: the aggregation function for baseblock, can be "concat" and "add". For "resgcn", the default
                           is "add", for others the default is "concat".
        :param dense: not applied. The default is False, cannot be changed.
        """
        super(InecptionGCNBlock, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.hiddendim = out_features
        self.nbaselayer = nbaselayer
        self.activation = activation
        self.aggrmethod = aggrmethod
        self.dropout = dropout
        self.withbn = withbn
        self.withloop = withloop
        self.midlayers = nn.ModuleList()
        self.__makehidden()

        if self.aggrmethod == "concat":
            self.out_features = in_features + out_features * nbaselayer
        elif self.aggrmethod == "add":
            if in_features != self.hiddendim:
                raise RuntimeError("The dimension of in_features and hiddendim should be matched in 'add' model.")
            self.out_features = out_features
        else:
            raise NotImplementedError("The aggregation method only support 'concat', 'add'.")

    def __makehidden(self):
        # for j in xrange(self.nhiddenlayer):
        for j in range(self.nbaselayer):
            reslayer = nn.ModuleList()
            # for i in xrange(j + 1):
            for i in range(j + 1):
                if i == 0:
                    layer = GraphConvolutionBS(self.in_features, self.hiddendim, self.activation, self.withbn,
                                               self.withloop)
                else:
                    layer = GraphConvolutionBS(self.hiddendim, self.hiddendim, self.activation, self.withbn,
                                               self.withloop)
                reslayer.append(layer)
            self.midlayers.append(reslayer)

    def forward(self, input, adj):
        x = input
        for reslayer in self.midlayers:
            subx = input
            for gc in reslayer:
                subx = gc(subx, adj)
                subx = F.dropout(subx, self.dropout, training=self.training)
            x = self._doconcat(x, subx)
        return x

    def get_outdim(self):
        return self.out_features

    def _doconcat(self, x, subx):
        if self.aggrmethod == "concat":
            return torch.cat((x, subx), 1)
        elif self.aggrmethod == "add":
            return x + subx

    def __repr__(self):
        return "%s %s (%d - [%d:%d] > %d)" % (self.__class__.__name__,
                                              self.aggrmethod,
                                              self.in_features,
                                              self.hiddendim,
                                              self.nbaselayer,
                                              self.out_features)

class Dense(Module):


    """
    Simple Dense layer, Do not consider adj.
    """

    def __init__(self, in_features, out_features, activation=lambda x: x, bias=True, res=False):
        super(Dense, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma = activation
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.res = res
        self.bn = nn.BatchNorm1d(out_features)
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        output = torch.mm(input, self.weight)
        if self.bias is not None:
            output = output + self.bias
        output = self.bn(output)
        return self.sigma(output)

    def __repr__(self):
        return self.__class__.__name__ + ' ('                + str(self.in_features) + ' -> '                + str(self.out_features) + ')'

class GCNModel(nn.Module):
    def __init__(self,
                 nfeat,
                 nhid,
                 nclass,
                 nhidlayer,
                 dropout,
                 baseblock="mutigcn",
                 inputlayer="gcn",
                 outputlayer="gcn",
                 nbaselayer=0,
                 activation=lambda x: x,
                 withbn=True,
                 withloop=True,
                 aggrmethod="add",
                 mixmode=False):
 
        super(GCNModel, self).__init__()
        self.mixmode = mixmode
        self.dropout = dropout

        if baseblock == "resgcn":
            self.BASEBLOCK = ResGCNBlock
        elif baseblock == "densegcn":
            self.BASEBLOCK = DenseGCNBlock
        elif baseblock == "mutigcn":
            self.BASEBLOCK = MultiLayerGCNBlock
        elif baseblock == "inceptiongcn":
            self.BASEBLOCK = InecptionGCNBlock
        else:
            raise NotImplementedError("Current baseblock %s is not supported." % (baseblock))
        if inputlayer == "gcn":
            # input gc
            self.ingc = GraphConvolutionBS(nfeat, nhid, activation, withbn, withloop)
            baseblockinput = nhid
        elif inputlayer == "none":
            self.ingc = lambda x: x
            baseblockinput = nfeat
        else:
            self.ingc = Dense(nfeat, nhid, activation)
            baseblockinput = nhid

        outactivation = lambda x: x
        if outputlayer == "gcn":
            self.outgc = GraphConvolutionBS(baseblockinput, nclass, outactivation, withbn, withloop)
        # elif outputlayer ==  "none": #here can not be none
        #    self.outgc = lambda x: x 
        else:
            self.outgc = Dense(nhid, nclass, activation)

        # hidden layer
        self.midlayer = nn.ModuleList()

        for i in range(nhidlayer):
            gcb = self.BASEBLOCK(in_features=baseblockinput,
                                 out_features=nhid,
                                 nbaselayer=nbaselayer,
                                 withbn=withbn,
                                 withloop=withloop,
                                 activation=activation,
                                 dropout=dropout,
                                 dense=False,
                                 aggrmethod=aggrmethod)
            self.midlayer.append(gcb)
            baseblockinput = gcb.get_outdim()
        # output gc
        outactivation = lambda x: x  # we donot need nonlinear activation here.
        self.outgc = GraphConvolutionBS(baseblockinput, nclass, outactivation, withbn, withloop)

        self.reset_parameters()
        if mixmode:
            self.midlayer = self.midlayer.to(device)
            self.outgc = self.outgc.to(device)

    def reset_parameters(self):
        pass

    def forward(self, fea, adj):
        # input
        if self.mixmode:
            x = self.ingc(fea, adj.cpu())
        else:
            x = self.ingc(fea, adj)

        x = F.dropout(x, self.dropout, training=self.training)
        if self.mixmode:
            x = x.to(device)

        # mid block connections
        # for i in xrange(len(self.midlayer)):
        for i in range(len(self.midlayer)):
            midgc = self.midlayer[i]
            x = midgc(x, adj)
        # output, no relu and dropput here.
        x = self.outgc(x, adj)
        x = F.log_softmax(x, dim=1)
        return x

class GCNFlatRes(nn.Module):
    """
    (Legacy)
    """
    def __init__(self, nfeat, nhid, nclass, withbn, nreslayer, dropout, mixmode=False):
        super(GCNFlatRes, self).__init__()

        self.nreslayer = nreslayer
        self.dropout = dropout
        self.ingc = GraphConvolution(nfeat, nhid, F.relu)
        self.reslayer = GCFlatResBlock(nhid, nclass, nhid, nreslayer, dropout)
        self.reset_parameters()

    def reset_parameters(self):
        # stdv = 1. / math.sqrt(self.attention.size(1))
        # self.attention.data.uniform_(-stdv, stdv)
        # print(self.attention)
        pass

    def forward(self, input, adj):
        x = self.ingc(input, adj)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.reslayer(x, adj)
        # x = F.dropout(x, self.dropout, training=self.training)
        return F.log_softmax(x, dim=1)


# Train_new.py

# In[15]:


no_cuda = True  # Disables CUDA training.
fastmode = False  # Disable validation during training.
seed = 44  # Random seed.
epochs = 200  # Number of epochs to train.
lr = 0.02  # Initial learning rate.
lradjust = False  # Enable learning rate adjust. (ReduceLROnPlateau or Linear Reduce)
weight_decay = 5e-4  # Weight decay (L2 loss on parameters).
mixmode = False  # Enable CPU-GPU mixing mode.
warm_start = ""  # The model name to be loaded for warm start.
debug = True  # Enable the detailed training output.

# dataset = "reed98"  # The data set
# datapath = "data_cora/"  # The data path.

dataset = DATASET_NAME  # The data set
datapath = DIR  # The data path.



early_stopping = 0  # The patience of early stopping. Set to 0 to disable.
no_tensorboard = False  # Disable writing logs to TensorBoard.

# Model parameters
model_type = None  # Choose the model to be trained (mutigcn, resgcn, densegcn, inceptiongcn).
inputlayer = "gcn"  # The input layer of the model.
outputlayer = "gcn"  # The output layer of the model.
hidden = 128  # Number of hidden units.
dropout = 0.5  # Dropout rate (1 - keep probability).
withbn = False  # Enable Batch Norm in GCN.
withloop = False  # Enable loop layer in GCN.
nhiddenlayer = 1  # The number of hidden layers.
normalization = "AugNormAdj"  # The normalization on the adjacency matrix.
sampling_percent = 0.20  # The percent of preserved edges. If set to 1, no sampling is done.
nbaseblocklayer = 1  # The number of layers in each base block.
aggrmethod = "default"  # Aggregation method for layer aggregation (add, concat).
task_type = "full"  # The node classification task type (full, semi). Only valid for specific datasets.
_type = 'gcn'


# In[16]:


is_train = True 
cuda = not no_cuda and torch.cuda.is_available()
mixmode = no_cuda and mixmode and torch.cuda.is_available()
if aggrmethod == "default":
    if type == "resgcn":
        aggrmethod = "add"
    else:
        aggrmethod = "concat"
if fastmode and early_stopping > 0:
    early_stopping = 0
    print("In the fast mode, early_stopping is not valid option. Setting early_stopping = 0.")
if _type == "mutigcn":
    print("For the multi-layer gcn model, the aggrmethod is fixed to nores and nhiddenlayers = 1.")
    nhiddenlayer = 1
    aggrmethod = "nores"

# random seed setting
np.random.seed(seed)
torch.manual_seed(seed)
if cuda or mixmode:
    torch.cuda.manual_seed(seed)
""
# should we need fix random seed here?
sampler = Sampler(dataset, datapath,task_type)

# get labels and indexes
labels, idx_train, idx_val, idx_test = sampler.get_label_and_idxes(cuda)
nfeat = sampler.nfeat
nclass = sampler.nclass
print("nclass: %d\tnfea:%d" % (nclass, nfeat))

# The model
model = GCNModel(nfeat=nfeat,
                 nhid=hidden,
                 nclass=nclass,
                 nhidlayer=nhiddenlayer,
                 dropout=dropout,
                 inputlayer=inputlayer,
                 outputlayer=outputlayer,
                 nbaselayer=nbaseblocklayer,
                 activation=F.relu,
                 withbn=withbn,
                 withloop=withloop,
                 aggrmethod=aggrmethod,
                 mixmode=mixmode)

optimizer = optim.Adam(model.parameters(),
                       lr=lr, weight_decay=weight_decay)


scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[200, 300, 400, 500, 600, 700], gamma=0.5)
if cuda:
    model.cuda()
if cuda or mixmode:
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()

if warm_start is not None and warm_start != "":
    early_stopping = EarlyStopping(fname=warm_start, verbose=False)
    #print("Restore checkpoint from %s" % (early_stopping.fname))
    model.load_state_dict(early_stopping.load_checkpoint())
if early_stopping > 0:
    early_stopping = EarlyStopping(patience=early_stopping, verbose=False)
    #print("Model is saving to: %s" % (early_stopping.fname))
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


# In[ ]:





# In[17]:


# define the training function.
def train(epoch, train_adj, train_fea, idx_train, val_adj=None, val_fea=None):
    if val_adj is None:
        val_adj = train_adj
        val_fea = train_fea

    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(train_fea, train_adj)
    # special for reddit
    if sampler.learning_type == "inductive":
        loss_train = F.nll_loss(output, labels[idx_train])
        acc_train = accuracy(output, labels[idx_train])
    else:
        loss_train = F.nll_loss(output[idx_train], labels[idx_train])
        acc_train = accuracy(output[idx_train], labels[idx_train])

    loss_train.backward()
    optimizer.step()
    train_t = time.time() - t
    val_t = time.time()
    # We can not apply the fastmode for the reddit dataset.
    # if sampler.learning_type == "inductive" or not args.fastmode:

    if early_stopping > 0 and sampler.dataset != "reddit":
        loss_val = F.nll_loss(output[idx_val], labels[idx_val]).item()
        early_stopping(loss_val, model)

    if not fastmode:
        #    # Evaluate validation set performance separately,
        #    # deactivates dropout during validation run.
        model.eval()
        output = model(val_fea, val_adj)
        loss_val = F.nll_loss(output[idx_val], labels[idx_val]).item()
        acc_val = accuracy(output[idx_val], labels[idx_val]).item()
        if sampler.dataset == "reddit":
            early_stopping(loss_val, model)
    else:
        loss_val = 0
        acc_val = 0

    if lradjust:
        scheduler.step()

    val_t = time.time() - val_t
    return (loss_train.item(), acc_train.item(), loss_val, acc_val, get_lr(optimizer), train_t, val_t)


def test(test_adj, test_fea):
    model.eval()
    output = model(test_fea, test_adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    #auc_test = roc_auc_compute_fn(output[idx_test], labels[idx_test])
    auc_test = 0
    if debug:
        print("Test set results:",
              "loss= {:.4f}".format(loss_test.item()),
              "auc= {:.4f}".format(auc_test),
              "accuracy= {:.4f}".format(acc_test.item()))
        print("accuracy=%.5f" % (acc_test.item()))
    return (loss_test.item(), acc_test.item())


# Train model
t_total = time.time()
loss_train = np.zeros((epochs,))
acc_train = np.zeros((epochs,))
loss_val = np.zeros((epochs,))
acc_val = np.zeros((epochs,))

sampling_t = 0

EpochTimes = []

for epoch in range(epochs):
    input_idx_train = idx_train
    sampling_t = time.time()
    # no sampling
    # randomedge sampling if args.sampling_percent >= 1.0, it behaves the same as stub_sampler.
    (train_adj, train_fea) = sampler.randomedge_sampler(percent=sampling_percent, normalization=normalization,
                                                        cuda=cuda)
    if mixmode:
        train_adj = train_adj.cuda()

    sampling_t = time.time() - sampling_t
    
    EpochTimes.append(sampling_t)
    
    # The validation set is controlled by idx_val
    # if sampler.learning_type == "transductive":
    if is_train:
        outputs = train(epoch, train_adj, train_fea, input_idx_train)
    else:
        (val_adj, val_fea) = sampler.get_test_set(normalization=normalization, cuda=cuda)
        if mixmode:
            val_adj = val_adj.cuda()
        outputs = train(epoch, train_adj, train_fea, input_idx_train, val_adj, val_fea)
    
    if args.log and debug and epoch % 1 == 0:
        print('Epoch: {:04d}'.format(epoch + 1),
              'loss_train: {:.4f}'.format(outputs[0]),
              'acc_train: {:.4f}'.format(outputs[1]),
              'loss_val: {:.4f}'.format(outputs[2]),
              'acc_val: {:.4f}'.format(outputs[3]),
              'cur_lr: {:.5f}'.format(outputs[4]),
              's_time: {:.4f}s'.format(sampling_t),
              't_time: {:.4f}s'.format(outputs[5]),
              'v_time: {:.4f}s'.format(outputs[6]))
    
    loss_train[epoch], acc_train[epoch], loss_val[epoch], acc_val[epoch] = outputs[0], outputs[1], outputs[2], outputs[
        3]

    if early_stopping > 0 and early_stopping.early_stop:
        if args.log:
            print("Early stopping.")
        model.load_state_dict(early_stopping.load_checkpoint())
        break

if early_stopping > 0:
    model.load_state_dict(early_stopping.load_checkpoint())

if debug:
    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Testing
(test_adj, test_fea) = sampler.get_test_set(normalization=normalization, cuda=cuda)
if mixmode:
    test_adj = test_adj.cuda()
(loss_test, acc_test) = test(test_adj, test_fea)
if args.log:
    print("%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f" % (
loss_train[-1], loss_val[-1], loss_test, acc_train[-1], acc_val[-1], acc_test))


print("Mean Epoch Times:", np.mean(EpochTimes))


# Mean and Standard Deviation

# In[18]:


a= [0.51351, 0.54054, 0.59459]
mean_a = np.mean(a)
std_dev_a = np.std(a)
print(f'{mean_a:.4f} +/- {std_dev_a:.4f}')


# In[ ]:




