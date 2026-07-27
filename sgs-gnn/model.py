import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GINConv, SAGEConv, ChebConv, GAT, GIN

# Define the MLP for edge probability with dropout
class EdgeProbMLP(nn.Module):
    def __init__(self, in_channels, hidden_dim, dropout_prob=0.2):
        super(EdgeProbMLP, self).__init__()
        self.gcn1 = GCNConv(in_channels, hidden_dim) #sample 1 hop neighborhood
        self.dropout = nn.Dropout(dropout_prob)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim) #sample 2 hop neighborhood
        self.fc1 = nn.Linear(2 * hidden_dim, hidden_dim)
        # self.fc_residual = nn.Linear(2 * hidden_dim, hidden_dim)  # For residual connection
        #self.fc1 = nn.Linear(2 * in_channels, hidden_dim) #if straightforward link
        self.dropout = nn.Dropout(dropout_prob)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, node_features, edge_index):

        out = F.relu(self.gcn1(node_features, edge_index))
        out = self.dropout(out)
        out = F.relu(self.gcn2(out, edge_index))
        
        x = out[edge_index[0]]
        y = out[edge_index[1]]
        
        # x = node_features[edge_index[0]]
        # y = node_features[edge_index[1]]

        edge_features = torch.cat([x*y,x-y], dim=1) #x*y|x-y
        x = F.relu(self.fc1(edge_features))
        # x = x + self.fc_residual(edge_features) #  # Add residual connection
        x = self.dropout(x)
        prob = torch.sigmoid(self.fc2(x))
        # prob.requires_grad_(True)
        return prob
    
# class EdgeProbMLP(nn.Module):
#     def __init__(self, in_channels, hidden_dim):
#         super(EdgeProbMLP, self).__init__()
#         self.fc1 = nn.Linear(2 * in_channels, hidden_dim)
#         self.fc2 = nn.Linear(hidden_dim, 1)

#     def forward(self, node_features, edge_index):
#         edge_features = torch.cat([node_features[edge_index[0]], node_features[edge_index[1]]], dim=1)
#         x = F.relu(self.fc1(edge_features))
#         prob = torch.sigmoid(self.fc2(x))
#         #prob = F.softmax(self.fc2(x),dim=0)
#         #prob = F.relu(self.fc2(x))
#         return prob
    
#     def reset_parameters(self):
#         self.fc1.reset_parameters()
#         self.fc2.reset_parameters()

# Define the overall model with Edge Probabilities and GCNConv (GNN) with dropout


class GNNModel(nn.Module):
    def __init__(self, in_channels, hidden_dim, num_classes, dropout_prob=0.3):
        super(GNNModel, self).__init__()
        self.edge_prob_mlp = EdgeProbMLP(in_channels, hidden_dim, dropout_prob)
        # self.edge_prob_mlp = EdgeProbMLP(in_channels, hidden_dim)
        self.gcn1 = GCNConv(in_channels, hidden_dim)
        self.dropout = nn.Dropout(dropout_prob)
        self.gcn2 = GCNConv(hidden_dim, num_classes)

    def forward(self, data, edge_index, edge_weight=None):
        x = F.relu(self.gcn1(data.x, edge_index, edge_weight))
        x = self.dropout(x)
        out = self.gcn2(x, edge_index, edge_weight)
        return out


class GINModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_dim, num_classes, dropout_prob=0.3):
        super(GINModel, self).__init__()
        self.edge_prob_mlp = EdgeProbMLP(in_channels, hidden_dim, dropout_prob)        
        self.dropout_prob = dropout_prob
        self.GIN = GIN(in_channels = in_channels,
                        hidden_channels = hidden_dim, 
                        num_layers = 2, out_channels = num_classes,
                        dropout = dropout_prob, 
                        act = 'relu')
                    
        
    def forward(self, data, edge_index, edge_weight=None):
        x = self.GIN(data.x, edge_index, edge_weight=edge_weight)
        return x


class GATModel(torch.nn.Module):        
    def __init__(self, in_channels, hidden_dim, num_classes, dropout_prob=0.3, heads=8):
        super(GATModel, self).__init__()
        self.edge_prob_mlp = EdgeProbMLP(in_channels, hidden_dim, dropout_prob)        
        self.dropout_prob = dropout_prob
        
        self.GAT =  GAT(in_channels = in_channels,
                        hidden_channels = hidden_dim, 
                        num_layers = 2, out_channels = num_classes,
                        dropout = dropout_prob, 
                        act = 'relu')
        
    def forward(self, data, edge_index, edge_weight=None):
        x = self.GAT(data.x, edge_index, edge_weight=edge_weight)        
        return x


class ChebModel(nn.Module):
    def __init__(self, in_channels, hidden_dim, num_classes, dropout_prob=0.3):
        super(ChebModel, self).__init__()
        self.edge_prob_mlp = EdgeProbMLP(in_channels, hidden_dim, dropout_prob)
        self.dropout_prob = dropout_prob
        
        self.gcn1 = ChebConv(in_channels, hidden_dim, K=1, normalization='sym')
        self.dropout = nn.Dropout(dropout_prob)
        self.gcn2 = ChebConv(hidden_dim, num_classes, K=1, normalization='sym')

    def forward(self, data, edge_index, edge_weight=None):
        x = F.relu(self.gcn1(data.x, edge_index, edge_weight))
        x = self.dropout(x)
        out = self.gcn2(x, edge_index, edge_weight)
        return out