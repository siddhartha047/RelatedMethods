import os
import os.path as osp

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import Compose, ToSparseTensor, ToUndirected, AddSelfLoops, NormalizeFeatures
from torch_geometric.utils import to_undirected, add_self_loops, remove_isolated_nodes, remove_self_loops
from ipynb.fs.full.SGSLoadDataset import LOAD_DATASET
import Notebooks.DeviceDir as DeviceDir

DIR, RESULTS_DIR = DeviceDir.get_directory()
device, NUM_PROCESSORS = DeviceDir.get_device()

import numpy as np
import random

from logger import Logger
from args import parser_loader
from MoG import MoG

def train(model:MoG,features,indices,labels,values,shape,train_idx,temp,optimizer,mask=None):
    """
    :params model: MoG
    :params values: value of edge(all one)
    :params shape: shape of adj matrix
    :parsms temp: temp of sparsification learner
    
    :return: loss
    """
    model.train()
    optimizer.zero_grad()
    mask,add_loss = model.learner(x = features, edge_index = indices, 
                                  temp = temp,shape = shape,
                                  edge_attr = values, training = True)  # masks:size(num_edges)
    output = model.gnn(features, indices ,mask) + add_loss
    loss = F.nll_loss(output[train_idx], labels[train_idx])
    loss.backward()   
    optimizer.step()
    return loss.item(),mask
    

@torch.no_grad()
def test(model:MoG,features,indices,labels,values,shape,split_idx,temp,mask=None):
    model.eval()
    mask,add_loss = model.learner(x = features, edge_index = indices, 
                                  temp = temp,shape = shape,
                                  edge_attr = values, training = False)  # mask:size(num_edges)
    output = model.gnn(features, indices, mask)
    sparsity = torch.nonzero(mask).size(0)/mask.numel()
    y_pred = output.argmax(dim=-1, keepdim=False)
    train_acc = y_pred[split_idx['train']].eq(labels[split_idx['train']]).sum().item()/split_idx['train'].sum().item()
    valid_acc = y_pred[split_idx['valid']].eq(labels[split_idx['valid']]).sum().item()/split_idx['valid'].sum().item()
    test_acc = y_pred[split_idx['test']].eq(labels[split_idx['test']]).sum().item()/split_idx['test'].sum().item()
    
    return train_acc, valid_acc, test_acc, sparsity
    

def save_edge_index(edge_index,mask,sparsity):
    edge_index = edge_index[:,mask.bool()]
    torch.save({'edge_index':edge_index},f'edge_index/{sparsity:.2f}_edge_index.pt')

def main():
    args = parser_loader()
    print(args)
    if args['device'] is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args['device']
    fix_seed(args['seed'])
    device = f"cuda:{args['device']}" if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    dataset_name = args['dataset']
    
    #dataset = Planetoid(root='/scratch/gilbreth/das90/Dataset/Planetoid', name=dataset_name, transform=NormalizeFeatures())
    # path= osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', f'{dataset_name}.pt')
    # dataset = torch.load(path)
    
    #data = dataset[0]
    #print(data)

    data, dataset = LOAD_DATASET(DIR, dataset_name)

    #data.adj_t = 

    #data.adj_t = data.adj_t.to_symmetric()
    data = data.to(device)

    split_idx = {'train':data.train_mask,'valid':data.val_mask,'test':data.test_mask}
    train_idx = split_idx['train'].to(device)

    # transform the data
    features = data.x
    labels = data.y
    #row,col,_= data.adj_t.coo()
    #indices = torch.stack([row,col],dim=0)

    # all_connected_nodes = torch.unique(data.edge_index.flatten())
    # # Step 2: Generate all nodes from 1 to num_nodes
    # all_nodes = torch.arange(0, data.num_nodes).to(device)
    # mask2 = ~torch.isin(all_nodes, all_connected_nodes)
    # isolated_vertices = all_nodes[mask2]
    # print("Isolated vertices:", isolated_vertices.tolist())

    data.edge_index = remove_self_loops(data.edge_index)[0]

    # _, _, isolated_mask = remove_isolated_nodes(data.edge_index,num_nodes=data.num_nodes)
    # isolated_indices = torch.nonzero(~isolated_mask).squeeze()
    # r = []
    # c = []

    # print(isolated_indices)

    # for i in isolated_indices.tolist():
    #     j = np.random.randint(0,data.num_nodes)
    #     while (i==j):
    #         j = np.random.randint(0,data.num_nodes)
    #     r.append(i)
    #     c.append(j)        
    # iso_indcies = torch.stack([torch.LongTensor(r),torch.LongTensor(c)],dim=0).to(device)

    # data.edge_index = torch.cat((data.edge_index,iso_indcies),dim=1)

    # print(data.edge_index.shape)

    data.edge_index  =  add_self_loops(to_undirected(data.edge_index))[0]
    indices = data.edge_index

    

    num_classes = max(data.y).item()+1

    values = torch.ones((data.num_edges,),device=device) # ones
    shape = (data.num_nodes,data.num_nodes)
    #topo_val =data.topo_val
    topo_val = 1

    logger = Logger(args['runs'], args)
    loss_5_epochs = []

    for run in range(args['runs']):
        # define the model
        model = MoG(data.num_features, num_classes, args, device)
        model = model.to(device)
        
        
        optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'])
        best_val_acc, best_test_acc, best_sparsity = 0, 0, 1
        if args['use_topo']:
            model.learner.topo_val = topo_val
        else:
            model.learner.topo_val = None
        for epoch in range(1, 1 + args['epochs']):
            # calculate the temperature of sparsity learner
            if (epoch-1) % args["temp_N"] == 0:
                decay_temp = np.exp(-1*args["temp_r"]*epoch)
                temp = max(0.05, decay_temp)
            
            # train and test
            loss,mask = train(model,features,indices,labels,values,shape,train_idx,temp,optimizer)
            result = test(model,features,indices,labels,values,shape,split_idx,temp)
            train_acc, valid_acc, test_acc, sparsity = result
                 
            logger.add_result(run, result)
            if valid_acc > best_val_acc:
                best_val_acc = valid_acc
                best_test_acc = test_acc
                best_sparsity = sparsity
                best_mask = mask
                
            if epoch >= 5 and np.std(logger.results[run][-5:]) < args['convergence']:
                            num_iteration = epoch + 1
                            print(f"Convergence achieved at Epoch: {num_iteration}")
                            break
                        
            if epoch % 50 == 0:
                print(result)
                print(f'Run: {run + 1:04d}, '
                      f'Epoch: {epoch:02d}, '
                      f'Loss: {loss:.4f}, '
                      f'Train: {100 * train_acc:.2f}%, '
                      f'Valid: {100 * valid_acc:.2f}%, '
                      f'Test: {100 * test_acc:.2f}%, '
                      f'Best Valid: {100 * best_val_acc:.2f}%, '
                      f'Best Test: {100 * best_test_acc:.4f}%, '
                      f'Best Sparsity: {100 * best_sparsity:.2f}%,')
            
            
        #save_edge_index(indices,best_mask,best_sparsity)
        
        logger.print_statistics(run)
    logger.print_statistics()

def fix_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)


if __name__ == "__main__":
    main()
