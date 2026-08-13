########################################

# #%%writefile magic_functions.py
# import pprint
# import os.path as osp
# import matplotlib.pyplot as plt
# from tqdm.notebook import tqdm
# import numpy as np
# import networkx as nx
# from scipy import sparse, stats
# from numpy import inf

# from scipy.sparse import identity
# from scipy.sparse import csgraph
# from scipy import linalg
# from scipy.sparse import csr_matrix
# #from scipy.sparse import linalg

# import ipynb.fs.full.utils.BarbellGraph as BGraph

# G, pos = BGraph.generate_barbell(10,10)
# BGraph.draw_graph(G,pos)

# X=np.zeros((len(pos),2))
# for key,value in pos.items():
#     X[key]=value
    
    
# minx=np.min(X,0); print(minx)
# X=X-minx
# minx=np.min(X,0); print(minx)
# X_pos={key:X[key] for key,value in pos.items()}
# #BGraph.draw_graph(G,X_pos)


########################################
import numpy as np
from apricot import FeatureBasedSelection, MaxCoverageSelection, FacilityLocationSelection
from apricot import GraphCutSelection, SumRedundancySelection, SaturatedCoverageSelection, MixtureSelection






func=FeatureBasedSelection
print("Used func: ", func)

########################################

def GetLocalEdgesParallel(org_u, u, nodes, features, K):
    
    #print(u, nodes)
    
    if len(nodes)-1<=K:
        nodes.remove(u)
        return org_u, np.array(nodes)
    
    model = func(K, initial_subset=[u], optimizer='lazy')
    model.fit(features)    

    return org_u, model.ranking

########################################


# from joblib import Parallel, delayed
    
    
# def LocalKSparsifyParallel(G, X, K):
        
#     H = nx.Graph()
    
#     nodes=list(G.nodes())    
#     for u in nodes:
#         H.add_node(u)
    
#     params={}
    
#     print("Preparing params....")
    
#     for u in nodes:    
#         print(u, end=" ")
#         neighbors=list(G.neighbors(u))
#         neighbors.append(u)        

#         node2index={neighbors[i]:i for i in range(len(neighbors))}
#         index2node={i:neighbors[i] for i in range(len(neighbors))}

#         params[u]=(node2index[u], list(index2node.keys()), X[neighbors], K)
        
    
#     print("\nStarting sparsify....")
    
#     results = Parallel(n_jobs=NUM_PROCESSORS)(
#         delayed(GetLocalEdgesParallel)(key, value[0],value[1],value[2],value[3]) for key,value in tqdm(params.items())
#     )
  
#     #pprint.pprint(results)

#     results_dict = dict(results)
    
#     #pprint.pprint(results_dict)
    
#     print("Add edges....")
    
#     for u in nodes:            
#         print(u, end=" ")
#         neighbors=list(G.neighbors(u))
#         neighbors.append(u)        

#         node2index={neighbors[i]:i for i in range(len(neighbors))}
#         index2node={i:neighbors[i] for i in range(len(neighbors))}

#         ranks=results_dict[u]
        
#         node_ranks=[index2node[i] for i in ranks]

#         for v in node_ranks:
#             H.add_edge(u,v)
            
#     return H
    
# K=2
# H = LocalKSparsifyParallel(G, X, K)
# BGraph.draw_graph(H,pos)
# print(len(H.edges))


########################################
    
# dic={1:(1,2,3,4)}
# results = Parallel(n_jobs=2)(
#     delayed(GetLocalEdgesParallel)(key, value[0],value[1],value[2],value[3]) for key,value in tqdm(dic.items())
# )
# print(results)

########################################


# def process_frame(u):
#     print(u)
#     return u, u

# from tqdm import tqdm
# from multiprocess import Pool
# #from LocalSubSelect import process_frame

# frames_list = [1, 2, 3, 4, 5, 6]
# max_pool = 5

# with Pool(max_pool) as p:
#     pool_outputs = list(
#         tqdm(
#             p.imap(process_frame, frames_list),total=len(frames_list)
#         )
#     )    

# print(pool_outputs)
# new_dict = dict(pool_outputs)

# print("dict:", new_dict)


# print("#"*100)

########################


