import numpy as np
import random
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

def GetLocalEdgesRandom(org_u, u, nodes, features, K):
    
    nodes.remove(u)
  
    if len(nodes)-1<=K:        
        return org_u, np.array(nodes)

    return org_u, random.sample(nodes, K) 


def GetLocalEdgesRandomWeighted(org_u, u, nodes, features, K, weights):
    
    #print(u, nodes)
    
    if len(nodes)-1<=K:
        nodes.remove(u)
        return org_u, np.array(nodes)
    
    
    
    idx = np.argpartition(weights, -K)[-K:]
    indexs = idx[np.argsort(weights[idx])][::-1]

    return org_u, indexs 


def kNNedgesParallel(u, neighbors, X_u, X_neighbors, K):
    
    if len(neighbors)<=K:
        return u, neighbors

    target_class_sim = euclidean_distances([X_u], X_neighbors)
    #print(target_class_sim[0])    
    ind = np.argpartition(target_class_sim[0], K)[:K]    
    #print(ind) 
    
#     target_class_sim = cosine_similarity([X_u], X_neighbors)
#     #print(target_class_sim[0])    
#     ind = np.argpartition(target_class_sim[0], -K)[-K:]    
#     #print(ind)
    
    return u, ind

if __name__ == '__main__':    
    print(GetLocalEdgesRandom(1, 1, list(range(10)), np.random.rand(10,100), 100))
