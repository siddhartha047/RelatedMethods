import math
import numpy as np
import networkx as nx
from random import choice
import sys

def random_walk(G, s, l):
    v = s;
    for i in range(l):  
        if (len(G[v]) == 0):
            continue;
        v = choice(list(G.neighbors(v)))

    return v


def effective_resistance_akp_parallel(iteration_no, G, s, t, eps=0.1, lmbda=0.1):
    
#     print(hex(id(G)))
#     sys.stdout.flush()
    
    l = math.ceil(math.log(4 / (eps * (1 - lmbda))) / math.log(1.0 / lmbda) / 2)
    r = int(math.ceil(40 * l * l * math.log(80 * l) / (eps * eps)))
    delta = 0

    for i in range(l):
        Xis = 0; Xit = 0; Yis = 0; Yit = 0;
        
        for j in range(r):
            v = random_walk(G, s, i)
            if (v == s):
                Xis+=1
            if (v == t):
                Xit+=1    
        
        for j in range(r):
            v = random_walk(G, t, i);
            if (v == s):
                Yis+=1;
            if (v == t):
                Yit+=1;
                
        deltai = float(Xis) / G.degree[s] - float(Xit) / G.degree[t] - float(Yis) / G.degree[s] + float(Yit) / G.degree[t]
        deltai /= r;
        delta += deltai;

    return iteration_no, max(0,delta)


if __name__ == '__main__':    
    G = nx.random_regular_graph(2, 10)
    print(hex(id(G)))
    print(effective_resistance_akp_parallel(0, G, 0, 1, eps=0.1, lmbda=0.1))