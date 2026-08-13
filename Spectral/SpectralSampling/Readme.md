Task:

row1-row2
1. Take a heterophilic graph, e.g. Cornell, Texas, Wisconsin, Cora
2. Use cosine similarity to compute edge weights, edge_weights
3. Run GraphSAGE with random sampling of k=[5,5], neighbors, hop = 2, to get results
4. Replace random sampling with weighted sampling (feature based) of same k and hop.   Function: Random.choice()

row3:

1. Use effective effective resistance to compute edge_weights, Pe
2. Try with exact and approximate (code in GraphSparsiferMain.ipynb box 6,7)
3. Run GraphSAGE with weighted sampling of same k and hop using selection probabilities of Pe. 


row4
1. First compute edge_weights from cosine similarity of earlier step 2
2. Modify effective resistance code to use weighted adjacency matrix or (weighted random walk for local) 
3. Run GraphSAGE with sampling of same k and hop using selection probabilities of Pe


row5: (optional)
1. get weights from cosine similarity
2. get weights from effective resistance
3. multply these weights
4. sample weights based on these and apply on graphsage.



Table results:

Row 1: Random neighbor sampling - GraphSAGE
Row 2: Feature based weighted sampling - e.g. Cosine similarity based weighted sample
Row 3: Effective resistance based weighted sampling
Row 4: Compute undweighted graph to weighted graph using cosine similarity
       Compute effective resistance weights on the weighted graph
       Sample neighbors based on these effective resistance weights
       
       
       
Starting:
For small graph, use true effective resistance weights. Eg. matrix based.
For large graph use local effective resistance computation weights, Eg. random walk based.

Links:


Papers:
https://arxiv.org/pdf/2106.03476.pdf

https://openreview.net/forum?id=sjGBjudWib

http://proceedings.mlr.press/v119/zheng20d/zheng20d.pdf