#!/bin/bash

for mode in "learned"; do
    echo $mode
    for dataset in "SmallCora" "Cora" "CiteSeer" "johnshopkins55" "Squirrel" "Roman-empire"; do
        echo ---------$dataset--------------
        for GNN in "GCN" "GIN" "Cheb" "GAT"; do
            echo $GNN
            #python main.py --GNN GCN --dataset SmallCora --mode learned --runs 1 --epochs 250 --save_csv True 
            python main.py --GNN $GNN --dataset $dataset --mode learned --runs 5 --epochs 200 --save_csv True 
        done
        echo ---------end--------------
    done 
done 