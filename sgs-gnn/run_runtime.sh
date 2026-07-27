#!/bin/bash

echo -------Perepochruntime------

for mode in "learned"; do
    echo $mode
    #for dataset in "Cornell" "Texas" "Wisconsin" "reed98" "amherst41" "penn94" "Roman-empire" "cornell5" "Squirrel" "johnshopkins55" "Actor" "Minesweeper" "Questions" "Chameleon" "Tolokers" "Amazon-ratings" "Cora" "DBLP" "Computers" "PubMed" "Cora_ML" "SmallCora" "CS" "Photo" "Physics" "CiteSeer" "wiki"; do
    for dataset in "CS"	"Questions"	"Amazon-ratings"	"johnshopkins55"	"amherst41"; do               
        python main.py --dataset $dataset --mode $mode --runs 3 --epochs 200 --save_csv True --sample_perc 0.2 --GNN GCN --nhid 128
        #python main.py --dataset SmallCora --mode edge --runs 1 --epochs 200 --save_csv True --sample_perc 0.2 --GNN GCN
    done
done