#!/bin/bash

echo ---------------Convergence----------

for mode in "edge"; do
    echo ------$mode-----------
    #for dataset in "Cornell" "Texas" "Wisconsin" "reed98" "amherst41" "penn94" "Roman-empire" "cornell5" "Squirrel" "johnshopkins55" "Actor" "Minesweeper" "Questions" "Chameleon" "Tolokers" "Amazon-ratings" "Cora" "DBLP" "Computers" "PubMed" "Cora_ML" "SmallCora" "CS" "Photo" "Physics" "CiteSeer" "wiki"; do
    for dataset in "SmallCora" "Cora" "CiteSeer" "johnshopkins55" "Squirrel" "Roman-empire"; do
        echo ---------$dataset----------        
        python main.py --dataset $dataset --mode $mode --runs 5 --epochs 500 --save_csv True --sample_perc 0.2 --convergence 0.001 --ER True
        #python main.py --dataset SmallCora --mode learned --runs 1 --epochs 500 --save_csv True --sample_perc 0.2 --convergence 0.001 --log True
        #python main.py --dataset SmallCora --mode random --runs 1 --epochs 500 --save_csv True --sample_perc 0.2 --convergence 0.001 --log True
    done
done