#!/bin/bash

for mode in "learned"; do
    echo $mode
    #for dataset in "SmallCora" "Cora_ML" "Cora" "CiteSeer" "Cornell" "Texas" "Wisconsin"; do
    for dataset in "Cora"; do
        echo $dataset
        for percent in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9; do
            echo $percent
            python main.py --dataset $dataset --mode learned --runs 5 --epochs 200 --save_csv True --sample_perc $percent
            #python main.py --dataset Cornell --mode learned --runs 5 --epochs 200 --save_csv True --sample_perc 0.2
        done
    done
done 