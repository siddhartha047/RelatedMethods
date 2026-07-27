#!/bin/bash

echo ---NeuralSparseResults------=

for dataset in "CiteSeer"; do
    
    echo ----$dataset-----
    for run in 1 2 3 4 5; do
        echo ----Run$run-----
        python NeuralSparse2.py --dataset $dataset --epochs 500 --k 3 --nolog False --nosparsify 
        # python NeuralSparse2.py --dataset $dataset --epochs 200 --k 10 --nolog  --nosparsify 
    done
done 

