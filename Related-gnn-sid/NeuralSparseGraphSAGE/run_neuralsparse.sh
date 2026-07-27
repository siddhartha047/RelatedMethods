#!/bin/bash

echo ----------NeuralSprase---------------------

#for i in 1 2 3 4 5; do   
for i in 1; do
    echo ------run$i---------
    for dataset in "arxiv-year"; do    
        echo $dataset                
        python train.py --dataset $dataset --epochs 3
    done
done 