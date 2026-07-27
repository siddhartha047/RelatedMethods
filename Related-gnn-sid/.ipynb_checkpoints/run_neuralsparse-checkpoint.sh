#!/bin/bash

echo ---NeuralSparseResults------=

# for dataset in "Cornell"; do
    
#     echo ----$dataset-----
#     for run in 1 2 3 4 5; do
#         echo ----Run$run-----
#         python NeuralSparse2.py --dataset $dataset --epochs 200 --k 10 --nolog
#         # python NeuralSparse2.py --dataset $dataset --epochs 200 --k 10 --nolog --nosparsify
#     done
# done 


for dataset in "CS"	"Questions"	"Amazon-ratings" "johnshopkins55" "amherst41" "Tolokers" "Physics"; do                  
    echo ----$dataset-----    
    python DropEdge.py --dataset $dataset --epochs 10 --log False
done 




