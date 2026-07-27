#!/bin/bash

echo ---MOG------=

for dataset in "johnshopkins55"; do
    echo ----$dataset-----
    #python main.py --k_list 0.2 0.2 0.2 --expert_select 3 --dataset SmallCora --log_steps 10 --epochs 200 --run 5
    python main.py --k_list 0.2 0.2 0.2 --expert_select 3 --dataset $dataset --log_steps 1 --epochs 500 --run 5
done 

