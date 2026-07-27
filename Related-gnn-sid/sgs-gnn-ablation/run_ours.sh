#!/bin/bash

for mode in "full" "random" "edge"  "learned"; do
# for mode in "learned"; do
    echo $mode
    # python main.py --dataset CiteSeer --mode $mode --runs 5 --save_csv True 
    # python main.py --dataset DBLP --mode $mode --runs 5 --epochs 500 --save_csv True
    # python main.py --dataset DBLP --mode $mode --runs 5 --epochs 500 --plot_curve True --degree_bias_coef 0.7 --consist_reg_coef 0 --num_samples_eval 10
    # python main.py --dataset Physics --mode $mode --runs 5 --save_csv True
    # python main.py --dataset CS --mode $mode --runs 5 --save_csv True
    #  python main.py --dataset Photo --mode $mode --runs 5 --save_csv True
    # python main.py --dataset SmallCora --mode $mode --runs 5 --save_csv True
    # python main.py --dataset PubMed --mode $mode --runs 5 --save_csv True
    # python main.py --dataset wiki --mode $mode --runs 5 --save_csv True
    #python main.py --dataset Cora_ML --mode $mode --runs 5 --save_csv True

    python main.py --dataset Cora --mode learned --runs 1 --epochs 100 --save_csv True
    
    python main.py --dataset pokec --mode $mode --runs 3 --epochs 50 --save_csv True
    python main.py --dataset arxiv-year --mode $mode --runs 3 --epochs 50 --save_csv True
    python main.py --dataset snap-patents --mode $mode --runs 3 --epochs 50 --save_csv True
    python main.py --dataset ogbn-proteins --mode $mode --runs 3 --epochs 50 --save_csv True
    python main.py --dataset Reddit --mode $mode --runs 3 --epochs 50 --save_csv True
    
    #python main.py --dataset Reddit --mode learned --runs 1 --epochs 1 --save_csv True
#     python main.py --dataset SmallCora --mode learned --runs 1 --epochs 200 --save_csv True --log True
    #python main.py --dataset SmallCora --mode $mode --runs 1 --epochs 1 --save_csv True --log True
    #python main.py --dataset ogbn-proteins --mode $mode --runs 5 --epochs 25 --save_csv True --log True
    #python main.py --dataset pokec --mode $mode --runs 1 --epochs 1 --save_csv True --log True
done 