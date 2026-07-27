#!/bin/bash

echo "Current date and time: $(date)"

# python main.py --dataset cora --embedding-dim 1433 512 7

#python main_pruning_imp.py --dataset cora --embedding-dim 1433 512 7 --lr 0.008 --weight-decay 8e-5 --pruning_percent_wei 0.2 --pruning_percent_adj 0.2 --total_epoch 200 --s1 1e-2 --s2 1e-2 --init_soft_mask_type all_one
# python main_gingat_imp.py --dataset cora --net gat --embedding-dim 1433 512 7 --lr 0.008 --weight-decay 8e-5 --pruning_percent_wei 0.2 --pruning_percent_adj 0.2 --mask_epoch 200 --fix_epoch 200 --s1 1e-3 --s2 1e-3


python main_pruning_imp.py --dataset SmallCora --embedding-dim 1433 512 7 --lr 0.008 --weight-decay 8e-5 --pruning_percent_wei 0.2 --pruning_percent_adj 0.2 --total_epoch 200 --s1 1e-2 --s2 1e-2 --init_soft_mask_type all_one
# python main_pruning_imp.py --dataset Photo --embedding-dim 745 512 8 --lr 0.008 --weight-decay 8e-5 --pruning_percent_wei 0.2 --pruning_percent_adj 0.2 --total_epoch 200 --s1 1e-2 --s2 1e-2 --init_soft_mask_type all_one
# python main_pruning_imp.py --dataset PubMed --embedding-dim 500 256 3 --lr 0.008 --weight-decay 8e-5 --pruning_percent_wei 0.2 --pruning_percent_adj 0.2 --total_epoch 200 --s1 1e-2 --s2 1e-2 --init_soft_mask_type all_one
# python main_pruning_imp.py --dataset CS --embedding-dim 6805 128 15 --lr 0.008 --weight-decay 8e-5 --pruning_percent_wei 0.2 --pruning_percent_adj 0.2 --total_epoch 200 --s1 1e-2 --s2 1e-2 --init_soft_mask_type all_one
# python main_pruning_imp.py --dataset Tolokers --embedding-dim 10 128 2 --lr 0.008 --weight-decay 8e-5 --pruning_percent_wei 0.2 --pruning_percent_adj 0.2 --total_epoch 200 --s1 1e-2 --s2 1e-2 --init_soft_mask_type all_one
# python main_pruning_imp.py --dataset amherst41 --embedding-dim 1193 256 3 --lr 0.008 --weight-decay 8e-5 --pruning_percent_wei 0.2 --pruning_percent_adj 0.2 --total_epoch 200 --s1 1e-2 --s2 1e-2 --init_soft_mask_type all_one
# python main_pruning_imp.py --dataset Squirrel --embedding-dim 2345 256 5 --lr 0.008 --weight-decay 8e-5 --pruning_percent_wei 0.2 --pruning_percent_adj 0.2 --total_epoch 200 --s1 1e-2 --s2 1e-2 --init_soft_mask_type all_one
# python main_pruning_imp.py --dataset cornell5 --embedding-dim 4735 256 3 --lr 0.008 --weight-decay 8e-5 --pruning_percent_wei 0.2 --pruning_percent_adj 0.2 --total_epoch 200 --s1 1e-2 --s2 1e-2 --init_soft_mask_type all_one
# python main_pruning_imp.py --dataset Computers --embedding-dim 767 256 10 --lr 0.008 --weight-decay 8e-5 --pruning_percent_wei 0.2 --pruning_percent_adj 0.2 --total_epoch 200 --s1 1e-2 --s2 1e-2 --init_soft_mask_type all_one
# python main_pruning_imp.py --dataset Cora --embedding-dim 8710 256 70 --lr 0.008 --weight-decay 8e-5 --pruning_percent_wei 0.2 --pruning_percent_adj 0.2 --total_epoch 200 --s1 1e-2 --s2 1e-2 --init_soft_mask_type all_one
# python main_pruning_imp.py --dataset Roman-empire --embedding-dim 300 256 18 --lr 0.008 --weight-decay 8e-5 --pruning_percent_wei 0.2 --pruning_percent_adj 0.2 --total_epoch 200 --s1 1e-2 --s2 1e-2 --init_soft_mask_type all_one
# python main_pruning_imp.py --dataset johnshopkins55 --embedding-dim 2406 256 3 --lr 0.008 --weight-decay 8e-5 --pruning_percent_wei 0.2 --pruning_percent_adj 0.2 --total_epoch 200 --s1 1e-2 --s2 1e-2 --init_soft_mask_type all_one
# python main_pruning_imp.py --dataset reed98 --embedding-dim 1001 256 3 --lr 0.008 --weight-decay 8e-5 --pruning_percent_wei 0.2 --pruning_percent_adj 0.2 --total_epoch 200 --s1 1e-2 --s2 1e-2 --init_soft_mask_type all_one
#python main_pruning_imp.py --dataset genius --embedding-dim 12 256 2 --lr 0.008 --weight-decay 8e-5 --pruning_percent_wei 0.2 --pruning_percent_adj 0.2 --total_epoch 200 --s1 1e-2 --s2 1e-2 --init_soft_mask_type all_one


