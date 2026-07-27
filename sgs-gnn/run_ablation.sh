#!/bin/bash

echo -------Effectiveregistancesampler------

for mode in "edge"; do
    echo $mode
    #for dataset in "Cornell" "Texas" "Wisconsin" "reed98" "amherst41" "penn94" "Roman-empire" "cornell5" "Squirrel" "johnshopkins55" "Actor" "Minesweeper" "Questions" "Chameleon" "Tolokers" "Amazon-ratings" "Cora" "DBLP" "Computers" "PubMed" "Cora_ML" "SmallCora" "CS" "Photo" "Physics" "CiteSeer" "wiki"; do
    for dataset in "genius" "pokec" "arxiv-year" "snap-patents" "ogbn-proteins"; do
        echo --------ER---$dataset----------        
        python main.py --dataset $dataset --mode $mode --runs 5 --epochs 200 --save_csv True --sample_perc 0.2 --GNN GCN --ER True
        #python main.py --dataset SmallCora --mode edge --runs 1 --epochs 200 --save_csv True --sample_perc 0.2 --GNN GCN --ER True
    done
done

# for mode in "learned"; do
#     echo $mode
#     for dataset in "Cornell" "Texas" "Wisconsin" "reed98" "amherst41" "penn94" "Roman-empire" "cornell5" "Squirrel" "johnshopkins55" "Actor" "Minesweeper" "Questions" "Chameleon" "Tolokers" "Amazon-ratings" "Cora" "DBLP" "Computers" "PubMed" "Cora_ML" "SmallCora" "CS" "Photo" "Physics" "CiteSeer" "wiki"; do
#         echo --------GCN---$dataset----------
#         # Uncomment the following line to run the script with the specified parameters
#         python main.py --dataset $dataset --mode $mode --runs 5 --epochs 200 --save_csv True --sample_perc 0.2 --GNN GCN
#     done
# done


# for mode in "learned"; do
#     echo $mode
#     for dataset in "SmallCora"; do
#         echo $dataset                
#         #python main.py --dataset $dataset --mode learned --runs 5 --epochs 250 --save_csv True --sample_perc 0.2 --GNN GAT       
#         # python main.py --dataset SmallCora --mode learned --runs 5 --epochs 250 --save_csv True --sample_perc 0.2        
#     done
# done 