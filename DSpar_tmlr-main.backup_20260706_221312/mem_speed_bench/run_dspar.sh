#!/bin/bash

echo "Current date and time: $(date)"

MODEL=gcn
SPARSIFY=spec_sparsify
DATASET=SmallCora

# python ./non_ogb_datasets/train_full_batch.py --conf ./non_ogb_datasets/conf/$MODEL.yaml --$SPARSIFY --dataset $DATASET
# python ./non_ogb_datasets/train_full_batch.py --conf ./non_ogb_datasets/conf/$MODEL.yaml --dataset $DATASET

# for DATASET in "reed98" "Squirrel" "cornell5"; do
for DATASET in "SmallCora" "Computers" "Photo" "PubMed" "CS" "Cora" "Tolokers" "Roman-empire" "johnshopkins55" "amherst41" "reed98" "Squirrel" "cornell5" "genius" "pokec" "arxiv-year" "Reddit"; do
    echo --------start-------$DATASET-----------------
    python ./non_ogb_datasets/train_full_batch.py --conf ./non_ogb_datasets/conf/$MODEL.yaml --$SPARSIFY --dataset $DATASET

    echo --------end---$DATASET----------

done

# "Cornell",
    # "Texas",
    # "Wisconsin",
    # "reed98",
    # "amherst41",
    # "penn94",
    # "Roman-empire",
    # "cornell5",
    # "Squirrel",
    # "johnshopkins55",
    # "Actor",
    # "Minesweeper",
    # "Questions",
    # "Chameleon",
    # "Tolokers",
    # "Amazon-ratings",
    # "genius",
    # "pokec",
    # "arxiv-year",
    # "snap-patents",
    # "ogbn-proteins",
    # "Cora",
    # "DBLP",
    # "Computers",
    # "PubMed",
    # "Cora_ML",
    # "SmallCora",
    # "CS",
    # "Photo",
    # "Physics",
    # "CiteSeer",
    # "wiki",
    # "Reddit"


