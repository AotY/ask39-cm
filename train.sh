#!/usr/bin/env bash
#
# train.sh
# Copyright (C) 2018 LeonTao
#
# Distributed under terms of the MIT license.
#
export CUDA_VISIBLE_DEVICES=0

mkdir -p data/
mkdir -p log/
mkdir -p modles/

python train.py \
    --data_path data/cleaned.question.txt \
    --data_dir data/ \
    --log log/ \
    --embedding_size 128 \
    --hidden_size 512 \
    --num_layers 1 \
    --bidirectional \
    --share_embedding \
    --dropout 0.0 \
    --clip 5.0 \
    --lr 0.001 \
    --batch_size 128 \
    --epochs 20 \
    --lr_patience 3 \
    --es_patience 5 \
    --device cuda \
    --seed 23 \
    --save_mode all \
    --save_model models/ \
    --smoothing \

/
