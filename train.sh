#!/usr/bin/env bash
#
# train.sh
# Copyright (C) 2018 LeonTao
#
# Distributed under terms of the MIT license.
#
export CUDA_VISIBLE_DEVICES=0

mkdir -p data/
mkdir -p data/generated
mkdir -p log/
mkdir -p models/

python train.py \
    --data_path data/cleaned.question.txt \
    --vocab_path data/vocab.word2idx.dict \
    --data_dir data/ \
    --log log/ \
    --embedding_size 128 \
    --hidden_size 256 \
    --enc_num_layers 1 \
    --dec_num_layers 1 \
    --bidirectional \
    --share_embedding \
    --dropout 0.0 \
    --lr 0.001 \
    --max_grad_norm 5.0 \
    --q_max_len 60 \
    --r_max_len 55 \
    --batch_size 64 \
    --valid_split 0.08 \
    --test_split 5 \
    --epochs 20 \
    --start_epoch 1 \
    --lr_patience 3 \
    --es_patience 5 \
    --device cuda \
    --seed 23 \
    --save_mode all \
    --save_model models/ \
    --smoothing \
    --mode train \
    --checkpoint ./models/accu_36.093.pth \


/
