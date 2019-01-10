#!/usr/bin/env bash

python misc/preprocess.py \
    --raw_question_path ./data/raw.question.txt \
    --cleaned_question_path ./data/cleaned.question.txt \
    --vocab_freq_path ./data/vocab.freq.txt \
    --save_dir ./data \
    --min_len 5 \
    --q_max_len 120 \
    --r_max_len 220 \

    /
