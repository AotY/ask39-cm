#!/usr/bin/env bash
#
# build_vocab.sh
# Copyright (C) 2019 LeonTao
#
# Distributed under terms of the MIT license.
#



python misc/build_vocab.py \
    --vocab_freq_path ./data/vocab.freq.txt \
    --vocab_path ./data/vocab.id2word.dict \
    --vocab_size 6e4 \
    --min_count 3 \

/
