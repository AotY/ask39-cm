#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright © 2018 LeonTao
#
# Distributed under terms of the MIT license.

import sys
sys.path.append('.')
import argparse
from tqdm import tqdm
from vocab import Vocab

parser = argparse.ArgumentParser()

parser.add_argument('--vocab_freq_path', type=str, help='')
parser.add_argument('--vocab_size', type=float, default=6e4)
parser.add_argument('--min_count', type=int, default=3)
parser.add_argument('--vocab_path', type=str, default='')

args = parser.parse_args()

args.vocab_size = int(args.vocab_size)
print('vocab_size: ', args.vocab_size)


def read_distribution():
    freq_list = []
    with open(args.vocab_freq_path, 'r') as f:
        for line in tqdm(f):
            line = line.rstrip()
            try:
                word, freq = line.split()
                freq = int(freq)
                freq_list.append((word, freq))
            except Exception as e:
                print(line)
                print(e)

    return freq_list


def build_vocab():
    freq_list = read_distribution()
    print('freq_list: ', len(freq_list))

    if args.min_count > 0:
        print('Clip tokens by min_count')
        freq_list = [item for item in freq_list if item[1] > args.min_count]

    if args.vocab_size > 0 and len(freq_list) >= args.vocab_size:
        print('Clip tokens by vocab_size')
        freq_list = freq_list[:args.vocab_size]

    vocab = Vocab()
    vocab.build_from_freq(freq_list)
    vocab.save(args.vocab_path)
    print('vocab_size: ', vocab.size)
    return vocab


if __name__ == '__main__':
    build_vocab()

