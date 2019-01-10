#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright © 2018 LeonTao
#
# Distributed under terms of the MIT license.

"""
Build userdict. (add word freq)
"""

import argparse
from tqdm import tqdm
import random

parser = argparse.ArgumentParser()

parser.add_argument('--dict_path', type=str, default='')
parser.add_argument('--save_path', type=str, default='')
parser.add_argument('--max_freq', type=int, default=100)
parser.add_argument('--min_freq', type=int, default=3)
args = parser.parse_args()

save_file = open(args.save_path, 'w')
words_set = set()
with open(args.dict_path, 'r') as f:
    for line in tqdm(f):
        line = line.rstrip()
        word = line.split()[0]
        if word not in words_set:
            freq = random.randint(args.min_freq, args.max_freq)
            save_file.write('%s %d\n' % (word, freq))
            words_set.add(word)

save_file.close()




