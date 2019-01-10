#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright © 2018 LeonTao
#
# Distributed under terms of the MIT license.

"""
preprocessing
"""

import os
import re
import sys
sys.path.append('.')
import argparse
import random

from tqdm import tqdm
from collections import Counter

from misc.tokenizer import Tokenizer
from misc.utils import save_distribution

parser = argparse.ArgumentParser()

parser.add_argument('--raw_question_path', type=str, default='')
parser.add_argument('--cleaned_question_path', type=str, default='')
parser.add_argument('--vocab_freq_path', type=str, default='')
parser.add_argument('--save_dir', type=str, default='./data')

parser.add_argument('--min_len', type=int, default=5)
parser.add_argument('--q_max_len', type=int, default=150)
parser.add_argument('--r_max_len', type=int, default=200)
args = parser.parse_args()

tokenizer = Tokenizer()

def cleaning_stats():
    q_ids_set = set()

    q_len_dict = {}
    r_len_dict = {}
    freq_dict = Counter()

    multi_turn_count = 0
    cleaned_question_file = open(args.cleaned_question_path, 'w', encoding='utf-8')
    cleaned_questions = list()
    with open(args.raw_question_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            line = line.rstrip()
            try:
                q_id, d_id, title, query, response, \
                    sub, gender, age, onset, labels = line.split('SPLIT')
            except ValueError as e:
                # print(line)
                # print(e)
                continue

            if not bool(query) or not bool(response):
                continue

            # multi-turn
            if response.count('EOS') > 0:
                multi_turn_count += 1
                continue

            if q_id in q_ids_set:
                continue
            else:
                q_ids_set.add(q_id)

            #  print('q_id: %s' % q_id)

            # query
            genders = re.findall(r'患者性别：\w', query)
            if genders is not None and len(genders) > 0:
                gender = genders[0].split('：')[1]
                #  print('gender: %s' % gender)
                query = re.sub(r'患者性别：\w', '', query)

            ages = re.findall(r'患者年龄：\d+', query)
            if ages is not None and len(ages) > 0:
                age = ages[0].split('：')[1]
                #  print('age: %s' % age)
                query = re.sub(r'患者年龄：\d+', '', query)

            freq_dict.update(genders)
            freq_dict.update(age)

            query = re.sub(r'问题描述：', '', query)
            query = re.sub(r'详细病情及咨询目的：', '', query)
            query = re.sub(r'本次发病及持续的时间：', ' 。', query)
            query = re.sub(r'病史：', ' 。', query)
            query = re.sub(r'目前一般情况：', ' 。', query)
            query = re.sub(r'细病情及咨询目的：', ' 。', query)
            query = re.sub(r'以往的诊断和治疗经过及效果：', ' 。', query)

            q_tokens = tokenizer.tokenize(query)
            freq_dict.update(q_tokens)
            q_len_dict[len(q_tokens)] = q_len_dict.get(len(q_tokens), 0) + 1
            #  print('q_tokens: {}'.format(q_tokens))
            if len(q_tokens) < args.min_len or len(q_tokens) > args.q_max_len:
                continue

            q = ' '.join(q_tokens)

            # query
            r_tokens = tokenizer.tokenize(response)
            freq_dict.update(r_tokens)
            r_len_dict[len(r_tokens)] = r_len_dict.get(len(r_tokens), 0) + 1
            #  print('r_tokens: {}'.format(r_tokens))
            if len(r_tokens) < args.min_len or len(r_tokens) > args.r_max_len:
                continue

            r = ' '.join(r_tokens)

            # sub
            sub = ' '.join(sub.split('_'))
            sub_tokens = tokenizer.tokenize(sub)
            sub_tokens = list(set(sub_tokens))
            freq_dict.update(sub_tokens)
            sub = ' '.join(sub_tokens)

            # labels
            label = ' '.join(labels.split(','))
            label_tokens = tokenizer.tokenize(label)
            label_tokens = list(set(label_tokens))
            freq_dict.update(label_tokens)
            label = ' '.join(label_tokens)

            # onset
            onset = re.sub(r'发病时间：不清楚', '', onset)
            if onset != '':
                onset_tokens = tokenizer.tokenize(onset)
                freq_dict.update(onset_tokens)
                onset = ' '.join(onset_tokens)

            #  print('q_id: %s' % q_id)
            cleaned_questions.append((q_id, d_id, q, r, sub, gender, age, onset, label))
            
    # shuffle
    random.shuffle(cleaned_questions)

    # re write
    for item in cleaned_questions:
        q_id, d_id, q, r, sub, gender, age, onset, label = item

        cleaned_question_file.write('%s SPLIT %s SPLIT %s SPLIT %s SPLIT %s SPLIT %s SPLIT %s SPLIT %s SPLIT %s\n' % \
                                    (q_id, d_id, q, r, sub, gender, age, onset, label))

    cleaned_question_file.close()
    print('multi_turn_count: %d' % multi_turn_count)

    return freq_dict, q_len_dict, r_len_dict


def main():
    freq_dict, q_len_dict, r_len_dict = cleaning_stats()
    freq_list = sorted(freq_dict.items(), key=lambda item: item[1], reverse=True)
    save_distribution(freq_list, args.vocab_freq_path)

    q_len_list = sorted(q_len_dict.items(), key=lambda item: item[0], reverse=False)
    save_distribution(q_len_list, os.path.join(args.save_dir, 'q.len.dist'))


    r_len_list = sorted(r_len_dict.items(), key=lambda item: item[0], reverse=False)
    save_distribution(r_len_list, os.path.join(args.save_dir, 'r.len.dist'))

if __name__ == '__main__':
    main()
