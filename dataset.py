#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright © 2018 LeonTao
#
# Distributed under terms of the MIT license.

import os
import pickle
import torch
import torch.utils.data as data
import numpy as np
from tqdm import tqdm

from vocab import PAD_ID, SOS_ID, EOS_ID


def load_data(config, vocab):
    print('load data...')

    datas_pkl_path = os.path.join(config.data_dir, 'datas.pkl')
    if not os.path.exists(datas_pkl_path):
        datas = list()
        with open(config.data_path, 'r') as f:
            for line in tqdm(f):
                line = line.rstrip()
                q_id, d_id, q, r, sub, \
                    gender, age, onset, label = line.split('SPLIT')

                q_tokens = q.split()
                q_tokens = [token for token in q_tokens if len(token.split()) > 0]
                
                r_tokens = r.split()
                r_tokens = [token for token in r_tokens if len(token.split()) > 0]

                q_ids = vocab.words_to_id(q_tokens)
                r_ids = vocab.words_to_id(r_tokens)

                datas.append((q_ids, r_ids))
        pickle.dump(datas, open(datas_pkl_path, 'wb'))
    else:
        datas = pickle.load(open(datas_pkl_path, 'rb'))

    return datas


def build_dataloader(config, datas):
    valid_split = int(config.val_split * len(datas))
    test_split = int(config.batch_size * config.test_split)

    valid_dataset = Dataset(datas[:valid_split])
    test_dataset = Dataset(datas[valid_split: valid_split + test_split])
    train_dataset = Dataset(datas[valid_split + test_split:])

    collate_fn = MyCollate(config)

    # data loader
    train_data = data.DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn
    )

    valid_data = data.DataLoader(
        valid_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn
    )

    test_data = data.DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn
    )

    return train_data, valid_data, test_data


class Dataset(data.Dataset):
    def __init__(self, datas):
        self._datas = datas

    def __len__(self):
        return len(self._datas)

    def __getitem__(self, idx):
        q_ids, r_ids = self._datas[idx]
        return q_ids, r_ids


class MyCollate:
    def __init__(self, config):
        self.config = config

    def __call__(self, batch_pair):
        q_max_len, r_max_len = self.config.q_max_len, self.config.r_max_len

        ''' Pad the instance to the max seq length in batch '''
        # sort by q length
        batch_pair.sort(key=lambda x: len(x[0]), reverse=True)

        enc_inputs, dec_inputs = list(), list()
        enc_lengths, dec_lengths = list(), list()

        for q_ids, r_ids in batch_pair:

            q_ids = q_ids[-min(q_max_len, len(q_ids)):]
            r_ids = r_ids[:min(r_max_len, len(r_ids))]

            enc_lengths.append(len(q_ids))
            dec_lengths.append(len(r_ids) + 1)

            # pad
            q_ids = q_ids + [PAD_ID] * (q_max_len - len(q_ids))
            r_ids = [SOS_ID] + r_ids + [EOS_ID] + [PAD_ID] * (r_max_len - len(r_ids))

            enc_inputs.append(q_ids)
            dec_inputs.append(r_ids)

        enc_inputs = torch.tensor(enc_inputs, dtype=torch.long)
        # to [max_len, batch_size]
        enc_inputs = enc_inputs.transpose(0, 1)
        # print(enc_inputs)

        dec_inputs = torch.tensor(dec_inputs, dtype=torch.long)
        # to [max_len, batch_size]
        dec_inputs = dec_inputs.transpose(0, 1)

        enc_lengths = torch.tensor(enc_lengths, dtype=torch.long)
        dec_lengths = torch.tensor(dec_lengths, dtype=torch.long)

        return enc_inputs, dec_inputs, enc_lengths, dec_lengths
