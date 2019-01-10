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
                q_id, d_id, q, r, sub, gender, age, onset, label = line.split(
                    ' SPLIT ')

                q_tokens = q.split()
                r_tokens = r.split()

                q_ids = vocab.words_to_id(q_tokens)
                r_ids = vocab.words_to_id(r_tokens)

                datas.append((q_ids, r_ids))
        pickle.dump(open(datas_pkl_path, 'wb'))
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

            enc_inputs.append(q_ids)
            dec_inputs.append(r_ids)

            enc_lengths.append(len(q_ids))
            dec_lengths.append(len(r_ids) + 1)

        enc_inputs = np.array([
            ids + [PAD_ID] * (q_max_len - len(ids))
            for ids in enc_inputs
        ])
        # to [max_len, batch_size]
        enc_inputs = enc_inputs.transpose()

        dec_inputs = np.array([
            [SOS_ID] + ids + [EOS_ID] + [PAD_ID] * (r_max_len - len(ids))
            for ids in dec_inputs
        ])
        # to [max_len, batch_size]
        dec_inputs = dec_inputs.transpose()

        enc_inputs = torch.LongTensor(enc_inputs)
        dec_inputs = torch.LongTensor(dec_inputs)

        enc_lengths = torch.LongTensor(enc_lengths)
        dec_lengths = torch.LongTensor(dec_lengths)

        return enc_inputs, dec_inputs, enc_lengths, dec_lengths
