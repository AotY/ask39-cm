#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright © 2018 LeonTao
#
# Distributed under terms of the MIT license.

"""
Utils
"""
import os
import sys



def save_distribution(dist_list, save_path):
    with open(os.path.join(save_path), 'w', encoding="utf-8") as f:
        for i, j in dist_list:
            f.write('%s\t%s\n' % (str(i), str(j)))


def generate_texts(vocab, batch_size, outputs, outputs_length=None, decode_type='greedy'):
    """ decode_type == greedy:
        outputs: [batch_size, max_len]
        return: [batch_size]
    decode_type == 'beam_search':
        outputs: [batch_size, topk, max_len]
        outputs_length: [batch_size, topk]
        return: [batch_size, topk]
    """

    batch_texts = []
    if decode_type == 'greedy':
        for bi in range(batch_size):
            text = vocab.ids_to_text(outputs[bi].tolist())
            batch_texts.append(text)
    elif decode_type == 'beam_search':
        for bi in range(batch_size):
            topk_ids = outputs[bi]
            topk_texts = []
            if outputs_length is not None:
                topk_length = outputs_length[bi]
                for ids, length in zip(topk_ids, topk_length):
                    text = vocab.ids_to_text(ids[:length])
                    topk_texts.append(text)
            else:
                for ids in topk_ids:
                    text = vocab.ids_to_text(ids)
                    topk_texts.append(text)

            batch_texts.append(topk_texts)

    return batch_texts


def save_generated_texts(epoch,
                         enc_texts,
                         greedy_texts,
                         beam_texts,
                         save_path):
    """
    enc_texts: [batch_size, max_len]
    greedy_texts: [batch_size, max_len]
    beam_texts: [batch_size, beam_size, max_len]
    """
    save_mode = 'w'
    with open(save_path, save_mode, encoding='utf-8') as f:
        for enc_text, g_text, b_text in zip(enc_texts, greedy_texts, beam_texts):

            # save query
            f.write('query: %s\n' % enc_text)

            f.write('greedy --> %s\n' % g_text)

            for i, text in enumerate(b_text):
                f.write('beam %d--> %s\n' % (i, text))

            f.write('\n')



