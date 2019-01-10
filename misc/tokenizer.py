#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright © 2018 LeonTao
#
# Distributed under terms of the MIT license.

"""
Tokenizer
"""
import re
import jieba

class Tokenizer:
    def __init__(self, userdict_path=None):
        # load user dict
        if userdict_path is not None:
            jieba.load_userdict(userdict_path)

        self.MAX_NUMBER_COUNT = 12

        self.datetime_regex_str = r'\d{4}-?\d{2}-?\d{2}\s?\d{2}:\d{2}:\d{2}'
        self.datetime_re = re.compile(self.datetime_regex_str)

    def tokenize(self, text):
        if isinstance(text, list):
            text = ' '.join(text)

        tokens = self.clean_str(text)
        tokens = [token for token in tokens if len(token.split()) > 0]
        return tokens

    def clean_str(self, text):
        text = text.lower()

        text = text.replace(':', ' : ')
        text = text.replace(',', ' , ')

        text = text.replace('：', ' ： ')
        text = text.replace('，', ' ， ')
        text = text.replace('。', ' 。 ')

        text = self.datetime_re.sub('DATETIME', text)

        tokens = list(jieba.cut(text))
        tokens = [token for token in tokens if len(token.split()) > 0]
        if len(tokens) == 0:
            return []


        new_tokens = []
        number_count = 0
        for token in tokens:
            try:
                float(token)
                new_tokens.append('NUMBER')
                number_count += 1
            except ValueError as e:
                new_tokens.append(token)
                continue

        if number_count >= self.MAX_NUMBER_COUNT:
            return [] # merge multi to single

        tokens = new_tokens

        text = ' '.join(tokens)
        text = text.replace('( NUMBER )', 'NUMBER')
        text = re.sub(r'(\s?NUMBER\s?)+', ' NUMBER ', text)
        text = re.sub(r'(\s?NUMBER NUMBER\s?)+', ' NUMBER ', text)
        text = re.sub(r'NUMBER\w+', 'NUMBER', text)
        text = re.sub(r'\w+NUMBER', 'NUMBER', text)
        text = re.sub(r'NUMBER \S NUMBER', 'NUMBER', text)

        tokens = text.split()
        # print(tokens)

        return tokens
