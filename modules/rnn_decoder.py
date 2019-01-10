#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright © 2018 LeonTao
#
# Distributed under terms of the MIT license.

"""
RNN Decoder
"""

import torch
import torch.nn as nn

from modules.utils import init_linear_wt
from modules.utils import init_gru_orth
from modules.attention import Attention


class RNNDecoder(nn.Module):
    def __init__(self, config, embedding):

        super(RNNDecoder, self).__init__()

        # embedding
        self.embedding = embedding
        self.embedding_size = embedding.embedding_dim

        # dropout
        self.dropout = nn.Dropout(config.dropout)

        self.rnn = nn.GRU(
            input_size=self.embedding_size,
            hidden_size=config.hidden_size,
            num_layers=config.decoder_num_layers,
            dropout=config.dropout
        )

        init_gru_orth(self.rnn)

        self.enc_attn = Attention(config.hidden_size)

        self.linear = nn.Linear(config.hidden_size, config.vocab_size)
        init_linear_wt(self.linear)

        if config.tied:
            self.linear.weight = self.embedding.weight

    def forward(self,
                dec_input,
                dec_hidden,
                enc_outputs=None,
                enc_length=None):
        '''
        Args:
            dec_input: [max_len, batch_size, hidden_size] or [1, batch_size, hidden_size]
            dec_hidden: [num_layers, batch_size, hidden_size]

            enc_outputs: [max_len, batch_size, hidden_size]
            enc_length: [batch_size]
        '''
        # embedded
        embedded = self.embedding(dec_input)  # [1, batch_size, embedding_size]
        embedded = self.dropout(embedded)

        output, dec_hidden = self.rnn(embedded, dec_hidden)

        if enc_outputs is not None:
            # [1, batch_size, hidden_size]
            output, _ = self.enc_attn(output, enc_outputs, enc_length)

        # [, batch_size, vocab_size]
        output = self.linear(output)

        return output, dec_hidden, None
