#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright © 2018 LeonTao
#

import os
import time
import math
import argparse

import torch
import torch.nn.functional as F

from tqdm import tqdm

from modules.optim import ScheduledOptimizer
from modules.early_stopping import EarlyStopping

from vocab import Vocab
from vocab import PAD_ID
from cm_model import CMModel
from dataset import load_data, build_dataloader
from utils import generate_texts, save_generated_texts

# Parse argument for language to train
parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, help='')
parser.add_argument('--vocab_path', type=str, help='')
parser.add_argument('--vocab_size', type=int, help='')
parser.add_argument('--embedding_size', type=int)
parser.add_argument('--hidden_size', type=int)
parser.add_argument('--bidirectional', action='store_true')
parser.add_argument('--num_layers', type=int)
parser.add_argument('--dropout', type=float)
parser.add_argument('--teacher_forcing_ratio', type=float, default=0.5)
parser.add_argument('--share_embedding', action='store_true')
parser.add_argument('--tied', action='store_true')
parser.add_argument('--clip', type=float, default=5.0)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--q_max_len', type=int, default=60)
parser.add_argument('--r_max_len', type=int, default=55)
parser.add_argument('--batch_size', type=int, help='')
parser.add_argument('--val_split', type=float, default=0.08)
parser.add_argument('--test_split', type=int, default=5)
parser.add_argument('--epochs', type=int)
parser.add_argument('--lr_patience', type=int, help='Number of epochs with no improvement after which learning rate will be reduced')
parser.add_argument('--es_patience', type=int, help='early stopping patience.')
parser.add_argument('--device', type=str, help='cpu or cuda')
parser.add_argument('--save_model', type=str, help='save path')
parser.add_argument('--save_mode', type=str, choices=['all', 'best'], default='best')
parser.add_argument('--smoothing', action='store_true')
parser.add_argument('--log', type=str, help='save log.')
parser.add_argument('--seed', type=str, help='random seed')
args = parser.parse_args()

torch.random.manual_seed(args.seed)
device = torch.device(args.device)

# load vocab
vocab = Vocab()
vocab.load(args.vocab_path)
args.vocab_size = int(vocab.size)

# load data
datas = load_data(args, vocab)

# dataset, data_load
train_data, valid_data, test_data = build_dataloader(args, datas)

# model
model = CMModel(
    args,
    device
).to(device)

print(model)

# optimizer
#  optimizer = optim.Adam(model.parameters(), lr=args.lr)
optim = torch.optim.Adam(
    model.parameters(),
    args.lr,
    betas=(0.9, 0.98),
    eps=1e-09
)

#  scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=2, gamma=0.5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optim,
    mode='min',
    factor=0.1,
    patience=args.lr_patience
)

optimizer = ScheduledOptimizer(
    optim,
    scheduler,
    args.max_grad_norm
)

# early stopping
early_stopping = EarlyStopping(
    type='min',
    min_delta=0.001,
    patience=args.es_patience
)

# train epochs
def train_epochs():
    ''' Start training '''
    log_train_file = None
    log_valid_file = None

    if args.log:
        log_train_file = os.path.join(args.log, 'train.log')
        log_valid_file = os.path.join(args.log, 'valid.log')

        print('[Info] Training performance will be written to file: {} and {}'.format(
            log_train_file, log_valid_file))

        with open(log_train_file, 'w') as log_tf, \
            open(log_valid_file, 'w') as log_vf:
            log_tf.write('epoch,loss,ppl,accuracy\n')
            log_vf.write('epoch,loss,ppl,accuracy\n')

    valid_accus = []
    for epoch in range(args.epochs):
        print('[ Epoch', epoch, ']')

        start = time.time()
        train_loss, train_accu = train(epoch)

        print(' (Training)   ppl: {ppl: 8.5f}, accuracy: {accu:3.3f} %, '
              'elapse: {elapse:3.3f} min'.format(
                  ppl=math.exp(min(train_loss, 100)), accu=100*train_accu,
                  elapse=(time.time()-start)/60))

        start = time.time()
        valid_loss, valid_accu = eval(epoch)
        print('  - (Validation) ppl: {ppl: 8.5f}, accuracy: {accu:3.3f} %, '
              'elapse: {elapse:3.3f} min'.format(
                  ppl=math.exp(min(valid_loss, 100)), accu=100*valid_accu,
                  elapse=(time.time()-start)/60))

        valid_accus += [valid_accu]

        model_state_dict = model.state_dict()
        checkpoint = {
            'model': model_state_dict(),
            'settings': args,
            'epoch': epoch,
            'optimizer': optimizer.optimizer.state_dict(),
            'valid_loss': valid_loss,
            'valid_accu': valid_accu
        }

        if args.save_model:
            if args.save_mode == 'all':
                model_name = os.path.join(args.save_model, 'accu_{accu:3.3f}.pth'.format(accu=100*valid_accu))
                torch.save(checkpoint, model_name)
            elif args.save_mode == 'best':
                model_name = os.path.join(args.save_model, 'best.pth')
                if valid_accu >= max(valid_accus):
                    torch.save(checkpoint, model_name)
                    print('   - [Info] The checkpoint file has been updated.')

        if log_train_file and log_valid_file:
            with open(log_train_file, 'a') as log_tf, open(log_valid_file, 'a') as log_vf:
                log_tf.write('{epoch},{loss: 8.5f},{ppl: 8.5f},{accu:3.3f}\n'.format(
                    epoch=epoch, loss=train_loss,
                    ppl=math.exp(min(train_loss, 100)), accu=100*train_accu))
                log_vf.write('{epoch},{loss: 8.5f},{ppl: 8.5f},{accu:3.3f}\n'.format(
                    epoch=epoch, loss=valid_loss,
                    ppl=math.exp(min(valid_loss, 100)), accu=100*valid_accu))

# train

def train(epoch):
    ''' Epoch operation in training phase'''
    model.train()

    total_loss = 0
    n_word_total = 0
    n_word_correct = 0

    for batch in tqdm(
            train_data, mininterval=2,
            desc=' (Training: %d) ' % epoch, leave=False):

        # prepare data
        enc_inputs, dec_inputs, enc_lengths, dec_lengths = map(
            lambda x: x.to(device), batch)
        # [batch_size, max_len]

        dec_targets = dec_inputs[:, 1:]
        dec_inputs = dec_inputs[:, :-1]

        print('enc_inputs: ', enc_inputs.shape)
        print('dec_inputs: ', dec_inputs.shape)
        print('dec_targets: ', dec_targets.shape)

        # forward
        optimizer.zero_grad()

        dec_outputs = model(
            enc_inputs,
            enc_lengths,
            dec_inputs,
            dec_lengths
        )

        # backward
        loss, n_correct = cal_performance(
            dec_outputs,
            dec_targets,
            smoothing=args.smoothing
        )

        loss.backward()

        # update parameters
        optimizer.step()

        # note keeping
        total_loss += loss.item()

        non_pad_mask = dec_targets.ne(PAD_ID)

        n_word = non_pad_mask.sum().item()

        n_word_total += n_word
        n_word_correct += n_correct

    loss_per_word = total_loss/n_word_total
    accuracy = n_word_correct/n_word_total
    return loss_per_word, accuracy


def eval(epoch):
    ''' Epoch operation in evaluation phase '''
    model.eval()

    total_loss = 0
    n_word_total = 0
    n_word_correct = 0

    with torch.no_grad():
        for batch in tqdm(
                valid_data, mininterval=2,
                desc=' (Validation: %d) ' % epoch, leave=False):

            enc_inputs, dec_inputs, enc_lengths, dec_lengths = map(
                lambda x: x.to(device), batch)

            dec_targets = dec_inputs[:, 1:]
            dec_inputs = dec_inputs[:, :-1]

            dec_outputs = model(
                enc_inputs,
                enc_lengths,
                dec_inputs,
                dec_lengths
            )

            # backward
            loss, n_correct = cal_performance(
                dec_outputs,
                dec_targets,
                smoothing=False
            )

            # note keeping
            total_loss += loss.item()

            non_pad_mask = dec_targets.ne(PAD_ID)
            n_word = non_pad_mask.sum().item()
            n_word_total += n_word
            n_word_correct += n_correct

    loss_per_word = total_loss/n_word_total
    accuracy = n_word_correct/n_word_total
    return loss_per_word, accuracy

def test(epoch):
    ''' Epoch operation in TEST phase '''
    model.eval()

    total_loss = 0
    n_word_total = 0
    n_word_correct = 0

    with torch.no_grad():
        for batch in tqdm(
                test_data, mininterval=2,
                desc=' (TEST: %d) ' % epoch, leave=False):

            enc_inputs, dec_inputs, enc_lengths, dec_lengths = map(
                lambda x: x.to(device), batch)

            dec_targets = dec_inputs[:, 1:]
            dec_inputs = dec_inputs[:, :-1]

            greedy_outputs, beam_outputs, beam_length = model.decode(
                enc_inputs,
                enc_lengths,
            )

            # [batch_size, max_len]
            greedy_texts = generate_texts(vocab, args.batch_size, greedy_outputs, decode_type='greedy')

            # [batch_size, topk, max_len]
            beam_texts = generate_texts(vocab, args.batch_size, beam_outputs, decode_type='beam_search')

            save_path = os.path.join(args.save_dir, 'generated/%d.txt' % epoch)

            save_generated_texts(epoch, greedy_texts, beam_texts, save_path)

def cal_performance(pred, gold, smoothing=False):
    ''' Apply label smoothing if needed '''
    # pred: [max_len * batch_size, vocab_size]
    # gold: [max_len * batch_size]

    loss = cal_loss(pred, gold, smoothing)

    pred = pred.max(1)[1]
    gold = gold.contiguous().view(-1)
    non_pad_mask = gold.ne(PAD_ID)
    n_correct = pred.eq(gold)
    n_correct = n_correct.masked_select(non_pad_mask).sum().item()

    return loss, n_correct


def cal_loss(pred, gold, smoothing):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.1
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        non_pad_mask = gold.ne(PAD_ID)
        loss = -(one_hot * log_prb).sum(dim=1)
        loss = loss.masked_select(non_pad_mask).sum()  # average later
    else:
        loss = F.cross_entropy(
            pred, gold, ignore_index=PAD_ID, reduction='sum')

    return loss


if __name__ == '__main__':
    train_epochs()
