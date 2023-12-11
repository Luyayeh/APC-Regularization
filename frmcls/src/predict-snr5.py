#!/usr/bin/env python3

import sys
import math
import data
import torch.nn
import torch.optim
import argparse
import json
import rand
import numpy


class ApcLstm(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.0):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout,
            bidirectional=False, batch_first=True)
        self.proj = torch.nn.Linear(hidden_size, input_size)

    def forward(self, feat):
        hidden, _ = self.lstm(feat)
        return hidden, self.proj(hidden)


class FrmLstm(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout=0.0):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout,
            bidirectional=False, batch_first=True)
        self.pred = torch.nn.Linear(hidden_size, output_size)

    def forward(self, feat):
        hidden, _ = self.lstm(feat)
        return hidden, self.pred(hidden)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--label-set')
    parser.add_argument('--feat-scp')
    parser.add_argument('--feat-mean-var')
    parser.add_argument('--layers', type=int)
    parser.add_argument('--hidden-size', type=int)
    parser.add_argument('--apc-param')
    parser.add_argument('--pred-param')
    
    if len(sys.argv) == 1:
        parser.print_help()
        exit(1)
    
    args = parser.parse_args()

    if args.config:
        f = open(args.config)
        config = json.load(f)
        f.close()
    
        for k, v in config.items():
            if k not in args.__dict__ or args.__dict__[k] is None:
                args.__dict__[k] = v

    for k, v in vars(args).items():
        print('{}: {}'.format(k, v))
    print()

    return args


print(' '.join(sys.argv))
args = parse_args()

f = open(args.label_set)
id_label = []
for line in f:
    id_label.append(line.strip())

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = FrmLstm(40, args.hidden_size, len(id_label), args.layers)

ckpt = torch.load(args.pred_param)
model.pred.load_state_dict(ckpt['pred'])

ckpt = torch.load(args.apc_param)
p = {}
for k in ckpt['model']:
    if k.startswith('lstm'):
        layer_index = int(k.split('_')[-1][1:])  # Extract the layer index from the key
        if layer_index < args.layers:
            p[k[len('lstm.'):]] = ckpt['model'][k]
model.lstm.load_state_dict(p)

model.to(device)
model.eval()

feat_mean, feat_var = data.load_mean_var(args.feat_mean_var)

dataset = data.WsjFeat(args.feat_scp, feat_mean, feat_var, snr = 5)

for sample, (key, feat) in enumerate(dataset):
    print(key)

    feat = torch.Tensor(feat).to(device)

    nframes, ndim = feat.shape
    feat = feat.reshape(1, nframes, ndim)

    hidden, pred = model(feat)

    _, nframes, nclass = pred.shape
    pred = pred.reshape(nframes, nclass)

    labels = torch.argmax(pred, dim=1)

    result = [id_label[int(e)] for e in labels]

    print(' '.join(result))
    print('.')

