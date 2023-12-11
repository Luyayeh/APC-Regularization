#!/usr/bin/env python3

import sys
import math
import data
import torch.nn
import torch.optim
import argparse
import json
import rand


class UniLstm(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.0):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout,
            bidirectional=False, batch_first=True)
        self.proj = torch.nn.Linear(hidden_size, input_size)

    def forward(self, feat):
        hidden, _ = self.lstm(feat)
        return hidden, self.proj(hidden)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--feat-scp')
    parser.add_argument('--feat-mean-var')
    parser.add_argument('--dropout', type=float)
    parser.add_argument('--layers', type=int)
    parser.add_argument('--hidden-size', type=int)
    parser.add_argument('--step-size', type=float)
    parser.add_argument('--grad-clip', type=float)
    parser.add_argument('--time-shift', type=int)
    parser.add_argument('--init', action="store_true")
    parser.add_argument('--param')
    parser.add_argument('--param-output')
    
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

    if args.dropout is None:
        args.dropout = 0.0

    for k, v in vars(args).items():
        print('{}: {}'.format(k, v))
    print()

    return args


print(' '.join(sys.argv))
args = parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('device:', device)
print()

model = UniLstm(40, args.hidden_size, args.layers, args.dropout)
loss_fn = torch.nn.MSELoss(reduction='sum')
total_param = list(model.parameters())

opt = torch.optim.SGD(total_param, lr=0.0)

if args.param:
    checkpoint = torch.load(args.param)
    model.load_state_dict(checkpoint['model'])
    opt.load_state_dict(checkpoint['opt'])

if args.init:
    torch.save(
        {
            'model': model.state_dict(),
            'opt': opt.state_dict()
        },
        args.param_output)
    exit()

model.to(device)

step_size = args.step_size
grad_clip = args.grad_clip
shift = args.time_shift

feat_mean, feat_var = data.load_mean_var(args.feat_mean_var)

rand_eng = rand.Random()

dataset = data.LibriSpeech(args.feat_scp, feat_mean, feat_var, shuffling=True, rand=rand_eng, snr = 10)

for sample, (key, feat) in enumerate(dataset):
    feat = torch.Tensor(feat).to(device)

    nframes, ndim = feat.shape
    feat = feat.reshape(1, nframes, ndim)

    opt.zero_grad()

    hidden, pred = model(feat)
    loss = loss_fn(pred[:, :-shift, :], feat[:, shift:, :])

    print('sample:', sample)
    print('key:', key)
    print('frames:', nframes)
    print('loss: {:.6}'.format(loss.item() / (nframes - shift)))

    loss.backward()

    total_norm = 0
    for p in total_param:
        n = p.grad.norm(2).item()
        total_norm += n * n
    grad_norm = math.sqrt(total_norm)

    print('grad norm: {:.6}'.format(grad_norm))

    param_0 = total_param[0][0, 0].item()

    if grad_norm > grad_clip:
        opt.param_groups[0]['lr'] = step_size / grad_norm * grad_clip
    else:
        opt.param_groups[0]['lr'] = step_size

    opt.step()

    param_0_new = total_param[0][0, 0].item()

    print('param: {:.6}, update: {:.6}, rate: {:.6}'.format(param_0, param_0_new - param_0, (param_0_new - param_0) / param_0))

    print()


torch.save(
    {
        'model': model.state_dict(),
        'opt': opt.state_dict()
    },
    args.param_output)

