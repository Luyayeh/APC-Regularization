#!/bin/bash

hostname

exp=$1
epoch=$2
dropout=$3

src/train-awgn-snr10.py --config $exp/train.conf \
    --dropout $dropout \
    --param $exp/param-$((epoch-1)) \
    --param-output $exp/param-$epoch \
    > $exp/log-$epoch
