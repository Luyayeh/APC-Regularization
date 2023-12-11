#!/bin/bash

exp=$1
dropout=$2

for i in {1..5}; do
    step=$(echo "scale=10; 0.05 * 0.5 ^ $i" | bc)

    src/train-awgn-snr10.py --config $exp/train.conf \
        --dropout $dropout \
        --param $exp/param-decay-$((i-1)) \
        --param-output $exp/param-decay-$i \
        --step-size $step \
        > $exp/log-decay-$i
done
