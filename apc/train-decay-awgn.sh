#!/bin/bash

exp=$1

for i in {1..5}; do
    step=$(echo "scale=10; 0.05 * 0.5 ^ $i" | bc)

    src/train-awgn-snr30.py --config $exp/train.conf \
        --param $exp/param-decay-$((i-1)) \
        --param-output $exp/param-decay-$i \
        --step-size $step \
        > $exp/log-decay-$i
done
