#!/bin/bash

exp=$1

for i in {1..10}; do
    src/train-probe.py \
        --config $exp/train.conf \
        --pred-param $exp/param-$((i-1)) \
        --pred-param-output $exp/param-$i \
        --seed $i \
        > $exp/log-$i
done
