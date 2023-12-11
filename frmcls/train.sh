#!/bin/bash

exp=$1

hostname

for i in {1..10}; do
    src/train.py \
        --config $exp/train.conf \
        --pred-param $exp/param-$((i-1)) \
        --pred-param-output $exp/param-$i \
        > $exp/log-$i
done
