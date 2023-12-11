#!/bin/bash

exp=$1

for i in {1..20}; do
    src/test.py --config $exp/train.conf \
        --param $exp/param-$((i)) \
        > $exp/test-log-$i
done
