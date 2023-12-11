#!/bin/bash

hostname

exp=$1
epoch=$2

src/train.py --config $exp/train.conf \
    --param $exp/param-$((epoch-1)) \
    --param-output $exp/param-$epoch \
    > $exp/log-$epoch
