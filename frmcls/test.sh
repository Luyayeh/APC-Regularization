#!/bin/bash

exp=$1
epoch=$2

hostname

src/predict.py \
    --config $exp/test.conf \
    --pred-param $exp/param-$epoch \
    > $exp/dev-$epoch.log
