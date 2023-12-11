#!/bin/bash

exp=$1

for i in {1..10}; do
	tail -n+11 $exp/dev-$i.log > $exp/dev-$i-nohead.log 
done

for i in {1..10}; do
	util/eval-frames.py \
		$exp/dev-$i-nohead.log \
		/disk/scratch/s2293376/dataset/wsj/extra/si284-0.9-dev.bpali \
		> $exp/eval-$i.log
done

for i in {1..10}; do
	rm $exp/dev-$i-nohead.log 
done 

