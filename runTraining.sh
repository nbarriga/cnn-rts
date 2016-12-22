#!/usr/bin/env sh
export GREP_COLOR='1;32'
echo "Saving parameters ..."
NOW=$(date +"%m.%d")
mkdir -p old/$NOW
mkdir -p old/$NOW/$1
cp *.prot* old/$NOW/$1/

# python ../../python/draw_net.py  train_rts.prototxt old/$NOW-$1/rts.png

echo "Starting to train RTSnet ..."
~/git-working/caffe/build/tools/caffe train --gpu 0 --solver rts_solver.prototxt 2>&1 | tee logs/log$1.log old/$NOW/$1/log.log | grep --color -E 'Test\ .*$|$'

# 2> log1.log

