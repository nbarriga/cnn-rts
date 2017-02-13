#!/usr/bin/env sh
export GREP_COLOR='1;32'
echo "Saving parameters ..."
NOW=$(date +"%m.%d")
mkdir -p old/$NOW
mkdir -p old/$NOW/$1
cp *.prot* old/$NOW/$1/

# python ../../python/draw_net.py  train_rts.prototxt old/$NOW-$1/rts.png

echo "Starting to train RTSnet ..."
#Start from scratch
#~/git-working/caffe/build/tools/caffe train --gpu 0 --solver rts_solver.prototxt 2>&1 | tee logs/log$1.log old/$NOW/$1/log.log | grep --color -E 'Test\ .*$|$'

#Start from pre-trained weights
#~/git-working/caffe/build/tools/caffe train --gpu 0 --solver rts_solver.prototxt --weights snapshots/24x24new-drop.2_iter_400000.caffemodel 2>&1 | tee logs/log$1.log old/$NOW/$1/log.log | grep --color -E 'Test\ .*$|$'

#Continue training (loads weights and solver state (LR, etc...))
#~/git-working/caffe/build/tools/caffe train --gpu 0 --solver rts_solver.prototxt --snapshot snapshots/24x24new_iter_50000.solverstate 2>&1 | tee logs/log$1.log old/$NOW/$1/log.log | grep --color -E 'Test\ .*$|$'

# 2> log1.log

