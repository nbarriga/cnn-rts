#!/usr/bin/env sh
export GREP_COLOR='1;32'
#echo "Saving parameters ..."
#NOW=$(date +"%m.%d")
#mkdir -p old/$NOW
#mkdir -p old/$NOW/$1
#cp *.prot* old/$NOW/$1/

# python ../../python/draw_net.py  train_rts.prototxt old/$NOW-$1/rts.png

echo "Starting to train RTSnet ..."
#Start from scratch
#~/git-working/caffe/build/tools/caffe train --gpu 0 --solver rts_solver.prototxt 2>&1 | tee logs/log$1.log | grep --color -E 'Test\ .*$|$'

#Start from pre-trained weights
#~/git-working/caffe/build/tools/caffe train --gpu 0 --solver rts_solver.prototxt --weights snapshots/24x24new-drop.2_iter_400000.caffemodel 2>&1 | tee logs/log$1.log old/$NOW/$1/log.log | grep --color -E 'Test\ .*$|$'

#Continue training (loads weights and solver state (LR, etc...))
#~/git-working/caffe/build/tools/caffe train --gpu 0 --solver rts_solver.prototxt --snapshot snapshots/24x24new_iter_50000.solverstate 2>&1 | tee logs/log$1.log old/$NOW/$1/log.log | grep --color -E 'Test\ .*$|$'

# 2> log1.log

rm train_list.txt
rm test_list.txt
ln -s /common/barriga/caffe-data/train8x8_list.txt train_list.txt
ln -s /common/barriga/caffe-data/test8x8_list.txt test_list.txt
sed "\
	s/base_lr:.*/base_lr: 1e-3/g;\
	s/test_iter:.*/test_iter: 32/g;\
	s/test_interval:.*/test_interval: 500/g;\
	s/display:.*/display: 100/g;\
	s/max_iter:.*/max_iter: 1000000/g;\
	s/snapshot:.*/snapshot: 10000/g;\
	s/snapshot_prefix:.*/snapshot_prefix: \"snapshots\/8x8lr3\"/g;\
	" rts_solver.prototxt >rts_solver.prototxt.tmp
~/git-working/caffe/build/tools/caffe train --gpu 0 --solver rts_solver.prototxt.tmp 2>&1 | tee logs/log8x8lr3.log | grep --color -E 'Test\ .*$|$'

rm train_list.txt
rm test_list.txt
ln -s /common/barriga/caffe-data/train16x16_list.txt train_list.txt
ln -s /common/barriga/caffe-data/test16x16_list.txt test_list.txt
sed "\
	s/base_lr:.*/base_lr: 1e-3/g;\
	s/test_iter:.*/test_iter: 32/g;\
	s/test_interval:.*/test_interval: 500/g;\
	s/display:.*/display: 100/g;\
	s/max_iter:.*/max_iter: 1000000/g;\
	s/snapshot:.*/snapshot: 10000/g;\
	s/snapshot_prefix:.*/snapshot_prefix: \"snapshots\/16x16lr3\"/g;\
	" rts_solver.prototxt >rts_solver.prototxt.tmp
~/git-working/caffe/build/tools/caffe train --gpu 0 --solver rts_solver.prototxt.tmp 2>&1 | tee logs/log16x16lr3.log | grep --color -E 'Test\ .*$|$'

rm train_list.txt
rm test_list.txt
ln -s /common/barriga/caffe-data/train24x24_list.txt train_list.txt
ln -s /common/barriga/caffe-data/test24x24_list.txt test_list.txt
sed "\
	s/base_lr:.*/base_lr: 1e-3/g;\
	s/test_iter:.*/test_iter: 28/g;\
	s/test_interval:.*/test_interval: 500/g;\
	s/display:.*/display: 100/g;\
	s/max_iter:.*/max_iter: 1000000/g;\
	s/snapshot:.*/snapshot: 10000/g;\
	s/snapshot_prefix:.*/snapshot_prefix: \"snapshots\/24x24lr3\"/g;\
	" rts_solver.prototxt >rts_solver.prototxt.tmp
~/git-working/caffe/build/tools/caffe train --gpu 0 --solver rts_solver.prototxt.tmp 2>&1 | tee logs/log24x24lr3.log | grep --color -E 'Test\ .*$|$'

rm train_list.txt
rm test_list.txt
ln -s /common/barriga/caffe-data/train128x128_list.txt train_list.txt
ln -s /common/barriga/caffe-data/test128x128_list.txt test_list.txt
sed "\
	s/base_lr:.*/base_lr: 1e-4/g;\
	s/test_iter:.*/test_iter: 12/g;\
	s/test_interval:.*/test_interval: 100/g;\
	s/display:.*/display: 20/g;\
	s/max_iter:.*/max_iter: 100000/g;\
	s/snapshot:.*/snapshot: 5000/g;\
	s/snapshot_prefix:.*/snapshot_prefix: \"snapshots\/128x128\"/g;\
	" rts_solver.prototxt >rts_solver.prototxt.tmp
~/git-working/caffe/build/tools/caffe train --gpu 0 --solver rts_solver.prototxt.tmp 2>&1 | tee logs/log$1.log | grep --color -E 'Test\ .*$|$'
