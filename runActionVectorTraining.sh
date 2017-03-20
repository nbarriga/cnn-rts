#!/usr/bin/env sh
export GREP_COLOR='1;32'
#echo "Saving parameters ..."
#NOW=$(date +"%m.%d")
#mkdir -p old/$NOW
#mkdir -p old/$NOW/$1
#cp *.prot* old/$NOW/$1/

# python ../../python/draw_net.py  train_rts.prototxt old/$NOW-$1/rts.png

echo "Starting to train RTSnet ..."
#rm train_list.txt
#rm test_list.txt
#ln -s /common/barriga/puppet-data/10secs4samples/train128x128 train128x128
#ln -s /common/barriga/puppet-data/10secs4samples/test128x128 test128x128
sed "\
	s/net:.*/net: \"actionvector.prototxt\"/g;\
	s/base_lr:.*/base_lr: 1e-4/g;\
	s/test_iter:.*/test_iter: 10/g;\
	s/test_interval:.*/test_interval: 100/g;\
	s/display:.*/display: 20/g;\
	s/max_iter:.*/max_iter: 100000/g;\
	s/snapshot:.*/snapshot: 5000/g;\
	s/snapshot_prefix:.*/snapshot_prefix: \"snapshots\/actionvectorSmall\"/g;\
	" rts_solver.prototxt >rts_solver.prototxt.action.tmp
~/git-working/caffe/build/tools/caffe train --gpu 0 --solver rts_solver.prototxt.action.tmp 2>&1 | tee logs/actionVectorSmall.log | grep --color -E 'Test\ .*$|$'
