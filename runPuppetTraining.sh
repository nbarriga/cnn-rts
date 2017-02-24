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
#ln -s /common/barriga/puppet-data/train128x128_list.txt train_list.txt
#ln -s /common/barriga/puppet-data/test128x128_list.txt test_list.txt
sed "\
	s/net:.*/net: \"puppet.prototxt\"/g;\
	s/base_lr:.*/base_lr: 1e-4/g;\
	s/test_iter:.*/test_iter: 4/g;\
	s/test_interval:.*/test_interval: 100/g;\
	s/display:.*/display: 20/g;\
	s/max_iter:.*/max_iter: 100000/g;\
	s/snapshot:.*/snapshot: 5000/g;\
	s/snapshot_prefix:.*/snapshot_prefix: \"snapshots\/puppet128x128\"/g;\
	" rts_solver.prototxt >rts_solver.prototxt.tmp
~/git-working/caffe/build/tools/caffe train --gpu 0 --solver rts_solver.prototxt.tmp 2>&1 | tee logs/logPuppet128x128.log | grep --color -E 'Test\ .*$|$'
