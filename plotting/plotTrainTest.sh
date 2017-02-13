#!/usr/bin/env sh
echo "Plot log"
cp ../logs/log$1.log log$1.log
./plotTrainTest.py 0 ../logs/log$1.png log$1.log

# train --solver rts_solver.prototxt 2>&1 | tee log$1.log
# /plot.py chart_type 6 tl1.png ../../log1.log 
# 2> log1.log

