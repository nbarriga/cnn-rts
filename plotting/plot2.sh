#!/usr/bin/env sh
echo "Plot log"
./plot.py $1 ../logs/log$2_$3.png ../logs/log$2.log ../logs/log$3.log

# train --solver rts_solver.prototxt 2>&1 | tee log$1.log
# /plot.py chart_type 6 tl1.png ../../log1.log 
# 2> log1.log

