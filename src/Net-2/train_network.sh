#!/usr/bin/env sh
set -e

nohup ./build/tools/caffe train --solver=solver.prototxt --gpu=0 2>&1 | tee network_train_val.log &
