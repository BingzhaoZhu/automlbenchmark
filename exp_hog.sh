#!/usr/bin/env bash

rm -f ~/.config/automlbenchmark/config.yaml
cp ./examples/config_gpu.yaml ~/.config/automlbenchmark/config.yaml

python hog_start.py
python runbenchmark.py FTT_AG_hog ag mytest2h -m aws -p 104 -f 0
