#!/usr/bin/env bash

rm -f ~/.config/automlbenchmark/config.yaml
cp ./examples/config_gpu.yaml ~/.config/automlbenchmark/config.yaml

python hog_start.py
python runbenchmark.py FastFTT_AG_hog ag_pretrain mytest8h -m aws -p 180 -f 0
#python runbenchmark.py FTT_AG_hog ag_pretrain mytest8h -m aws -p 180 -f 0
#python runbenchmark.py FTT_AG_hog_pretrain ag_pretrain mytest8h -m aws -p 180 -f 0
