#!/usr/bin/env bash

rm -f ~/.config/automlbenchmark/config.yaml
cp ./examples/config_gpu.yaml ~/.config/automlbenchmark/config.yaml

python runbenchmark.py FTT_AG_hog_ft0 ag mytest1h -m aws -p 180 -f 0
python runbenchmark.py FTT_AG_hog_ft500 ag mytest1h -m aws -p 180 -f 0
python runbenchmark.py FTT_AG_hog_ft1000 ag mytest1h -m aws -p 180 -f 0

python runbenchmark.py FTT_AG_hog_ft0_lowp ag mytest1h -m aws -p 180 -f 0
python runbenchmark.py FTT_AG_hog_ft500_lowp ag mytest1h -m aws -p 180 -f 0
python runbenchmark.py FTT_AG_hog_ft1000_lowp ag mytest1h -m aws -p 180 -f 0