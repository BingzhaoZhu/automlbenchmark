#!/usr/bin/env bash

rm -f ~/.config/automlbenchmark/config.yaml
cp ./examples/config_gpu.yaml ~/.config/automlbenchmark/config.yaml

#python runbenchmark.py GBM_AG ag mytest -m aws -p 1500 -f 0
#python runbenchmark.py RF_AG ag mytest -m aws -p 1500 -f 0
#python runbenchmark.py CAT_AG ag mytest -m aws -p 1500 -f 0
#python runbenchmark.py XGB_AG ag mytest -m aws -p 1500 -f 0

python runbenchmark.py FTTransformer_gpu ag mytest -m aws -p 150 -f 0

#python runbenchmark.py FTTransformer test mytest -f 0