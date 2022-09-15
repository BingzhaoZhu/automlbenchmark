#!/usr/bin/env bash

rm -f ~/.config/automlbenchmark/config.yaml
cp ./examples/config.yaml ~/.config/automlbenchmark/

#python runbenchmark.py GBM_AG ag mytest -m aws -p 1500 -f 0
#python runbenchmark.py RF_AG ag mytest -m aws -p 1500 -f 0
#python runbenchmark.py CAT_AG ag mytest -m aws -p 1500 -f 0
#python runbenchmark.py XGB_AG ag mytest -m aws -p 1500 -f 0
python runbenchmark.py GBM_AG test mytest -f 0