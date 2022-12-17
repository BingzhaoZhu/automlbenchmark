#!/usr/bin/env bash

# rm -f ~/.config/automlbenchmark/config.yaml
# cp ./examples/config_gpu.yaml ~/.config/automlbenchmark/config.yaml

# python runbenchmark.py FTT_pretrain_reconstruction ag_pretrain mytest8h -m aws -p 180 -f 0
python runbenchmark.py FTT_pretrain_supervised ag_pretrain mytest8h -m aws -p 180 -f 0
python runbenchmark.py FTT_pretrain_contrastive ag_pretrain mytest8h -m aws -p 180 -f 0
