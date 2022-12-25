#!/usr/bin/env bash

# rm -f ~/.config/automlbenchmark/config.yaml
# cp ./examples/config_gpu.yaml ~/.config/automlbenchmark/config.yaml

# python runbenchmark.py FTT_pretrain_reconstruction ag_pretrain mytest8h -m aws -p 180 -f 0
# python runbenchmark.py FTT_pretrain_all ag_pretrain mytest8h -m aws -p 180 -f 0
# python runbenchmark.py FTT_pretrain_reconstruction_1 ag_pretrain mytest8h -m aws -p 180 -f 0
# python runbenchmark.py FTT_pretrain_reconstruction_10 ag_pretrain mytest8h -m aws -p 180 -f 0
# python runbenchmark.py FTT_pretrain_reconstruction_25 ag_pretrain mytest8h -m aws -p 180 -f 0

python runbenchmark.py Saint_pretrain_reconstruction ag_pretrain mytest8h -m aws -p 180 -f 0
python runbenchmark.py Fastformer_pretrain_reconstruction ag_pretrain mytest8h -m aws -p 180 -f 0
