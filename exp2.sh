#!/usr/bin/env bash

# rm -f ~/.config/automlbenchmark/config.yaml
# cp ./examples/config_gpu.yaml ~/.config/automlbenchmark/config.yaml

python runbenchmark.py FTT_ft0 ag_finetune mytest1h -m aws -p 520
python runbenchmark.py FTT_ft250 ag_finetune mytest1h -m aws -p 520
python runbenchmark.py FTT_ft500 ag_finetune mytest1h -m aws -p 520
python runbenchmark.py FTT_ft1000 ag_finetune mytest1h -m aws -p 520
python runbenchmark.py FTT_ft1500 ag_finetune mytest1h -m aws -p 520
python runbenchmark.py FTT_ft2000 ag_finetune mytest1h -m aws -p 520
