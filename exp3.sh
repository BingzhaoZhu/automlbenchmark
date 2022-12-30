#!/usr/bin/env bash

# rm -f ~/.config/automlbenchmark/config.yaml
# cp ./examples/config_gpu.yaml ~/.config/automlbenchmark/config.yaml

python runbenchmark.py FTT_ft0_light ag_finetune mytest1h -m aws -p 520
python runbenchmark.py FTT_ft250_light ag_finetune mytest1h -m aws -p 520
python runbenchmark.py FTT_ft500_light ag_finetune mytest1h -m aws -p 520
python runbenchmark.py FTT_ft1000_light ag_finetune mytest1h -m aws -p 520
python runbenchmark.py FTT_ft1500_light ag_finetune mytest1h -m aws -p 520
python runbenchmark.py FTT_ft2000_light ag_finetune mytest1h -m aws -p 520

python runbenchmark.py FTT_pretrain_supervised_only_cls ag_pretrain mytest8h -m aws -p 180 -f 0
python runbenchmark.py FTT_ft1500 ag_finetune mytest1h -m aws -p 520
python runbenchmark.py FTT_ft2000 ag_finetune mytest1h -m aws -p 520
