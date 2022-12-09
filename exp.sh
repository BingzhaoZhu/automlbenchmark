#!/usr/bin/env bash

rm -f ~/.config/automlbenchmark/config.yaml
cp ./examples/config_gpu.yaml ~/.config/automlbenchmark/config.yaml

python runbenchmark.py FTT_ft0_fewshot ag_finetune mytest1h -m aws -p 520
python runbenchmark.py FTT_ft15k_fewshot ag_finetune mytest1h -m aws -p 520
python runbenchmark.py FTT_ft30k_fewshot ag_finetune mytest1h -m aws -p 520
python runbenchmark.py FTT_ft45k_fewshot ag_finetune mytest1h -m aws -p 520
