#!/usr/bin/env bash

# rm -f ~/.config/automlbenchmark/config.yaml
# cp ./examples/config_gpu.yaml ~/.config/automlbenchmark/config.yaml

python runbenchmark.py FTT_ft0_cont ag_finetune mytest1h -m aws -p 520
python runbenchmark.py FTT_ft250_cont ag_finetune mytest1h -m aws -p 520
python runbenchmark.py FTT_ft500_cont ag_finetune mytest1h -m aws -p 520
python runbenchmark.py FTT_ft1000_cont ag_finetune mytest1h -m aws -p 520
python runbenchmark.py FTT_ft1500_cont ag_finetune mytest1h -m aws -p 520
python runbenchmark.py FTT_ft2000_cont ag_finetune mytest1h -m aws -p 520
