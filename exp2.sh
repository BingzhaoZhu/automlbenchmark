#!/usr/bin/env bash

# rm -f ~/.config/automlbenchmark/config.yaml
# cp ./examples/config_gpu.yaml ~/.config/automlbenchmark/config.yaml

python runbenchmark.py FTT_ft0_fewshot_2k ag_finetune mytest1h -m aws -p 520
python runbenchmark.py FTT_ft250_fewshot_2k ag_finetune mytest1h -m aws -p 520
python runbenchmark.py FTT_ft500_fewshot_2k ag_finetune mytest1h -m aws -p 520
python runbenchmark.py FTT_ft1000_fewshot_2k ag_finetune mytest1h -m aws -p 520
python runbenchmark.py FTT_ft1500_fewshot_2k ag_finetune mytest1h -m aws -p 520
python runbenchmark.py FTT_ft2000_fewshot_2k ag_finetune mytest1h -m aws -p 520
