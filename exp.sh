#!/usr/bin/env bash

# rm -f ~/.config/automlbenchmark/config.yaml
# cp ./examples/config_gpu.yaml ~/.config/automlbenchmark/config.yaml

# python runbenchmark.py FTT_ft0_fewshot ag_finetune mytest1h -m aws -p 520
# python runbenchmark.py FTT_ft250_fewshot ag_finetune mytest1h -m aws -p 520
# python runbenchmark.py FTT_ft500_fewshot ag_finetune mytest1h -m aws -p 520
# python runbenchmark.py FTT_ft1000_fewshot ag_finetune mytest1h -m aws -p 520
# python runbenchmark.py FTT_ft1500_fewshot ag_finetune mytest1h -m aws -p 520
# python runbenchmark.py FTT_ft2000_fewshot ag_finetune mytest1h -m aws -p 520

python runbenchmark.py Saint_ft0_light ag_finetune mytest1h -m aws -p 520
python runbenchmark.py Saint_ft250_light ag_finetune mytest1h -m aws -p 520
python runbenchmark.py Saint_ft500_light ag_finetune mytest1h -m aws -p 520
python runbenchmark.py Saint_ft1000_light ag_finetune mytest1h -m aws -p 520
python runbenchmark.py Saint_ft1500_light ag_finetune mytest1h -m aws -p 520
python runbenchmark.py Saint_ft2000_light ag_finetune mytest1h -m aws -p 520

python runbenchmark.py FTT_ft500 ag_finetune mytest1h -m aws -p 520
python runbenchmark.py FTT_ft1000 ag_finetune mytest1h -m aws -p 520