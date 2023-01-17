#!/usr/bin/env bash

rm -f ~/.config/automlbenchmark/config.yaml
cp ./examples/config_gpu.yaml ~/.config/automlbenchmark/config.yaml

# python runbenchmark.py FTT_ft0_fewshot ag_finetune mytest1h -m aws -p 520
# python runbenchmark.py FTT_ft250_fewshot ag_finetune mytest1h -m aws -p 520
# python runbenchmark.py FTT_ft500_fewshot ag_finetune mytest1h -m aws -p 520
# python runbenchmark.py FTT_ft1000_fewshot ag_finetune mytest1h -m aws -p 520
# python runbenchmark.py FTT_ft1500_fewshot ag_finetune mytest1h -m aws -p 520
# python runbenchmark.py FTT_ft2000_fewshot ag_finetune mytest1h -m aws -p 520

# python runbenchmark.py FTT_ft0_with_cls ag_finetune mytest1h -m aws -p 520
# python runbenchmark.py FTT_ft250_with_cls ag_finetune mytest1h -m aws -p 520
# python runbenchmark.py FTT_ft500_with_cls ag_finetune mytest1h -m aws -p 520
# python runbenchmark.py FTT_ft1000_with_cls ag_finetune mytest1h -m aws -p 520
# python runbenchmark.py FTT_ft1500_with_cls ag_finetune mytest1h -m aws -p 520
# python runbenchmark.py FTT_ft2000_with_cls ag_finetune mytest1h -m aws -p 520

# python runbenchmark.py FTT_ft500_fewshot ag_finetune mytest1h -m aws -p 520
# python runbenchmark.py FTT_ft1000_fewshot ag_finetune mytest1h -m aws -p 520
# python runbenchmark.py FTT_ft1500_fewshot ag_finetune mytest1h -m aws -p 520
# python runbenchmark.py FTT_ft2000_fewshot ag_finetune mytest1h -m aws -p 520

python runbenchmark.py FTT_ft0 ag_pretrain mytest1h -m aws -p 520
python runbenchmark.py FTT_ft0_intense_1 ag_pretrain mytest1h -m aws -p 520
python runbenchmark.py FTT_ft0_intense_2 ag_pretrain mytest1h -m aws -p 520