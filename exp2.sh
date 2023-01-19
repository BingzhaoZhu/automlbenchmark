#!/usr/bin/env bash

# rm -f ~/.config/automlbenchmark/config.yaml
# cp ./examples/config_gpu.yaml ~/.config/automlbenchmark/config.yaml

# python runbenchmark.py FTT_ft0 ag_finetune mytest1h -m aws -p 520
# python runbenchmark.py FTT_ft250 ag_finetune mytest1h -m aws -p 520
# python runbenchmark.py FTT_ft500 ag_finetune mytest1h -m aws -p 520
# python runbenchmark.py FTT_ft1000 ag_finetune mytest1h -m aws -p 520
# python runbenchmark.py FTT_ft1500 ag_finetune mytest1h -m aws -p 520
# python runbenchmark.py FTT_ft2000 ag_finetune mytest1h -m aws -p 520

# python runbenchmark.py FTT_ft0_only_cls ag_finetune mytest1h -m aws -p 520
# python runbenchmark.py FTT_ft250_only_cls ag_finetune mytest1h -m aws -p 520
# python runbenchmark.py FTT_ft500_only_cls ag_finetune mytest1h -m aws -p 520
# python runbenchmark.py FTT_ft1000_only_cls ag_finetune mytest1h -m aws -p 520
# python runbenchmark.py FTT_ft1500_only_cls ag_finetune mytest1h -m aws -p 520
# python runbenchmark.py FTT_ft2000_only_cls ag_finetune mytest1h -m aws -p 520

# python runbenchmark.py FTT_ft0 ag_finetune mytest1h -m aws -p 520
# python runbenchmark.py FTT_ft250 ag_finetune mytest1h -m aws -p 520
# python runbenchmark.py FTT_ft0_fewshot ag_finetune mytest1h -m aws -p 520
# python runbenchmark.py FTT_ft250_fewshot ag_finetune mytest1h -m aws -p 520

# python runbenchmark.py FTT_ft2000 ag_pretrain mytest1h -m aws -p 520
# python runbenchmark.py FTT_ft2000_intense_1 ag_pretrain mytest1h -m aws -p 520
# python runbenchmark.py FTT_ft2000_intense_2 ag_pretrain mytest1h -m aws -p 520

# python runbenchmark.py FTT_ft500_intense ag_finetune mytest1h -m aws -p 520
# python runbenchmark.py FTT_ft1000_intense ag_finetune mytest1h -m aws -p 520

# python runbenchmark.py FTT_ft1500_fold_1 ag_finetune mytest1h -m aws -p 520
# python runbenchmark.py FTT_ft2000_fold_1 ag_finetune mytest1h -m aws -p 520

# python runbenchmark.py FTT_ft1500 ag_pretrain mytest1h -m aws -p 520
# python runbenchmark.py FTT_ft2000 ag_pretrain mytest1h -m aws -p 520

rm -f ~/.config/automlbenchmark/config.yaml
cp ./examples/config_m6i.yaml ~/.config/automlbenchmark/config.yaml
python runbenchmark.py CAT_HPO ag mytest1h -m aws -p 1040
python runbenchmark.py XGB_HPO ag mytest1h -m aws -p 1040
