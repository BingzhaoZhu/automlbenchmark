#!/usr/bin/env bash

# rm -f ~/.config/automlbenchmark/config.yaml
# cp ./examples/config_gpu.yaml ~/.config/automlbenchmark/config.yaml

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

# python runbenchmark.py FTT_ft0_intense ag_finetune mytest1h -m aws -p 520
# python runbenchmark.py FTT_ft250_intense ag_finetune mytest1h -m aws -p 520

# python runbenchmark.py FTT_ft0_fold_1 ag_finetune mytest1h -m aws -p 520
# python runbenchmark.py FTT_ft250_fold_1 ag_finetune mytest1h -m aws -p 520
# python runbenchmark.py FTT_ft500_fold_1 ag_finetune mytest1h -m aws -p 520
# python runbenchmark.py FTT_ft1000_fold_1 ag_finetune mytest1h -m aws -p 520

rm -f ~/.config/automlbenchmark/config.yaml
cp ./examples/config_gpu.yaml ~/.config/automlbenchmark/config.yaml
# python runbenchmark.py FTT_HPO ag mytest1h -m aws -p 520
# python runbenchmark.py FTT_fold1_pretrained ag_finetune mytest1h -m aws -p 520
# python runbenchmark.py FTT_fold2_pretrained ag_pretrain mytest1h -m aws -p 520

# python runbenchmark.py TransTab ag mytest1h -m aws -p 520
# python runbenchmark.py FTT_rebuttal ag mytest1h -m aws -p 520
# python runbenchmark.py FTT_rebuttal_2000 ag mytest1h -m aws -p 520

python runbenchmark.py FTT_rebuttal_seed_0 ag_finetune mytest1h -m aws -p 520
python runbenchmark.py FTT_rebuttal_2000_seed_0 ag_finetune mytest1h -m aws -p 520
python runbenchmark.py FTT_rebuttal_seed_1 ag_finetune mytest1h -m aws -p 520
python runbenchmark.py FTT_rebuttal_2000_seed_1 ag_finetune mytest1h -m aws -p 520
