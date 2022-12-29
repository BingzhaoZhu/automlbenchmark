#!/usr/bin/env bash

# rm -f ~/.config/automlbenchmark/config.yaml
# cp ./examples/config_gpu.yaml ~/.config/automlbenchmark/config.yaml

# python runbenchmark.py FTT_pretrain_reconstruction ag_pretrain mytest8h -m aws -p 180 -f 0
# python runbenchmark.py FTT_pretrain_all ag_pretrain mytest8h -m aws -p 180 -f 0
# python runbenchmark.py FTT_pretrain_reconstruction_10 ag_pretrain mytest8h -m aws -p 180 -f 0
# python runbenchmark.py FTT_pretrain_reconstruction_5 ag_pretrain mytest8h -m aws -p 180 -f 0
# python runbenchmark.py FTT_pretrain_reconstruction_1 ag_pretrain mytest8h -m aws -p 180 -f 0

# python runbenchmark.py Saint_pretrain_reconstruction ag_pretrain mytest8h -m aws -p 180 -f 0
# python runbenchmark.py Fastformer_pretrain_reconstruction ag_pretrain mytest8h -m aws -p 180 -f 0

# python runbenchmark.py Saint_ft0 ag_finetune mytest1h -m aws -p 520
# python runbenchmark.py Saint_ft250 ag_finetune mytest1h -m aws -p 520
# python runbenchmark.py Saint_ft500 ag_finetune mytest1h -m aws -p 520
# python runbenchmark.py Saint_ft1000 ag_finetune mytest1h -m aws -p 520
# python runbenchmark.py Saint_ft1500 ag_finetune mytest1h -m aws -p 520
# python runbenchmark.py Saint_ft2000 ag_finetune mytest1h -m aws -p 520

python runbenchmark.py FTT_pretrain_reconstruction_18_task ag_pretrain_18 mytest8h -m aws -p 180 -f 0
python runbenchmark.py FTT_pretrain_reconstruction_36_task ag_pretrain_36 mytest8h -m aws -p 180 -f 0
python runbenchmark.py FTT_pretrain_reconstruction_52_task ag_pretrain mytest8h -m aws -p 180 -f 0

python runbenchmark.py FTT_pretrain_supervised_blk_1 ag_pretrain mytest8h -m aws -p 180 -f 0
python runbenchmark.py FTT_pretrain_supervised_blk_2 ag_pretrain mytest8h -m aws -p 180 -f 0
python runbenchmark.py FTT_pretrain_supervised_blk_3 ag_pretrain mytest8h -m aws -p 180 -f 0
python runbenchmark.py FTT_pretrain_supervised_with_cls ag_pretrain mytest8h -m aws -p 180 -f 0
python runbenchmark.py FTT_pretrain_supervised_only_cls ag_pretrain mytest8h -m aws -p 180 -f 0

