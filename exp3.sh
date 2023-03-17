#!/usr/bin/env bash

# rm -f ~/.config/automlbenchmark/config.yaml
# cp ./examples/config_gpu.yaml ~/.config/automlbenchmark/config.yaml

# python runbenchmark.py FTT_ft0_light ag_finetune mytest1h -m aws -p 520
# python runbenchmark.py FTT_ft250_light ag_finetune mytest1h -m aws -p 520
# python runbenchmark.py FTT_ft500_light ag_finetune mytest1h -m aws -p 520
# python runbenchmark.py FTT_ft1000_light ag_finetune mytest1h -m aws -p 520
# python runbenchmark.py FTT_ft1500_light ag_finetune mytest1h -m aws -p 520
# python runbenchmark.py FTT_ft2000_light ag_finetune mytest1h -m aws -p 520

# python runbenchmark.py FTT_ft1500_intense ag_finetune mytest1h -m aws -p 520
# python runbenchmark.py FTT_ft2000_intense ag_finetune mytest1h -m aws -p 520

# python runbenchmark.py FTT_ft0 ag_pretrain mytest1h -m aws -p 520
# python runbenchmark.py FTT_ft250 ag_pretrain mytest1h -m aws -p 520
# python runbenchmark.py FTT_ft500 ag_pretrain mytest1h -m aws -p 520
# python runbenchmark.py FTT_ft1000 ag_pretrain mytest1h -m aws -p 520

rm -f ~/.config/automlbenchmark/config.yaml
cp ./examples/config_gpu.yaml ~/.config/automlbenchmark/config.yaml
# python runbenchmark.py FASTAI_HPO ag mytest1h -m aws -p 270
python runbenchmark.py FTT_rebuttal_comp ag mytest1h -m aws -p 520
