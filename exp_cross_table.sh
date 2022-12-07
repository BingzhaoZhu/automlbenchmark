#!/usr/bin/env bash

rm -f ~/.config/automlbenchmark/config.yaml
cp ./examples/config_gpu.yaml ~/.config/automlbenchmark/config.yaml

#python runbenchmark.py FastFTT_AG_hog_ft0_lowe ag_finetune mytest1h -m aws -p 520
python runbenchmark.py FastFTT_AG_hog_ft250_lowe ag_finetune mytest1h -m aws -p 520
#python runbenchmark.py FastFTT_AG_hog_ft500_lowe ag_finetune mytest1h -m aws -p 520
python runbenchmark.py FastFTT_AG_hog_ft750_lowe ag_finetune mytest1h -m aws -p 520
#python runbenchmark.py FastFTT_AG_hog_ft1000_lowe ag_finetune mytest1h -m aws -p 520

python runbenchmark.py FastFTT_AG_hog_ft750 ag_finetune mytest1h -m aws -p 520
#python runbenchmark.py FastFTT_AG_hog_ft1000 ag_finetune mytest1h -m aws -p 520

#python runbenchmark.py FTT_AG_hog_ft0_lowi ag_finetune mytest1h -m aws -p 520
#python runbenchmark.py FTT_AG_hog_ft250_lowi ag_finetune mytest1h -m aws -p 520
#python runbenchmark.py FTT_AG_hog_ft500_lowi ag_finetune mytest1h -m aws -p 520
#python runbenchmark.py FTT_AG_hog_ft750_lowi ag_finetune mytest1h -m aws -p 520
#python runbenchmark.py FTT_AG_hog_ft1000_lowi ag_finetune mytest1h -m aws -p 520