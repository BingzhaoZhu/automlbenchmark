#!/usr/bin/env bash

#rm -f ~/.config/automlbenchmark/config.yaml
#cp ./examples/config_cpu.yaml ~/.config/automlbenchmark/config.yaml
#python runbenchmark.py GBM_AG ag mytest -m aws -p 1500 -f 0
#python runbenchmark.py RF_AG ag mytest -m aws -p 1500 -f 0
#python runbenchmark.py CAT_AG ag mytest -m aws -p 1500 -f 0
#python runbenchmark.py XGB_AG ag mytest -m aws -p 1500 -f 0
#python runbenchmark.py CAT_AG_pretrain ag mytest -m aws -p 1500 -f 0
#python runbenchmark.py NN_AG ag mytest -m aws -p 1500 -f 0
#python runbenchmark.py FASTAI_AG ag mytest -m aws -p 1500 -f 0
#python runbenchmark.py FTT_AG ag mytest -m aws -p 1500 -f 0

#python runbenchmark.py CAT_AG_pretrain ag mytest -m aws -p 1500 -f 0

#sleep 3600
rm -f ~/.config/automlbenchmark/config.yaml
cp ./examples/config_gpu.yaml ~/.config/automlbenchmark/config.yaml

#python runbenchmark.py FTT_AG ag mytest -m aws -p 104 -f 0
#python runbenchmark.py FastFTT_AG_32 ag mytest -m aws -p 104 -f 0
#python runbenchmark.py FTT_AG_32 ag mytest -m aws -p 104 -f 0
#python runbenchmark.py FastFTT_AG ag mytest -m aws -p 104 -f 0

#python runbenchmark.py FTT_AG_pretrain_identical ag mytest -m aws -p 104 -f 0
python runbenchmark.py FTT_AG_pretrain_randperm_03 ag mytest -m aws -p 104 -f 0
#python runbenchmark.py FTT_AG_pretrain_randperm_06 ag mytest -m aws -p 104 -f 0
python runbenchmark.py FTT_AG_pretrain_randperm_09 ag mytest -m aws -p 104 -f 0

#python runbenchmark.py FTT_AG_pretrain_randblk_03 ag mytest -m aws -p 104 -f 0
#python runbenchmark.py FTT_AG_pretrain_randblk_06 ag mytest -m aws -p 104 -f 0
#python runbenchmark.py FTT_AG_pretrain_randblk_09 ag mytest -m aws -p 104 -f 0

#python runbenchmark.py FTT_AG_row_attention ag mytest -m aws -p 104 -f 0

#python runbenchmark.py HTT_AG ag mytest -m aws -p 104 -f 0