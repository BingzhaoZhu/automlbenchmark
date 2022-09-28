#!/usr/bin/env bash

cd ../autogluon
git commit -a -m "added pretrain"
git push
cd ../automlbenchmark
git commit -a -m "added pretrain"
git push


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
#rm -f ~/.config/automlbenchmark/config.yaml
#cp ./examples/config_gpu.yaml ~/.config/automlbenchmark/config.yaml
##python runbenchmark.py FTT_AG ag mytest -m aws -p 150 -f 0
#python runbenchmark.py FTT_AG_pretrain_identical ag mytest -m aws -p 30 -f 0
#python runbenchmark.py FTT_AG_pretrain_randperm_03 ag mytest -m aws -p 30 -f 0
#python runbenchmark.py FTT_AG_pretrain_randperm_06 ag mytest -m aws -p 30 -f 0
#python runbenchmark.py FTT_AG_pretrain_randperm_09 ag mytest -m aws -p 30 -f 0
#python runbenchmark.py FTT_AG_pretrain_randblk_03 ag mytest -m aws -p 30 -f 0
#python runbenchmark.py FTT_AG_pretrain_randblk_06 ag mytest -m aws -p 30 -f 0
#python runbenchmark.py FTT_AG_pretrain_randblk_09 ag mytest -m aws -p 30 -f 0


#python runbenchmark.py FTTransformer_gpu_1 ag mytest -m aws -p 150 -f 0
#python runbenchmark.py FTTransformer_gpu_pretrain_1 ag mytest -m aws -p 150 -f 0
#
#python runbenchmark.py FTTransformer_gpu_3 ag mytest -m aws -p 150 -f 0
#python runbenchmark.py FTTransformer_gpu_pretrain_3 ag mytest -m aws -p 150 -f 0
#
#python runbenchmark.py FTTransformer_gpu_5 ag mytest -m aws -p 150 -f 0
#python runbenchmark.py FTTransformer_gpu_pretrain_5 ag mytest -m aws -p 150 -f 0
#
#python runbenchmark.py WideDeep ag mytest -m aws -p 150 -f 0
#python runbenchmark.py WideDeep_pretrain ag mytest -m aws -p 150 -f 0