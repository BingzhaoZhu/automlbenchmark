#!/usr/bin/env bash

rm -f ~/.config/automlbenchmark/config.yaml
cp ./examples/config_gpu.yaml ~/.config/automlbenchmark/config.yaml

python runbenchmark.py Ensemble_AG_FTT_all_mq ag mytest4h -m aws -p 520
# python runbenchmark.py Ensemble_AG_FTT_all_bq ag mytest4h -m aws -p 520
# python runbenchmark.py Ensemble_AG_FTT_pretrain_bq ag mytest4h -m aws -p 520