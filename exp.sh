#!/usr/bin/env bash

rm -f ~/.config/automlbenchmark/config.yaml
cp ./examples/config_gpu.yaml ~/.config/automlbenchmark/config.yaml

python runbenchmark.py Ensemble_AG_FastFTT_bq ag mytest4h -m aws -p 180