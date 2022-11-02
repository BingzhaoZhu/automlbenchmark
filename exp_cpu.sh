#!/usr/bin/env bash

rm -f ~/.config/automlbenchmark/config.yaml
cp ./examples/config_cpu.yaml ~/.config/automlbenchmark/config.yaml

python runbenchmark.py Ensemble_AG_bq ag mytest4h -m aws -p 1040
# python runbenchmark.py Ensemble_AG_mq ag mytest4h -m aws -p 1040
