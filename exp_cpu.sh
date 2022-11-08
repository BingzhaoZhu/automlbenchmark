#!/usr/bin/env bash

rm -f ~/.config/automlbenchmark/config.yaml
cp ./examples/config_m6i.yaml ~/.config/automlbenchmark/config.yaml

python runbenchmark.py Ensemble_AG_mq adult mytest -m aws -p 1040
