#!/usr/bin/env bash

rm -f ~/.config/automlbenchmark/config.yaml
cp ./examples/config_m6i.yaml ~/.config/automlbenchmark/config.yaml

python runbenchmark.py Ensemble_AG_FTT_all_bq_cpu ag mytest24h -m aws -p 1040
