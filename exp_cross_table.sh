#!/usr/bin/env bash

rm -f ~/.config/automlbenchmark/config.yaml
cp ./examples/config_gpu.yaml ~/.config/automlbenchmark/config.yaml

python cross_table_start.py
python runbenchmark.py FTT_AG ag mytest1h -m aws -p 104 -f 0
python cross_backbone_update.py
python runbenchmark.py FTT_AG ag mytest1h -m aws -p 104 -f 0
python cross_backbone_update.py
python runbenchmark.py FTT_AG ag mytest1h -m aws -p 104 -f 0
