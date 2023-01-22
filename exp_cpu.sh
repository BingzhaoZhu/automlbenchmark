#!/usr/bin/env bash

# rm -f ~/.config/automlbenchmark/config.yaml
# cp ./examples/config_m6i.yaml ~/.config/automlbenchmark/config.yaml

python runbenchmark.py RF_HPO ag mytest1h -m aws -p 1040
# python runbenchmark.py GBM_HPO ag mytest1h -m aws -p 1040

# python runbenchmark.py RF_AG ag mytest1h -m aws -p 1040
# python runbenchmark.py GBM_AG ag mytest1h -m aws -p 1040
# python runbenchmark.py CAT_AG ag mytest1h -m aws -p 1040
# python runbenchmark.py XGB_AG ag mytest1h -m aws -p 1040
# python runbenchmark.py NN_AG ag mytest1h -m aws -p 1040
# python runbenchmark.py FASTAI_AG ag mytest1h -m aws -p 1040

