import logging
import os
import shutil
import warnings
import sys
import tempfile
warnings.simplefilter("ignore")

if sys.platform == 'darwin':
    os.environ['OMP_NUM_THREADS'] = '1'

import matplotlib
import pandas as pd
matplotlib.use('agg')  # no need for tk

from autogluon.tabular import TabularPredictor, TabularDataset
from autogluon.core.utils.savers import save_pd, save_pkl
import autogluon.core.metrics as metrics
from autogluon.tabular.version import __version__

from frameworks.shared.callee import call_run, result, output_subdir
from frameworks.shared.utils import Timer, zip_path
from models.FTTransformer import FTTransformer

log = logging.getLogger(__name__)


def run(dataset, config):
    log.info(f"\n**** FTTransformer [v{__version__}] ****\n")

    is_classification = config.type == 'classification'
    training_params = {k: v for k, v in config.framework_params.items() if not k.startswith('_')}




    log.info("Running FTTransformer with a maximum time of {}s.".format(config.max_runtime_seconds))
    ftt = FTTransformer(**training_params)

    with Timer() as training:
        # ftt.fit(loader_train, epoch=1)
        print(ftt.device)

    with Timer() as predict:
        predictions = None #ftt.predict(loader_test)
    probabilities = None #rf.predict_proba(X_test) if is_classification else None

    return result(output_file=config.output_predictions_file,
                  predictions=predictions,
                  probabilities=probabilities,
                  target_is_encoded=False,
                  models_count=1,
                  training_duration=training.duration,
                  predict_duration=predict.duration)


if __name__ == '__main__':
    call_run(run)
