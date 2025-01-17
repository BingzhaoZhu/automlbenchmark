import copy
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


from frameworks.shared.callee import call_run, result, output_subdir
from frameworks.shared.utils import Timer, zip_path
from ftt.FTTransformer import FTTransformer
from ftt.DataLoader import get_torch_dataloader
import numpy as np
from scipy.special import softmax


log = logging.getLogger(__name__)


def run(dataset, config):
    log.info(f"\n**** FTTransformer ****\n")

    is_classification = config.type == 'classification'
    n_epoch = config.framework_params.get('_n_epoch', 100)
    n_pretrain_epoch = config.framework_params.get('_n_pretrain_epoch', 0)
    patience = config.framework_params.get('_patience', 10)
    device = config.framework_params.get('_device', 'cuda')
    batch_size = config.framework_params.get('_batch_size', 64)
    pretrain_batch_size = config.framework_params.get('_pretrain_batch_size', 256)
    training_params = {k: v for k, v in config.framework_params.items() if not k.startswith('_')}

    dl_train_pretrain, _, _, _ = get_torch_dataloader(dataset, is_classification, batch_size=pretrain_batch_size)
    dl_train, dl_valid, dl_test, info = get_torch_dataloader(dataset, is_classification, batch_size=batch_size)
    n_cat, n_con = dl_train.dataset.X1.shape[1], dl_train.dataset.X2.shape[1]

    cat_dims = info["cat_dims"]
    num_classes = len(np.unique(dl_train.dataset.y)) if is_classification else 1

    log.info("Running FTTransformer with a maximum time of {}s.".format(config.max_runtime_seconds))


    ftt_ = FTTransformer(
        cat_dims=cat_dims,
        n_con=n_con,
        num_classes=num_classes,
        is_classification=is_classification,
        device=device,
        **training_params)
    best_model = copy.deepcopy(ftt_)

    for epoch in range(1, 1 + n_pretrain_epoch):
        ftt_.pretrain(dl_train_pretrain, epoch)

    with Timer() as training:
        trigger_times, best_loss = 0, float('inf')
        for epoch in range(1, n_epoch+1):
            train_loss = ftt_.fit(dl_train, epoch=epoch)
            valid_loss = ftt_.validate(dl_valid)
            if valid_loss > best_loss and patience != -1:
                trigger_times += 1
                if trigger_times >= patience:
                    print(f"Validation loss not improving, training stopped at {epoch} epoch, retrieving the best model.")
                    break
            else:
                best_loss = valid_loss
                trigger_times = 0
                best_model = copy.deepcopy(ftt_)

    ftt_ = best_model

    with Timer() as predict:
        yhat, y_test = ftt_.predict(dl_test)

    l_enc = info["label_encoder"]
    predictions = np.argmax(yhat, axis=1) if is_classification else yhat
    predictions = l_enc.inverse_transform(predictions)
    y_test = l_enc.inverse_transform(y_test)
    probabilities = softmax(yhat, axis=1) if is_classification else None

    return result(output_file=config.output_predictions_file,
                  predictions=predictions,
                  probabilities=probabilities,
                  truth=y_test,
                  target_is_encoded=False,
                  models_count=1,
                  training_duration=training.duration,
                  predict_duration=predict.duration)


if __name__ == '__main__':
    call_run(run)
