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
import numpy as np
from scipy.special import softmax

import torch
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pyarrow.parquet as pq
from pytorch_widedeep import Trainer
from pytorch_widedeep.models import WideDeep, FTTransformer
from pytorch_widedeep.metrics import Accuracy
from pytorch_widedeep.datasets import load_adult
from pytorch_widedeep.preprocessing import TabPreprocessor
from pytorch_widedeep.self_supervised_training import (
    ContrastiveDenoisingTrainer,
)

log = logging.getLogger(__name__)


def run(dataset, config):
    log.info(f"\n**** WideDeep ****\n")

    is_classification = config.type == 'classification'
    training_params = {k: v for k, v in config.framework_params.items() if not k.startswith('_')}

    train_path, test_path = dataset.train.path, dataset.test.path
    target_col = dataset.target.name
    train_data = pq.ParquetDataset(train_path).read().to_pandas()
    test_data = pq.ParquetDataset(test_path).read().to_pandas()
    cat_embed_cols, continuous_cols = is_categorical(train_data)

    tab_preprocessor = TabPreprocessor(
        cat_embed_cols=cat_embed_cols,
        continuous_cols=continuous_cols,
        with_attention=True,
        with_cls_token=True,  # this is optional
    )
    X_tab = tab_preprocessor.fit_transform(train_data)
    target = train_data[target_col].values

    log.info("Running WideDeep with a maximum time of {}s.".format(config.max_runtime_seconds))
    ft_transformer = FTTransformer(
        column_idx=tab_preprocessor.column_idx,
        cat_embed_input=tab_preprocessor.cat_embed_input,
        continuous_cols=tab_preprocessor.continuous_cols,
        input_dim=192,
        n_blocks=3,
        n_heads=8,
    )
    contrastive_denoising_trainer = ContrastiveDenoisingTrainer(
        model=ft_transformer,
        preprocessor=tab_preprocessor,
    )
    contrastive_denoising_trainer.pretrain(X_tab, n_epochs=5, batch_size=256)

    pretrained_model = contrastive_denoising_trainer.model
    model = WideDeep(deeptabular=pretrained_model)
    trainer = Trainer(model=model, objective="binary", metrics=[Accuracy])

    with Timer() as training:
        trainer.fit(X_tab=X_tab, target=target, n_epochs=5, batch_size=256)

    # get a test metric
    X_tab_te = tab_preprocessor.transform(test_data)
    y_test = test_data[test_data].values

    with Timer() as predict:
        predictions = trainer.predict(X_tab=X_tab_te)
    probabilities = trainer.predict_proba(X_tab=X_tab_te) if is_classification else None

    return result(output_file=config.output_predictions_file,
                  predictions=predictions,
                  probabilities=probabilities,
                  truth=y_test,
                  target_is_encoded=False,
                  models_count=1,
                  training_duration=training.duration,
                  predict_duration=predict.duration)


def is_categorical(df):
    categorical_indicator = []
    numerical_indicator = []
    for c in df.columns:
        if pd.api.types.is_categorical_dtype(df[c]):
            categorical_indicator.append(c)
        else:
            numerical_indicator.append(c)
    return categorical_indicator, numerical_indicator


if __name__ == '__main__':
    call_run(run)
