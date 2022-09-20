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

import pandas as pd

from frameworks.shared.callee import call_run, result, output_subdir
from frameworks.shared.utils import Timer, zip_path
import numpy as np
from scipy.special import softmax

import torch
from sklearn.preprocessing import LabelEncoder, StandardScaler
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
    y_train, y_test = train_data[target_col], test_data[target_col]
    X_train = train_data.loc[:, train_data.columns != target_col]
    X_test = test_data.loc[:, test_data.columns != target_col]
    cat_embed_cols, continuous_cols = is_categorical(X_train)
    X_train, X_test = handle_missing_value(X_train, X_test)

    tab_preprocessor = TabPreprocessor(
        cat_embed_cols=cat_embed_cols,
        continuous_cols=continuous_cols,
        with_attention=True,
        with_cls_token=True,  # this is optional
    )
    X_train = tab_preprocessor.fit_transform(X_train)

    if is_classification:
        l_enc = LabelEncoder()
        y_train = l_enc.fit_transform(y_train)
    else:
        y_train, y_test = y_train.astype('float32'), y_test.astype('float32')
        l_enc = StandardScaler()
        y_train = l_enc.fit_transform(y_train[:, None])


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
    contrastive_denoising_trainer.pretrain(X_train, n_epochs=5, batch_size=256)
    contrastive_denoising_trainer.save(
        path="pretrained", model_filename="contrastive_denoising_model.pt"
    )
    contrastive_denoising_model = torch.load(
        "pretrained/contrastive_denoising_model.pt"
    )
    pred_dim = len(l_enc.classes_) if config.type_ == "multiclass" else 1

    pretrained_model = contrastive_denoising_model.model
    model = WideDeep(deeptabular=pretrained_model, pred_dim=pred_dim)
    trainer = Trainer(model=model, objective=config.type_)

    with Timer() as training:
        trainer.fit(X_tab=X_train, target=y_train, n_epochs=5, batch_size=256)

    # get a test metric
    X_test = tab_preprocessor.transform(X_test)
    print(X_test)

    with Timer() as predict:
        predictions = trainer.predict(X_tab=X_test)
    predictions = l_enc.inverse_transform(predictions[:, None])
    probabilities = trainer.predict_proba(X_tab=X_test) if is_classification else None

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
    categorical_indicator = None if len(categorical_indicator) == 0 else categorical_indicator
    numerical_indicator = None if len(numerical_indicator) == 0 else numerical_indicator
    return categorical_indicator, numerical_indicator


def handle_missing_value(X_train, X_test):
    n_train = X_train.shape[0]
    X = pd.concat((X_train, X_test), axis=0)
    for col in X.columns:
        if pd.api.types.is_categorical_dtype(X[col]):
            X[col] = X[col].astype("object")
            X[col].fillna("MissingValue", inplace=True)
            l_enc = LabelEncoder()
            X[col] = l_enc.fit_transform(X[col].values)
        else:
            X[col].fillna(X_train.loc[:, col].mean(), inplace=True)
    X_train, X_test = X.iloc[:n_train, :], X.iloc[n_train:, :]
    return X_train, X_test


if __name__ == '__main__':
    call_run(run)
