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
import transtab
from torch import nn
from sklearn.preprocessing import LabelEncoder
import openml

from frameworks.shared.callee import call_run, result, output_subdir
from frameworks.shared.utils import Timer, zip_path
import numpy as np
from scipy.special import softmax


log = logging.getLogger(__name__)

def is_categorical(df):
    categorical_indicator = []
    for c in df.columns:
        categorical_indicator.append(pd.api.types.is_categorical_dtype(df[c]))
    return categorical_indicator

def gen_config(df):
    config = {}
    X = df.drop(['target_label'],axis=1)
    drop_cols = [col for col in X if X[col].nunique()<=1]
    all_cols = np.array([col.lower() for col in X.columns.tolist()])

    categorical_indicator = np.array(is_categorical(X))
    cat_cols = [col for col in all_cols[categorical_indicator] if col not in drop_cols]
    num_cols = [col for col in all_cols[~categorical_indicator] if col not in drop_cols]
    all_cols = [col for col in all_cols if col not in drop_cols]
    
    bin_cols = []
    cat_cols = [c for c in cat_cols if c not in bin_cols]

    config["./train"] = {'bin': bin_cols, "cat": cat_cols, "num": num_cols}
    config["./test"] = {'bin': bin_cols, "cat": cat_cols, "num": num_cols}

    return config


def run(dataset, config):
    log.info(f"\n**** TransTab ****\n")

    if not os.path.exists("./train"):
        os.mkdir("./train")
    if not os.path.exists("./test"):
        os.mkdir("./test")
    
    df = pd.read_parquet(dataset.train.path)
    df = df.rename(columns={dataset.target.name: "target_label"})

    dataset_config = gen_config(df)
    df.to_csv('./train/data_processed.csv')
    df = pd.read_parquet(dataset.test.path)
    df = df.rename(columns={dataset.target.name: "target_label"})
    df.to_csv('./test/data_processed.csv')

    trainset, _, _, _, cat_cols, num_cols, bin_cols = transtab.load_data('./train', dataset_config)
    testset, _, _, _, _, _, _ = transtab.load_data('./test', dataset_config)

    training_arguments = {
        'num_epoch':50,
        'batch_size':128,
        'lr':1e-4,
        'eval_metric':'val_loss',
        'eval_less_is_better':True,
        'output_dir':'./checkpoint'
    }
     
    is_classification = config.type == 'classification'

    X, y = trainset
    num_classes = len(np.unique(y.values)) if is_classification else 1
    l_enc = LabelEncoder()
    X, y = trainset
    y = l_enc.fit_transform(y.values)
    y = pd.Series(y,index=X.index)

    val_size = int(len(y)*0.125)
    trainset = (X.iloc[:-val_size], y[:-val_size])
    valset = (X.iloc[-val_size:], y[-val_size:])

    X, y = testset
    y = l_enc.transform(y.values)
    y = pd.Series(y,index=X.index)
    testset = (X, y)

    # contrastive pretraining
    # allset_, trainset_, valset_, testset_, cat_cols_, num_cols_, bin_cols_ = transtab.load_data(['credit-g','credit-approval'])
    # model, collate_fn = transtab.build_contrastive_learner(
    #     cat_cols_, num_cols_, bin_cols_, 
    #     supervised=True, # if take supervised CL
    #     num_partition=4, # num of column partitions for pos/neg sampling
    #     overlap_ratio=0.5, # specify the overlap ratio of column partitions during the CL
    # )
    # transtab.train(model, trainset_, valset_, collate_fn=collate_fn, **training_arguments)

    model, collate_fn = transtab.build_contrastive_learner(
        cat_cols, num_cols, bin_cols, 
        supervised=True, # if take supervised CL
        num_partition=4, # num of column partitions for pos/neg sampling
        overlap_ratio=0.5, # specify the overlap ratio of column partitions during the CL
    )
    transtab.train(model, trainset, valset, collate_fn=collate_fn, **training_arguments)
    model = transtab.build_classifier(checkpoint='./checkpoint')
    model.update({'cat':cat_cols,'num':num_cols,'bin':bin_cols, 'num_class':num_classes})

    model.num_class = num_classes
    if model.num_class > 2:
        model.loss_fn = nn.CrossEntropyLoss(reduction='none')
    else:
        model.loss_fn = nn.BCEWithLogitsLoss(reduction='none')

    ## regular training
    # model = transtab.build_classifier(cat_cols, num_cols, bin_cols, num_class=num_classes)

    # n_epoch = config.framework_params.get('_n_epoch', 100)
    # n_pretrain_epoch = config.framework_params.get('_n_pretrain_epoch', 0)
    # patience = config.framework_params.get('_patience', 10)
    # device = config.framework_params.get('_device', 'cuda')
    # batch_size = config.framework_params.get('_batch_size', 64)
    # pretrain_batch_size = config.framework_params.get('_pretrain_batch_size', 256)
    # training_params = {k: v for k, v in config.framework_params.items() if not k.startswith('_')}

    with Timer() as training:
        transtab.train(model, trainset, valset, **training_arguments)

    with Timer() as predict:
        x_test, y_test = testset
        probabilities = transtab.predict(model, x_test)
        # print(probabilities)
        if len(probabilities.shape) == 1:
            probabilities = np.concatenate((1-probabilities[:, None], probabilities[:, None]), axis=1)

    if config.type == 'classification':
        predictions = np.argmax(probabilities, axis=1)
        
        predictions = l_enc.inverse_transform(predictions.astype(int))
        y_test = l_enc.inverse_transform(y_test.astype(int))
    else:
        predictions = None
    

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
