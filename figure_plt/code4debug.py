"""
Example script to predict the vital status of patients with Head-Neck Squamous Cell Carcinoma.
Dataset is originally from https://portal.gdc.cancer.gov/projects/TCGA-HNSC.
Paper working on similar task: https://bmcbioinformatics.biomedcentral.com/track/pdf/10.1186/s12859-019-2929-8.pdf
"""

import pandas as pd
import numpy as np
import argparse
import os
import random
from autogluon.tabular import TabularPredictor, TabularDataset
from autogluon.tabular.configs.hyperparameter_configs import get_hyperparameter_config
import torch as th
from sklearn.model_selection import train_test_split
try:
    from autogluon.multimodal.utils import download
except:
    from autogluon.multimodal.utils.download import download

import warnings
warnings.filterwarnings('ignore')


# Dataset information for TCGA dataset
INFO = {
    "name": "cancer_survival.tsv",
    "url": "s3://automl-mm-bench/life-science/clinical.tsv",
    "sha1sum": "6d19609c2a8492f767efd9f2c0b7687bcd3845a3"
}


def get_parser():
    parser = argparse.ArgumentParser(
        description='The Basic Example of AutoGluon for TCGA dataset.')
    parser.add_argument('--path', default='./dataset')
    parser.add_argument('--test_size', default=0.3)
    parser.add_argument('--shuffle', default=True)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--task', choices=['TCGA_HNSC', 'adult'],
                        default='adult')
    parser.add_argument('--mode', choices=['FT_Transformer', 'all_models'],
                        default='FT_Transformer')
    parser.add_argument('--num_gpus', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=8)
    return parser


def data_loader(path="./dataset/", ):
    name = INFO["name"]
    full_path = os.path.join(path, name)
    if os.path.exists(full_path):
        print(f"Existing dataset: {name}")
    else:
        print(f"Dataset not exist. Start downloading: {name}")
        download(INFO["url"], path=full_path, sha1_hash=INFO["sha1sum"])
    df = pd.read_csv(full_path, sep='\t')
    return df


# Preprocessing steps include:
# (1) Remove "id"-related columns and columns with the same values;
# (2) Some column shared common information with the target label. Those "shortcuts" were removed.
#      e.g. We aim to predict whether patents are alive or not. The "death_date" is an invalid feature.
# (3) Split data into train/test sets, by specifying "test_size" and "shuffle".
def preprocess(df, test_size, shuffle):
    N, _ = df.shape

    df = df[df != "'--"]  # Replace missing entries with nan
    n_unique = df.nunique()

    for col, n in n_unique.items():
        if "id" in col or n <= 1:
            df.drop(col, axis=1, inplace=True)

    shortcut_col = ["days_to_death", "year_of_death"] # Shortcut columns should be removed
    for col in shortcut_col:
        df.drop(col, axis=1, inplace=True)

    df_train, df_test = train_test_split(df, test_size=test_size, shuffle=shuffle)
    return df_train, df_test


def train(args):
    if args.task == "adult":
        df_train = TabularDataset('https://autogluon.s3.amazonaws.com/datasets/Inc/train.csv')
        df_test = TabularDataset('https://autogluon.s3.amazonaws.com/datasets/Inc/test.csv')
        label = "class"
    elif args.task == "TCGA_HNSC":
        df = data_loader(args.path)
        df_train, df_test = preprocess(df, args.test_size, args.shuffle)
        label = "vital_status"
    else:
        raise NotImplementedError

    metric = "accuracy"
    hyperparameters = {} # get_hyperparameter_config('default')
    hyperparameters['FT_TRANSFORMER'] = {
                                         "env.per_gpu_batch_size": 128,
                                         # "model.numerical_transformer.embedding_arch": ["sigmoid"],  #positional
                                         # "model.fusion_transformer.additive_attention": "auto",
                                         # "model.fusion_transformer.share_qv_weights": None,
                                         # "knn_dataloader": False,
                                         # "model.fusion_transformer.row_attention": True,
                                         # "env.eval_batch_size_ratio": 1,
                                         "env.num_workers": 0,
                                         "env.num_workers_evaluation": 0,
                                         "optimization.max_steps": 100,
                                         "optimization.val_check_interval": 100,

                                         # "model.fusion_transformer.row_attention": True,
                                         # "model.fusion_transformer.row_attention_layer": "last",
                                         # "model.fusion_transformer.global_token": True,
                                         # "optimization.row_attention_weight_decay": 0.1,
                                         # "env.test_ensemble_rounds": 10,

                                         # "pretrainer": True,
                                         # "pretrainer.augmentation_type": "permutation",
                                         # "pretrainer.corruption_rate": 0.6,
                                         # "pretrainer.pretrain_epochs": 5,
                                         # "pretrainer.objective": "reconstruction",
                                         # "pretrainer.start_pretrain_coefficient": 1,
                                         # "pretrainer.end_pretrain_coefficient": 0.1,
                                         # "pretrainer.decay_pretrain_coefficient": 0.6,
                                         # "pretrainer.temperature": 1,


                                         # "pretrainer": True,
                                         # "pretrainer.augmentation_type": "random_perm",
                                         # "pretrainer.corruption_rate": 0.6,
                                         # "pretrainer.pretrain_epochs": 0,
                                         # "pretrainer.loss_mixup": "self_distill",
                                         # "pretrainer.start_loss_coefficient": 1,
                                         # "pretrainer.end_loss_coefficient": 1,
                                         }
    # hyperparameters = {}
    # hyperparameters['AG_AUTOMM'] = {
    #     "env.num_gpus": 1,
    #     "env.per_gpu_batch_size": 128,
    #     "model.names": ["categorical_mlp", "numerical_mlp", "fusion_mlp"],
    #     "env.batch_size": 128,
    #     "env.num_workers": 0,
    #     "env.num_workers_evaluation": 0,
    #     "env.eval_batch_size_ratio": 1,
    #     "optimization.max_epochs": 2000,  # Specify a large value to train until convergence
    #     "optimization.weight_decay": 1.0e-5,
    #     "optimization.lr_choice": None,
    #     "optimization.lr_schedule": "polynomial_decay",
    #     "optimization.warmup_steps": 0.0,
    #     "optimization.patience": 20,
    #     "optimization.top_k": 3,
    #     "data.categorical.convert_to_text": False,
    # }
    # hyperparameters['FT_TRANSFORMER'] = [{"env.num_gpus": 1,
    #                                      "env.per_gpu_batch_size": 128,
    #                                      "env.eval_batch_size_ratio": 1,
    #                                      "env.num_workers": 0,
    #                                      "env.num_workers_evaluation": 0,
    #                                      }, {"env.num_gpus": 1,
    #                                      "env.per_gpu_batch_size": 128,
    #                                      "env.eval_batch_size_ratio": 1,
    #                                      "env.num_workers": 0,
    #                                      "env.num_workers_evaluation": 0,}]
    predictor = TabularPredictor(label=label,
                                 eval_metric="roc_auc",
                                 # problem_type="regression",
                                 )
    # predictor = TabularPredictor(label="age",
    #                              eval_metric="mse",
    #                              problem_type="regression",
    #                              )
    df_train = df_train.dropna(subset=[label])
    df_test = df_test.dropna(subset=[label])

    predictor.fit(
        train_data=df_train,
        hyperparameters=hyperparameters,
        # presets="best_quality",
        time_limit=3600,
        # is_pretrain={"is_pretrain": True,
        #              "name": "adult",
        #              "num_tasks": 2,
        #              "iter_per_save": 100,
        #              "max_iter": 200,
        #              "upload_per_n_iter": 5,
        #              },
        is_pretrain={"is_pretrain": False,
                     "finetune_on": "iter_0/pretrained.ckpt",
                     },
    )

    probabilities = predictor.predict(df_test, as_pandas=True) #, support_data=df_train.drop(label, axis=1))
    # predictions = probabilities.idxmax(axis=1).to_numpy()
    # from sklearn.metrics import roc_auc_score, accuracy_score
    # print("guaguaguagua", accuracy_score(df_test[label].to_numpy(), predictions))
    leaderboard = predictor.leaderboard(df_test)
    leaderboard.to_csv("./leaderboard.csv")
    return


def set_seed(seed):
    th.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    set_seed(args.seed)
    train(args)