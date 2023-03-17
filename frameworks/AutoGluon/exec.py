import copy
import logging
import os
import shutil
import warnings
import sys
import tempfile

import numpy as np

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
from autogluon.core.space import Categorical, Int, Real
import math

from frameworks.shared.callee import call_run, result, output_subdir
from frameworks.shared.utils import Timer, zip_path

# this is needed to avoid issues with multiprocessing
# https://github.com/pytorch/pytorch/issues/11201
# import torch.multiprocessing
# torch.multiprocessing.set_sharing_strategy('file_system')

log = logging.getLogger(__name__)

def hyperparameter_search_space(training_params):
    if "HPO" in training_params:
        if training_params.pop("HPO"):
            if "XGB" in training_params["hyperparameters"]:
                search_space = {
                    'learning_rate': Real(lower=math.exp(-7), upper=1, default=0.1, log=True),
                    'max_depth': Int(lower=1, upper=10, default=6),
                    'subsample': Real(lower=0.2, upper=1.0, default=1.0),
                    'colsample_bytree': Real(lower=0.2, upper=1.0, default=1.0),
                    'colsample_bylevel': Real(lower=0.2, upper=1.0, default=1.0),
                    'min_child_weight': Real(lower=math.exp(-16), upper=math.exp(5), default=1, log=True),
                    'reg_alpha': Real(lower=math.exp(-16), upper=math.exp(2), default=1, log=True),
                    'reg_lambda': Real(lower=math.exp(-16), upper=math.exp(2), default=1, log=True),
                    'gamma': Real(lower=math.exp(-16), upper=math.exp(2), default=1, log=True),
                    'n_estimators': Int(lower=100, upper=4000, default=600),
                }
                training_params["hyperparameters"]["XGB"] = search_space
            
            if "GBM" in training_params["hyperparameters"]:
                search_space = {
                    'num_leaves': Int(lower=5, upper=50, default=31),
                    'max_depth': Int(lower=3, upper=20),
                    'learning_rate': Real(lower=math.exp(-3), upper=1, log=True),
                    'n_estimators': Int(lower=50, upper=2000),
                    'min_child_weight': Real(lower=math.exp(-5), upper=math.exp(4), log=True),
                    'reg_alpha': Categorical(0, 0.1, 1, 2, 5, 7, 10, 50 , 100),
                    'reg_lambda': Categorical(0, 0.1, 1, 5, 10, 20, 50, 100),
                    'subsample': Real(lower=0.2, upper=0.8),
                }
                training_params["hyperparameters"]["GBM"] = search_space
            
            if "CAT" in training_params["hyperparameters"]:
                search_space = {
                    'learning_rate': Real(lower=math.exp(-5), upper=1, log=True),
                    'random_strength': Int(lower=1, upper=20, default=31),
                    'l2_leaf_reg': Real(lower=1, upper=10, log=True),
                    'bagging_temperature': Real(lower=0, upper=1),
                    'leaf_estimation_iterations': Int(lower=1, upper=20, default=31),
                    'iterations': Int(lower=100, upper=4000),
                }
                training_params["hyperparameters"]["CAT"] = search_space

            if "RF" in training_params["hyperparameters"]:
                search_space = {
                    'n_estimators': Int(lower=10, upper=1000, default=300),
                    'max_features': Categorical('auto', 0.5, 0.25),
                    'max_leaf_nodes': Int(lower=100, upper=4000),
                }
                training_params["hyperparameters"]["RF"] = search_space

            if "FTT" in training_params["hyperparameters"]:
                training_params["hyperparameters"].pop("FTT")
                training_params["hyperparameters"]['FT_TRANSFORMER'] = {
                    "env.per_gpu_batch_size": 128,
                    "pretrainer": True,
                    "pretrainer.start_pretrain_coefficient": 0,
                    "pretrainer.end_pretrain_coefficient": 0,
                    "pretrainer.pretrain_epochs": 0,

                    "optimization.patience": 20,
                    "env.batch_size": Categorical(128, 32, 8, 1),
                    "optimization.top_k": Categorical(3, 5, 1),
                    "optimization.val_check_interval": Categorical(0.5, 1.0),
                    "finetune_on": "pretrain_reconstruction_1_sum/iter_0/pretrained.ckpt",
                }

            if "FTT_fold1_pretrained" in training_params["hyperparameters"]:
                training_params["hyperparameters"].pop("FTT_fold1_pretrained")
                training_params["hyperparameters"]['FT_TRANSFORMER'] = {
                    "env.per_gpu_batch_size": 128,
                    "pretrainer": True,
                    "pretrainer.start_pretrain_coefficient": 0,
                    "pretrainer.end_pretrain_coefficient": 0,
                    "pretrainer.pretrain_epochs": 0,

                    "optimization.patience": 20,
                    "env.batch_size": Categorical(128, 32, 8, 1),
                    "optimization.top_k": Categorical(3, 5, 1),
                    "optimization.val_check_interval": Categorical(0.5, 1.0),
                    "finetune_on": Categorical(
                        "pretrain_reconstruction_1_sum/iter_2000/pretrained.ckpt", 
                        "pretrain_reconstruction_1_sum/iter_0/pretrained.ckpt", 
                        "pretrain_reconstruction_1_sum/iter_250/pretrained.ckpt", 
                        "pretrain_reconstruction_1_sum/iter_1000/pretrained.ckpt"),
                }

            if "FTT_fold2_pretrained" in training_params["hyperparameters"]:
                training_params["hyperparameters"].pop("FTT_fold2_pretrained")
                training_params["hyperparameters"]['FT_TRANSFORMER'] = {
                    "env.per_gpu_batch_size": 128,
                    "pretrainer": True,
                    "pretrainer.start_pretrain_coefficient": 0,
                    "pretrainer.end_pretrain_coefficient": 0,
                    "pretrainer.pretrain_epochs": 0,

                    "optimization.patience": 20,
                    "env.batch_size": Categorical(128, 32, 8, 1),
                    "optimization.top_k": Categorical(3, 5, 1),
                    "optimization.val_check_interval": Categorical(0.5, 1.0),
                    "finetune_on": Categorical(
                        "pretrain_reconstruction_1_sum_fold_2/iter_2000/pretrained.ckpt", 
                        "pretrain_reconstruction_1_sum_fold_2/iter_0/pretrained.ckpt", 
                        "pretrain_reconstruction_1_sum_fold_2/iter_250/pretrained.ckpt", 
                        "pretrain_reconstruction_1_sum_fold_2/iter_1000/pretrained.ckpt"),
                }

            if "RF" in training_params["hyperparameters"] or "XGB" in training_params["hyperparameters"] or "GBM" in training_params["hyperparameters"] or "CAT" in training_params["hyperparameters"]:
                training_params["hyperparameter_tune_kwargs"] = {'num_trials': 100,  'searcher': 'random', "scheduler": "local"}
            else:
                training_params["hyperparameter_tune_kwargs"] = {'num_trials': 30,  'searcher': 'random', "scheduler": "local"}
            training_params["keep_only_best"] = True
            training_params["fit_weighted_ensemble"] = False

    return training_params

def run(dataset, config):
    log.info(f"\n**** AutoGluon [v{__version__}] ****\n")
    print("__dataset", dataset)
    print("__config",  config)

    metrics_mapping = dict(
        acc=metrics.accuracy,
        auc=metrics.roc_auc,
        f1=metrics.f1,
        logloss=metrics.log_loss,
        mae=metrics.mean_absolute_error,
        mse=metrics.mean_squared_error,
        r2=metrics.r2,
        rmse=metrics.root_mean_squared_error,
    )

    perf_metric = metrics_mapping[config.metric] if config.metric in metrics_mapping else None
    if perf_metric is None:
        # TODO: figure out if we are going to blindly pass metrics through, or if we use a strict mapping
        log.warning("Performance metric %s not supported.", config.metric)

    is_classification = config.type == 'classification'
    training_params = {k: v for k, v in config.framework_params.items() if not k.startswith('_')}
    infer_rounds = config.framework_params.get('_infer_rounds', 1)

    train_path, test_path = dataset.train.path, dataset.test.path
    label = dataset.target.name
    problem_type = dataset.problem_type

    models_dir = tempfile.mkdtemp() + os.sep  # passed to AG

    if "is_pretrain" in training_params and "name" in training_params["is_pretrain"]:
        training_params["is_pretrain"]["name"] = config.name

    training_params = hyperparameter_search_space(training_params)
    test_data = TabularDataset(test_path)
    train_data = TabularDataset(train_path)
    if "is_pretrain" in training_params and "seed" in training_params["is_pretrain"]:
        seed = training_params["is_pretrain"]["seed"]
    else:
        seed = 0
    train_data = train_data.sample(frac=1.0, axis=1, random_state=seed)
    test_data = test_data.reindex(columns = train_data.columns)

    with Timer() as training:
        predictor = TabularPredictor(
            label=label,
            eval_metric=perf_metric.name,
            path=models_dir,
            problem_type=problem_type,
        ).fit(
            train_data=train_data,
            time_limit=config.max_runtime_seconds,
            **training_params
        )

    # Persist model in memory that is going to be predicting to get correct inference latency
    predictor.persist_models('best')

    if is_classification:
        with Timer() as predict:
            probabilities = []
            for _ in range(abs(infer_rounds)):
                if infer_rounds > 0:
                    proba_per_round = predictor.predict_proba(test_data, as_multiclass=True)
                else:
                    proba_per_round = predictor.predict_proba(test_data,
                                                              as_multiclass=True,
                                                              support_data=train_data.drop(label, axis=1))
                probabilities.append(proba_per_round)
            probabilities = pd.concat(probabilities)
            probabilities = probabilities.groupby(probabilities.index).median()
            probabilities = probabilities.reindex_like(proba_per_round)
        predictions = proba_per_round.idxmax(axis=1).to_numpy()
    else:
        with Timer() as predict:
            predictions = []
            for _ in range(abs(infer_rounds)):
                if infer_rounds > 0:
                    pred_per_round = predictor.predict(test_data, as_pandas=False)
                else:
                    pred_per_round = predictor.predict(test_data,
                                                       as_pandas=False,
                                                       support_data=train_data.drop(label, axis=1))
                predictions.append(pred_per_round)
            predictions = np.median(predictions, axis=0)
        probabilities = None

    prob_labels = probabilities.columns.values.astype(str).tolist() if probabilities is not None else None

    _leaderboard_extra_info = config.framework_params.get('_leaderboard_extra_info', False)  # whether to get extra model info (very verbose)
    _leaderboard_test = config.framework_params.get('_leaderboard_test', False)  # whether to compute test scores in leaderboard (expensive)
    leaderboard_kwargs = dict(silent=True, extra_info=_leaderboard_extra_info)
    # Disabled leaderboard test data input by default to avoid long running computation, remove 7200s timeout limitation to re-enable
    if _leaderboard_test:
        leaderboard_kwargs['data'] = test_data

    leaderboard = predictor.leaderboard(**leaderboard_kwargs)
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000):
        log.info(leaderboard)

    num_models_trained = len(leaderboard)
    if predictor._trainer.model_best is not None:
        num_models_ensemble = len(predictor._trainer.get_minimum_model_set(predictor._trainer.model_best))
    else:
        num_models_ensemble = 1

    save_artifacts(predictor, leaderboard, config)
    shutil.rmtree(predictor.path, ignore_errors=True)

    return result(output_file=config.output_predictions_file,
                  predictions=predictions,
                  probabilities=probabilities,
                  probabilities_labels=prob_labels,
                  target_is_encoded=False,
                  models_count=num_models_trained,
                  models_ensemble_count=num_models_ensemble,
                  training_duration=training.duration,
                  predict_duration=predict.duration)


def save_artifacts(predictor, leaderboard, config):
    artifacts = config.framework_params.get('_save_artifacts', ['leaderboard'])
    try:
        if 'leaderboard' in artifacts:
            leaderboard_dir = output_subdir("leaderboard", config)
            save_pd.save(path=os.path.join(leaderboard_dir, "leaderboard.csv"), df=leaderboard)

        if 'info' in artifacts:
            ag_info = predictor.info()
            info_dir = output_subdir("info", config)
            save_pkl.save(path=os.path.join(info_dir, "info.pkl"), object=ag_info)

        if 'models' in artifacts:
            shutil.rmtree(os.path.join(predictor.path, "utils"), ignore_errors=True)
            models_dir = output_subdir("models", config)
            zip_path(predictor.path, os.path.join(models_dir, "models.zip"))
    except Exception:
        log.warning("Error when saving artifacts.", exc_info=True)


if __name__ == '__main__':
    call_run(run)
