import pandas as pd
import numpy as np
from scipy.stats import rankdata
import boto3

locations = {
            # "FTT_light_0": "ftt_ft0_fold_2_light.ag_pretrain.mytest1h.aws.20230116T032244/",
            # "FTT_light_2000": "ftt_ft2000_fold_2_light.ag_pretrain.mytest1h.aws.20230116T084225/",


            # "FTT_heavy_0_fold_1": "ftt_ft0_fold_1.ag_finetune.mytest1h.aws.20230118T090255/",
            # "FTT_heavy_0_fold_2": "ftt_ft0.ag_pretrain.mytest1h.aws.20230118T090250/",

            # "FTT_heavy_2000_fold_1": "ftt_ft2000_fold_1.ag_finetune.mytest1h.aws.20230118T142647/",
            # "FTT_heavy_2000_fold_2": "ftt_ft2000.ag_pretrain.mytest1h.aws.20230118T172423/",


            # "FTT_light_0_fold_1": "ftt_ft0.ag_finetune.mytest1h.aws.20221228T063055/",
            # "FTT_light_0_fold_2": "ftt_ft0.ag_pretrain.mytest1h.aws.20230116T155747/",

            # "FTT_light_2000_fold_1": "ftt_ft2000.ag_finetune.mytest1h.aws.20221228T155807/",
            # "FTT_light_2000_fold_2": "ftt_ft2000.ag_pretrain.mytest1h.aws.20230117T004156/",

            # "FTT_intense_0_fold_1": "ftt_ft0_intense.ag_finetune.mytest1h.aws.20230118T055947/",
            # "FTT_intense_0_fold_2": "ftt_ft0_intense.ag_pretrain.mytest1h.aws.20230117T160037/",

            # "FTT_intense_2000_fold_1": "ftt_ft2000_intense.ag_finetune.mytest1h.aws.20230118T073213/",
            # "FTT_intense_2000_fold_2": "ftt_ft2000_intense.ag_pretrain.mytest1h.aws.20230117T173928/",

            # "FTT_HPO": "ftt_hpo.ag.mytest1h.aws.20230122T063922/",
            # "FTT_HPO_fold_1": "ftt_fold1_pretrained.ag_finetune.mytest1h.aws.20230122T082750/",
            # "FTT_HPO_fold_2": "ftt_fold2_pretrained.ag_pretrain.mytest1h.aws.20230122T095459/",


            # "CAT": "cat_ag.ag.mytest1h.aws.20230114T215102/",
            # "LGBM": "gbm_ag.ag.mytest1h.aws.20230114T204503/",
            # "RF": "rf_ag.ag.mytest1h.aws.20230114T152933/",
            # "XGB": "xgb_ag.ag.mytest1h.aws.20230114T233850/",
            # "NN": "nn_ag.ag.mytest1h.aws.20230115T012407/",
            # "FASTAI": "fastai_ag.ag.mytest1h.aws.20230115T070511/",
            # "TransTab": "transtab.ag.mytest1h.aws.20230123T055954/",

            # "CAT": "cat_hpo.ag.mytest1h.aws.20230119T154044/",
            # "LGBM": "gbm_hpo.ag.mytest1h.aws.20230119T180251/",
            # "RF": "rf_hpo.ag.mytest1h.aws.20230122T153659/",
            # "XGB": "xgb_hpo.ag.mytest1h.aws.20230119T221737/",
            # "NN": "nn_hpo.ag.mytest1h.aws.20230122T063906/",
            # "FASTAI": "fastai_hpo.ag.mytest1h.aws.20230122T063933/",

#   "N0":       "ftt_ft0.ag_pretrain.mytest1h.aws.20230116T155747/",
#   "N250":   "ftt_ft250.ag_pretrain.mytest1h.aws.20230116T155804/",
#   "N500":   "ftt_ft500.ag_pretrain.mytest1h.aws.20230116T173057/",
#   "N1000": "ftt_ft1000.ag_pretrain.mytest1h.aws.20230116T233633/",
#   "N1500": "ftt_ft1500.ag_pretrain.mytest1h.aws.20230116T233702/",
#   "N2000": "ftt_ft2000.ag_pretrain.mytest1h.aws.20230117T004156/",

# "base": "ftt_ft0.ag_pretrain.mytest1h.aws.20230116T155747/",
# "N0": "ftt_rebuttal.ag.mytest1h.aws.20230317T044916/",
# "N2000": "ftt_rebuttal_2000.ag.mytest1h.aws.20230317T064127/",

"base": "ftt_ft0.ag_pretrain.mytest1h.aws.20230117T054745/",
"N0": "ftt_rebuttal_heavy.ag.mytest1h.aws.20230317T045028/",
"N2000": "ftt_rebuttal_heavy_2000.ag.mytest1h.aws.20230317T064205/",

}

# models = ["FTT_heavy_0_fold_1", "FTT_heavy_0_fold_2", "FTT_heavy_2000_fold_1", "FTT_heavy_2000_fold_2", 
# "FTT_light_0_fold_1", "FTT_light_0_fold_2", "FTT_light_2000_fold_1", "FTT_light_2000_fold_2", 
# "FTT_intense_0_fold_1", "FTT_intense_0_fold_2", "FTT_intense_2000_fold_1", "FTT_intense_2000_fold_2",
# 'FASTAI', 'NN', "CAT", "LGBM", "RF", "XGB"]

models = ["FTT_HPO", "FTT_HPO_fold_1", "FTT_HPO_fold_2",
'FASTAI', 'NN', "CAT", "LGBM", "RF", "XGB"]

models = ["TransTab"]

models = ["N0", "base", "N2000"]

# models = ["N0", "N250", "N500", "N1000", "N1500", "N2000"] # "N1", "N2", "N3", "N4", "N5", "N6"]
# models = ["FTT_BL_lowe", "FTT_BL", "FastFTT_BL_lowe", "FastFTT_BL",
#           "FTT_ft_lowe", "FTT_ft", "FTT_pretrain", "FTT_pretrain_lowe", "FastFTT_ft_lowe", "FastFTT_ft"]


s3_client = boto3.client('s3')
bucket = 'automl-benchmark-bingzzhu'

def collect_performance(model):
    df = []
    prefix = "ec2/2022_09_14/" + locations[model]
    results = s3_client.list_objects(Bucket=bucket, Prefix=prefix, Delimiter='/')
    path_to_result = "output/results.csv"
    for obj in results.get('CommonPrefixes'):
        try:
            obj = s3_client.get_object(Bucket='automl-benchmark-bingzzhu', Key=obj.get('Prefix')+path_to_result)
            df.append(pd.read_csv(obj["Body"]))
        except:
            continue
    return pd.concat(df, axis=0)


def separate(model, df, previous):
    # df = df[df["fold"]==0]
    stat = pd.read_csv("./dataset_stat.csv")
    df.rename(columns={"task": "name"}, inplace=True)
    df = stat.merge(df[df.columns], on='name', how='outer')
    # df = df.merge(stat[stat.columns], on='name', how='outer')
    group = {k: v for k, v in df.groupby('type')}
    task_metric = {"regression": "rmse", "binary": "auc", "multiclass": "logloss"}
    for task in group:
        metric = task_metric[task]
        group[task] = group[task][["name", "num_features", "num_instances", "fold", metric, "training_duration", "predict_duration"]]
        group[task].rename(columns={metric: model, "training_duration": model+"_train_time", "predict_duration": model+"_test_time"}, inplace=True)
        # group[task].fillna(-1, inplace=True)
        if task not in previous:
            previous[task] = group[task]
        else:
            previous[task] = previous[task].merge(group[task][['name', 'fold', model, model+"_train_time", model+"_test_time"]], on=['name', 'fold'], how='outer')
    return previous


def rank_models(models, task="binary"):
    n = len(models)
    ranker = np.zeros(n)
    num_tasks = 0
    summary = pd.read_csv("./" + task + ".csv")
    data_stat = pd.read_csv("./dataset_stat.csv")
    summary = summary[["name", "num_features", "num_instances"]+models]
    for _, row in summary.iterrows():
        tmp = []

        if row[models].isna().any():
            continue

        try:
            if data_stat.loc[data_stat['name'] == row["name"]].iloc[0]["num_instances"] < 0:
                continue
        except:
            pass

        for m in models:
            perf = -row[m] if task == "binary" else row[m]
            tmp.append(perf)
        tmp = (rankdata(tmp, method='average'))
        # if tmp[0] == tmp[1]:
        #     continue
        num_tasks += 1
        for idx, rank in enumerate(tmp):
            ranker[idx] += rank

    print("numer of tasks:", num_tasks)
    return ranker

def win_rate(models, tasks, normalize_on=0):
    all_winrate = []
    for i in range(len(models)):
        if i == normalize_on:
            continue
        rk = [rank_models([models[normalize_on], models[i]], task) for task in tasks]
        all_rk = 0
        for rk_ in rk:
            all_rk += rk_
        winrate = all_rk / np.sum(all_rk) * 3
        all_winrate.append(winrate[0]-1)

    return all_winrate

def model_speed(models, tasks, normalize_on=0):
    train_time, test_time = [], []
    for task in tasks:
        summary = pd.read_csv("./" + task + ".csv")
        for _, row in summary.iterrows():
            tmp_train, tmp_test = [], []

            # models_ = ["ensemble", "ensemble_FTT", "ensemble_FastFTT", "ensemble_FTT_row", "ensemble_FTT_pretrain",
            #           "ensemble_ag_ftt_all"]
            if row[models].isna().any():
                continue

            div_train, div_test = 1, 1
            for idx, m in enumerate(models):
                model_train_time = m + "_train_time"
                model_test_time = m + "_test_time"
                if idx == normalize_on:
                    div_train = row[model_train_time]
                    div_test = row[model_test_time]
                tmp_train.append(row[model_train_time])
                tmp_test.append(row[model_test_time])

            train_time.append([i / div_train for i in tmp_train])
            test_time.append([i / div_test for i in tmp_test])

    train_time = np.array(train_time)
    average_train_time = np.mean(train_time, axis=0)
    print("train time: ", average_train_time)

    test_time = np.array(test_time)
    average_test_time = np.mean(test_time, axis=0)
    print("test time: ", average_test_time)
    return average_train_time, average_test_time

if __name__ == "__main__":
    summary = {}
    for model in locations:
        print(f"collecting results for {model}...")
        model_performance = collect_performance(model)
        summary = separate(model, model_performance, summary)

    for task in summary:
        pd.DataFrame(summary[task]).to_csv("./" + task + ".csv")

    print("Comparing among models:", models)
    print("regression:", rank_models(models, "regression"))
    print("binary:", rank_models(models, "binary"))
    print("multiclass:", rank_models(models, "multiclass"))

    model_speed(models, tasks=("regression", "binary", "multiclass"))
    print("win_rate:", win_rate(models, tasks=("regression", "binary", "multiclass")))

    all = rank_models(models, "regression") + rank_models(models, "binary") + rank_models(models, "multiclass")
    # all = rank_models(models, "multiclass")

    print(all / np.sum(all) * sum(range(1, 1+len(models))))





