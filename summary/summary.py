import pandas as pd
import numpy as np
from scipy.stats import rankdata
import boto3

locations = {"FTTrans_pretrain": "fttransformer_gpu.ag.mytest.aws.20220917T020912/",
             "FTTrans": "fttransformer_gpu.ag.mytest.aws.20220917T055318/",
             "CAT": "cat_ag.ag.mytest.aws.20220917T190721/",
             "LGBM": "gbm_ag.ag.mytest.aws.20220917T173005/",
             "RF": "rf_ag.ag.mytest.aws.20220917T181110/",
             "XGB": "xgb_ag.ag.mytest.aws.20220917T202434/",
             }
s3_client = boto3.client('s3')
bucket = 'automl-benchmark-bingzzhu'


def collect_performance(model):
    df = []
    prefix = "ec2/2022_09_14/" + locations[model]
    results = s3_client.list_objects(Bucket=bucket, Prefix=prefix, Delimiter='/')
    path_to_result = "output/results.csv"
    for obj in results.get('CommonPrefixes'):
        obj = s3_client.get_object(Bucket='automl-benchmark-bingzzhu', Key=obj.get('Prefix')+path_to_result)
        df.append(pd.read_csv(obj["Body"]))
    return pd.concat(df, axis=0)


def separate(model, df, previous):
    group = {k: v for k, v in df.groupby('type')}
    for task in group:
        metric = "rmse" if task == "regression" else "acc"
        group[task] = group[task][["task", metric]]
        group[task].rename(columns={metric: model}, inplace=True)
        # group[task].fillna(-1, inplace=True)
        if task not in previous:
            previous[task] = group[task]
        else:
            previous[task] = previous[task].merge(group[task][['task', model]], on='task', how='right')
    return previous


def rank_models(models, task="binary"):
    n = len(models)
    ranker = np.zeros((n, n))
    summary = pd.read_csv("./" + task + ".csv")
    for _, row in summary.iterrows():
        tmp = []

        if row.isna().any():
            continue

        for m in models:
            perf = row[m] if task == "regression" else -row[m]
            tmp.append(perf)
        tmp = (rankdata(tmp)-1).astype(int)
        for idx, rank in enumerate(tmp):
            ranker[idx, rank] += 1

    return ranker


if __name__ == "__main__":
    # summary = {}
    # for model in locations:
    #     model_performance = collect_performance(model)
    #     summary = separate(model, model_performance, summary)
    #
    # for task in summary:
    #     pd.DataFrame(summary[task]).to_csv("./" + task + ".csv")

    models = ['FTTrans_pretrain', 'FTTrans', "CAT", "LGBM", "RF", "XGB"]
    print("regression:", rank_models(models, "regression"))
    print("binary:", rank_models(models, "binary"))
    print("multiclass:", rank_models(models, "multiclass"))



