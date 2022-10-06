import pandas as pd
import numpy as np
from scipy.stats import rankdata
import boto3

locations = {
            # "FTTrans_pretrain": "fttransformer_gpu_pretrain_3.ag.mytest.aws.20220921T144652/",
            # "FTTrans": "fttransformer_gpu_3.ag.mytest.aws.20220921T122437/",
            # "WideDeep": "widedeep.ag.mytest.aws.20220921T180925/",
            # "WideDeep_pretrain": "widedeep_pretrain.ag.mytest.aws.20220921T172633/",
            # "CAT": "cat_ag.ag.mytest.aws.20220927T070920/",
            # # "CAT_pretrain": "cat_ag_pretrain.ag.mytest.aws.20220927T230900/",
            # "LGBM": "gbm_ag.ag.mytest.aws.20220917T173005/",
            # "RF": "rf_ag.ag.mytest.aws.20220917T181110/",
            # "XGB": "xgb_ag.ag.mytest.aws.20220917T202434/",
            # "NN": "nn_ag.ag.mytest.aws.20220920T174058/",
            # "FASTAI": "fastai_ag.ag.mytest.aws.20220920T185736/",

            "FTT": "ftt_ag.ag.mytest.aws.20221003T205638/",
            # "HTT": "htt_ag.ag.mytest.aws.20221005T170353/",
            # "FTT_pretrain_identical": "ftt_ag_pretrain_identical.ag.mytest.aws.20220928T200551/",
            "FTT_pretrain_randblk_03": "ftt_ag_pretrain_randperm_03.ag.mytest.aws.20221005T072915/",
            "FTT_pretrain_randblk_06": "ftt_ag_pretrain_randperm_06.ag.mytest.aws.20221003T231923/",
            # "FTT_pretrain_randblk_09": "ftt_ag_pretrain_randblk_09.ag.mytest.aws.20221005T144032/",
            "FTT_pretrain_randperm_03": "ftt_ag_pretrain_randperm_03.ag.mytest.aws.20221006T000916/",
            "FTT_pretrain_randperm_06": "ftt_ag_pretrain_randperm_06.ag.mytest.aws.20221006T023141/",
            # "FTT_pretrain_randperm_09": "ftt_ag_pretrain_randperm_09.ag.mytest.aws.20221004T014300/",

            # "FTT": "ftt_ag.ag.mytest.aws.20221003T205638/",
            # "FTT_row_attention": "ftt_ag_row_attention.ag.mytest.aws.20221004T040737/",
}
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
    stat = pd.read_csv("./dataset_stat.csv")
    df.rename(columns={"task": "name"}, inplace=True)
    df = stat.merge(df, on='name', how='right')
    group = {k: v for k, v in df.groupby('type')}
    task_metric = {"regression": "rmse", "binary": "auc", "multiclass": "acc"}
    for task in group:
        metric = task_metric[task]
        group[task] = group[task][["name", "num_features", "num_instances", metric]]
        group[task].rename(columns={metric: model}, inplace=True)
        # group[task].fillna(-1, inplace=True)
        if task not in previous:
            previous[task] = group[task]
        else:
            previous[task] = previous[task].merge(group[task][['name', model]], on='name', how='right')
    return previous


def rank_models(models, task="binary"):
    n = len(models)
    ranker = np.zeros((n, n))
    summary = pd.read_csv("./" + task + ".csv")
    summary = summary[["name", "num_features", "num_instances"]+models]
    for _, row in summary.iterrows():
        tmp = []

        if row.isna().any():
            continue

        if row["num_instances"] < 0:
            continue

        for m in models:
            perf = row[m] if task == "regression" else -row[m]
            tmp.append(perf)
        tmp = (rankdata(tmp)-1).astype(int)
        for idx, rank in enumerate(tmp):
            ranker[idx, rank] += 1

    num_tasks = np.sum(ranker)/n
    average_rank = ranker * np.arange(n)[None, :]
    average_rank = np.sum(average_rank, axis=1)/num_tasks + 1
    print(average_rank)

    return ranker


if __name__ == "__main__":
    summary = {}
    for model in locations:
        print(f"collecting results for {model}...")
        model_performance = collect_performance(model)
        summary = separate(model, model_performance, summary)

    for task in summary:
        pd.DataFrame(summary[task]).to_csv("./" + task + ".csv")

    models = ['FASTAI', 'NN', 'FTT', 'FTT_pretrain_randperm_09', "CAT", "LGBM", "RF", "XGB"]
    models = ["FTT", "FTT_pretrain_identical",
              "FTT_pretrain_randperm_03", "FTT_pretrain_randperm_06", "FTT_pretrain_randperm_09",
              "FTT_pretrain_randblk_03",  "FTT_pretrain_randblk_06",  "FTT_pretrain_randblk_09"]
    # models = ["FTT", "HTT"]
    models = ["FTT",
              "FTT_pretrain_randperm_03", "FTT_pretrain_randperm_06",
              "FTT_pretrain_randblk_03",  "FTT_pretrain_randblk_06"]
    print("Comparing among models:", models)
    print("regression:", rank_models(models, "regression"))
    print("binary:", rank_models(models, "binary"))
    print("multiclass:", rank_models(models, "multiclass"))



