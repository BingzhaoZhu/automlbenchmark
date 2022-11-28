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

            # "FTT": "ftt_ag_identical.ag.mytest.aws.20221018T065139/",
            # "FastFTT": "fastftt_ag.ag.mytest.aws.20221012T060457/",
            # "FTT_batchsize_32": "ftt_ag_32.ag.mytest.aws.20221011T080250/",
            # "FastFTT_batchsize_32": "fastftt_ag_32.ag.mytest.aws.20221012T181213/",
            # "HTT": "htt_ag.ag.mytest.aws.20221006T045542/",

            # "FTT": "ftt_ag.ag.mytest.aws.20221020T235009/",
            # "FTT_pretrain_pretrain_fine": "ftt_ag_pretrain_cont.ag.mytest.aws.20221021T050134/",
            # "FTT_pretrain_softpretrain_end0": "ftt_ag_pretrain_cont.ag.mytest.aws.20221022T020151/",
            # "FTT_pretrain_softpretrain_end01": "ftt_ag_pretrain_cont.ag.mytest.aws.20221022T075728/",
            # "FTT_pretrain_mix_loss": "ftt_ag_pretrain_cont.ag.mytest.aws.20221022T230745/",

            # "ensemble": "ensemble_ag.ag.mytest4h.aws.20221027T233620/",
            # "ensemble_FTT": "ensemble_ag_ftt.ag.mytest4h.aws.20221027T233709/",
            # "ensemble_FastFTT": "ensemble_ag_fastftt.ag.mytest4h.aws.20221031T044235/",
            # "ensemble_FTT_row": "ensemble_ag_ftt_rowatt.ag.mytest4h.aws.20221027T233735/",
            # "ensemble_FTT_pretrain": "ensemble_ag_ftt_pretrain.ag.mytest4h.aws.20221027T233830/",
            # "ensemble_ag_ftt_all": "ensemble_ag_ftt_all.ag.mytest4h.aws.20221031T211433/",

            # "ensemble_bq": "ensemble_ag_bq.ag.mytest24h.aws.20221105T052819/",
            # "ensemble_bq_m6i": "ensemble_ag_bq.ag.mytest24h.aws.20221109T020748/",
            # # "ensemble_bq": "ensemble_ag_bq.ag.mytest4h.aws.20221102T230938/",
            # # "ensemble_FTT_pretrain_bq": "ensemble_ag_ftt_pretrain_bq.ag.mytest4h.aws.20221103T051502/",
            # "ensemble_ag_ftt_all_bq": "ensemble_ag_ftt_all_bq.ag.mytest24h.aws.20221105T061959/", #"ensemble_ag_ftt_all_mq.ag.mytest4h.aws.20221104T174422/",
            # "ensemble_ag_ftt_all_bq_m6i": "ensemble_ag_ftt_all_bq_cpu.ag.mytest24h.aws.20221109T020548/",

            # "ensemble_bq": "ensemble_ag_bq.ag.mytest4h.aws.20221102T230938/",
            # "ensemble_fastftt_bq": "ensemble_ag_fastftt_bq.ag.mytest4h.aws.20221110T232827/",
            # "ensemble_autoftt_bq": "ensemble_ag_autoftt_bq.ag.mytest4h.aws.20221110T232806/",
            # "ensemble_ftt_bq": "ensemble_ag_accurateftt_bq.ag.mytest4h.aws.20221110T232734/",

            # "FTT_dist": "ftt_ag_pretrain_dist.ag.mytest.aws.20221027T170021/",
            # "FTT_cont": "ftt_ag_pretrain_cont.ag.mytest.aws.20221027T035040/",
            # "FTT_recon": "ftt_ag_pretrain_recon.ag.mytest.aws.20221026T004154/",
            # "FTT_both": "ftt_ag_pretrain_both.ag.mytest.aws.20221027T035048/",
            # "FTT": "ftt_ag.ag.mytest.aws.20221025T020434/",
            # "FTT_nodecay": "ftt_ag_pretrain_dist.ag.mytest.aws.20221026T004113/",
            # "FTT_row_attention_1_gt": "ftt_ag_row_attention_1_gt.ag.mytest.aws.20221024T074835/",
            # # "FTT_row_attention_10": "ftt_ag_row_attention_10.ag.mytest.aws.20221022T020145/",
            # "FTT_row_attention_10_gt": "ftt_ag_row_attention_10_gt.ag.mytest.aws.20221024T174142/",
            # # "FTT_row_attention_gt": "ftt_ag_row_attention_gt.ag.mytest.aws.20221020T234929/",
            # "FTT_row_attention_20_gt": "ftt_ag_row_attention_-1_gt.ag.mytest.aws.20221024T074900/",

            # "FTT_row_attention_first": "ftt_ag_row_attention.ag.mytest.aws.20221001T180711/",
            # "FTT_row_attention_last": "ftt_ag_row_attention_10.ag.mytest.aws.20221019T215218/",
            # "FTT_row_attention_alter": "ftt_ag_row_attention_10.ag.mytest.aws.20221020T043406/",
            # "FTT_row_attention_cls": "ftt_ag_row_attention_10.ag.mytest.aws.20221019T215218/",

    # "N0": "ftt_ag.ag.mytest1h.aws.20221121T220729/",
    # "N1": "ftt_ag.ag.mytest1h.aws.20221121T233555/",
    # "N2": "ftt_ag.ag.mytest1h.aws.20221122T060222/",
    # "N3": "ftt_ag.ag.mytest1h.aws.20221122T073032/",
    # "N4": "ftt_ag.ag.mytest1h.aws.20221122T085738/",
    # "N5": "ftt_ag.ag.mytest1h.aws.20221122T102505/",
    # "N6": "ftt_ag.ag.mytest1h.aws.20221122T115204/",

    "linear": "ftt_ag_hog_ft0.ag.mytest1h.aws.20221128T065929/",
    "positional": "ftt_ag_hog_ft1000.ag.mytest1h.aws.20221128T155247/",
}

# models = ['FASTAI', 'NN', 'FTT_row_attention_20_gt', "CAT", "LGBM", "RF", "XGB"]
# models = ["FTT", "FTT_pretrain_identical",
#           "FTT_pretrain_randperm_03", "FTT_pretrain_randperm_06", "FTT_pretrain_randperm_09",
#           "FTT_pretrain_randblk_03",  "FTT_pretrain_randblk_06",  "FTT_pretrain_randblk_09"]
# models = ["FTT", "FTT_pretrain_randperm_06"]
# models = ["FTT", "FTT_batchsize_32", "FastFTT", "FastFTT_batchsize_32"]
# models = ["FTT_row_attention_first", "FTT_row_attention_last", "FTT_row_attention_alter", "FTT_row_attention_cls"]
models = ["FTT", "FTT_row_attention_1_gt", "FTT_row_attention_10_gt", "FTT_row_attention_20_gt"]
# models = ["FTT_row_attention_last", "FTT_row_attention_alter"]
# models = ["FTT", "FTT_pretrain_pretrain_fine", "FTT_pretrain_softpretrain_end0", "FTT_pretrain_softpretrain_end01", "FTT_pretrain_mix_loss"]
models = ["FTT", "FTT_recon"]
models = ["FTT", "FTT_cont", "FTT_recon", "FTT_both", "FTT_dist"]
models = ["ensemble", "ensemble_FTT", "ensemble_FastFTT", "ensemble_FTT_row", "ensemble_FTT_pretrain", "ensemble_ag_ftt_all"]
models = ["ensemble_autoftt_bq", "ensemble_ftt_bq"] #, "ensemble_autoftt_bq", "ensemble_ftt_bq"] #, ensemble_bq, ensemble_FTT_pretrain_bq, ensemble_ag_ftt_all_bq]
models = ["linear", "positional"] # "N1", "N2", "N3", "N4", "N5", "N6"]

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

    all = rank_models(models, "regression") + rank_models(models, "binary") + rank_models(models, "multiclass")

    print(all / np.sum(all) * 3)





