import openml
import pandas as pd

def get_tk_id():
    task_id = {
        "regression": [359944, 359929, 233212, 359937, 359950,
                       359938, 233213, 359942, 233211, 359936,
                       359952, 359951, 359949, 233215, 360945,
                       167210, 359943, 359941, 359946, 360933,
                       360932, 359930, 233214, 359948, 359931,
                       359932, 359933, 359934, 359939, 359945,
                       359935, 317614, 359940],
        "classification": [190411, 359983, 189354, 189356, 10090,
                           359979, 168868, 190412, 146818, 359982,
                           359967, 359955, 359960, 359973, 359968,
                           359992, 359959, 359957, 359977, 7593,
                           168757, 211986, 168909, 189355, 359964,
                           359954, 168910, 359976, 359969, 359970,
                           189922, 359988, 359984, 360114, 359966,
                           211979, 168911, 359981, 359962, 360975,
                           3945, 360112, 359991, 359965, 190392,
                           359961, 359953, 359990, 359980, 167120,
                           359993, 190137, 359958, 190410, 359971,
                           168350, 360113, 359956, 359989, 359986,
                           359975, 359963, 359994, 359987, 168784,
                           359972, 190146, 359985, 146820, 359974,
                           2073]
    }
    return task_id


def _get_ds_id(tid):
    try:
        return openml.tasks.get_task(tid, download_data=False).dataset_id
    except:
        print('Failed to get task', tid)

def get_ds_id():
    ts_id = get_tk_id()
    ds_id = {}
    for task_type in ts_id:
        ds_id[task_type] = [_get_ds_id(i) for i in ts_id[task_type]]
    return ds_id

def reorg(df):
    model = df["1"][0]
    df.drop(["Unnamed: 0", "1"], axis=1, inplace=True)
    df.rename(columns={'0': 'dataset_id', '2': 'task', '3': model}, inplace=True)
    return df

def load_dataset_stat():
    all_dsid = get_ds_id()
    dsid = all_dsid["regression"] + all_dsid["classification"]
    df = openml.datasets.list_datasets(dsid, output_format="dataframe")
    df = df[['did', 'name', 'NumberOfFeatures', 'NumberOfInstances']]
    df.rename(columns={'did': 'dataset_id',
                       'NumberOfFeatures': 'num_features',
                       'NumberOfInstances': 'num_instances'}, inplace=True)
    return df

summary = load_dataset_stat()
pd.DataFrame(summary).to_csv("./dataset_stat.csv")
