from torch.utils.data import Dataset, DataLoader
import numpy as np
import pyarrow.parquet as pq
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

def is_categorical(df):
    categorical_indicator = []
    for c in df.columns:
        categorical_indicator.append(pd.api.types.is_categorical_dtype(df[c]))
    return categorical_indicator


def data_handle_missing_value(X_train, X_test, cat_columns, con_columns):
    cat_dims = []
    n_train = X_train.shape[0]
    X = pd.concat((X_train, X_test), axis=0)
    for col in cat_columns:
        X[col] = X[col].astype("object")
        X[col].fillna("MissingValue", inplace=True)
        l_enc = LabelEncoder()
        X[col] = l_enc.fit_transform(X[col].values)
        cat_dims.append(len(l_enc.classes_))
    for col in con_columns:
        X[col].fillna(X_train.loc[:, col].mean(), inplace=True)
    X_train, X_test = X.iloc[:n_train, :], X.iloc[n_train:, :]
    return X_train.values, X_test.values, cat_dims


def pre_process(dataset, is_classification):
    train_path, test_path = dataset.train.path, dataset.test.path
    label = dataset.target.name
    train_data = pq.ParquetDataset(train_path).read().to_pandas()
    test_data = pq.ParquetDataset(test_path).read().to_pandas()

    y_train, y_test = train_data[label], test_data[label]
    X_train = train_data.loc[:, train_data.columns != label]
    X_test = test_data.loc[:, test_data.columns != label]

    categorical_indicator = is_categorical(X_train)
    attribute_names = X_train.columns

    # separate categorical columns from continous columns
    cat_columns = X_train.columns[list(np.where(np.array(categorical_indicator) == True)[0])].tolist()
    con_columns = list(set(X_train.columns.tolist()) - set(cat_columns))
    cat_idxs = list(np.where(np.array(categorical_indicator) == True)[0])
    con_idxs = list(set(range(len(X_train.columns))) - set(cat_idxs))

    X_train, X_test, cat_dims = data_handle_missing_value(X_train, X_test, cat_columns, con_columns)

    if is_classification:
        l_enc = LabelEncoder()
        y_train = l_enc.fit_transform(y_train)
        y_test = l_enc.transform(y_test)
    else:
        y_train, y_test = y_train.astype('float32'), y_test.astype('float32')
        l_enc = StandardScaler()
        y_train = l_enc.fit_transform(y_train[:, None])
        y_test = l_enc.transform(y_test[:, None])

    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, shuffle=True, test_size=0.125, random_state=0)

    train_mean = np.array(X_train[:, con_idxs], dtype=np.float32).mean(0)
    train_std = np.array(X_train[:, con_idxs], dtype=np.float32).std(0)
    train_std = np.where(train_std < 1e-6, 1e-6, train_std)

    data = {"X_train": np.asarray(X_train), "y_train": np.asarray(y_train),
            "X_valid": np.asarray(X_valid), "y_valid": np.asarray(y_valid),
            "X_test": np.asarray(X_test), "y_test": np.asarray(y_test)}
    info = {"train_mean": train_mean, "train_std": train_std,
            "cat_dims": cat_dims, "cat_idxs": cat_idxs,
            "con_idxs": con_idxs, "attribute_names": attribute_names,
            "target_name": label, "label_encoder": l_enc}

    return data, info


class TabularDataset(Dataset):
    def __init__(self, data, info, split="train", normalize=True):
        cat_idxs, con_idxs = info["cat_idxs"], info["con_idxs"]
        X, self.y = data["X_" + split], data["y_" + split]
        self.X1 = X[:, cat_idxs].copy().astype(np.int64)  # categorical columns
        self.X2 = X[:, con_idxs].copy().astype(np.float32)  # numerical columns
        if normalize:
            mean, std = info["train_mean"], info["train_std"]
            self.X2 = (self.X2 - mean) / std

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X1[idx], self.X2[idx], self.y[idx]


def get_torch_dataloader(dataset, is_classification, batch_size=64, shuffle=True):
    data, info = pre_process(dataset, is_classification)
    ds_train = TabularDataset(data, info, "train")
    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=shuffle)

    ds_valid = TabularDataset(data, info, "valid")
    dl_valid = DataLoader(ds_valid, batch_size=batch_size, shuffle=shuffle)

    ds_test = TabularDataset(data, info, "test")
    dl_test = DataLoader(ds_test, batch_size=batch_size, shuffle=shuffle)

    return dl_train, dl_valid, dl_test, info

