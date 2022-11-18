import boto3
import torch
from pathlib import Path

def average_checkpoints(
    checkpoint_paths,
):
    """
    Average a list of checkpoints' state_dicts.
    Reference: https://github.com/rwightman/pytorch-image-models/blob/master/avg_checkpoints.py

    Parameters
    ----------
    checkpoint_paths
        A list of model checkpoint paths.

    Returns
    -------
    The averaged state_dict.
    """
    if len(checkpoint_paths) > 1:
        avg_state_dict = {}
        avg_counts = {}
        for per_path in checkpoint_paths:
            state_dict = torch.load(per_path, map_location=torch.device("cpu"))["state_dict"]
            for k, v in state_dict.items():
                if k not in avg_state_dict:
                    avg_state_dict[k] = v.clone().to(dtype=torch.float64)
                    avg_counts[k] = 1
                else:
                    avg_state_dict[k] += v.to(dtype=torch.float64)
                    avg_counts[k] += 1
            del state_dict

        for k, v in avg_state_dict.items():
            v.div_(avg_counts[k])

        # convert to float32.
        float32_info = torch.finfo(torch.float32)
        for k in avg_state_dict:
            avg_state_dict[k].clamp_(float32_info.min, float32_info.max).to(dtype=torch.float32)
    else:
        avg_state_dict = torch.load(checkpoint_paths[0], map_location=torch.device("cpu"))["state_dict"]

    return avg_state_dict


def get_file_folders(s3_client, bucket_name, prefix=""):
    file_names = []
    folders = []

    default_kwargs = {
        "Bucket": bucket_name,
        "Prefix": prefix
    }
    next_token = ""

    while next_token is not None:
        updated_kwargs = default_kwargs.copy()
        if next_token != "":
            updated_kwargs["ContinuationToken"] = next_token

        response = s3_client.list_objects_v2(**updated_kwargs)
        contents = response.get("Contents")

        for result in contents:
            key = result.get("Key")
            if key[-1] == "/":
                folders.append(key)
            else:
                file_names.append(key)

        next_token = response.get("NextContinuationToken")

    return file_names, folders


def download_files(s3_client, bucket_name, local_path, file_names, folders):
    local_path = Path(local_path)

    for folder in folders:
        folder_path = Path.joinpath(local_path, folder)
        folder_path.mkdir(parents=True, exist_ok=True)

    for file_name in file_names:
        file_path = Path.joinpath(local_path, file_name)

        file_path.parent.mkdir(parents=True, exist_ok=True)
        s3_client.download_file(
            bucket_name,
            file_name,
            str(file_path)
        )

s3_client = boto3.client('s3')
BUCKET = 'automl-benchmark-bingzzhu'
PREFIX = 'ec2/2022_09_14/cross_table_pretrain/raw'
file_names, folders = get_file_folders(s3_client, BUCKET, PREFIX)
download_files(
    s3_client,
    BUCKET,
    "../ckpts",
    file_names,
    folders
)

import os
path = "../ckpts/cross_table_pretrain/raw/"
dir_list = os.listdir(path)
dir_list = [path + i for i in dir_list]
checkpoint = average_checkpoints(dir_list)
torch.save(checkpoint, os.path.join("../ckpts/", "pretrained.ckpt"))
s3 = boto3.resource('s3')
s3.Bucket('automl-benchmark-bingzzhu').upload_file('../ckpts/pretrained.ckpt', 'ec2/2022_09_14/cross_table_pretrain/pretrained.ckpt')

for f in os.listdir(path):
    os.remove(os.path.join(path, f))
os.remove(os.path.join("../ckpts/", "pretrained.ckpt"))



response = s3_client.list_objects_v2(Bucket=BUCKET, Prefix=PREFIX)
for object in response['Contents']:
    print('Deleting', object['Key'])
    s3_client.delete_object(Bucket=BUCKET, Key=object['Key'])

s3 = boto3.client('s3')
directory_name = 'ec2/2022_09_14/cross_table_pretrain/raw'
s3.put_object(Bucket=BUCKET, Key=(directory_name+'/'))