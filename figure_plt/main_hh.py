import boto3
import os, json
from pathlib import Path
import torch


s3 = boto3.resource('s3')
s3.Bucket('automl-benchmark-bingzzhu').download_file(
    'ec2/2022_09_14/cross_table_pretrain/pretrained_hogwild.ckpt',
    './pretrain1.ckpt'
)
pretrain_path = os.path.join("./", 'pretrain1.ckpt')
state_dict1 = torch.load(pretrain_path, map_location=torch.device("cpu"))["state_dict"]


s3.Bucket('automl-benchmark-bingzzhu').download_file(
    'ec2/2022_09_14/cross_table_pretrain/iter_100/pretrained.ckpt',
    './pretrain2.ckpt'
)
pretrain_path = os.path.join("./", 'pretrain2.ckpt')
state_dict2 = torch.load(pretrain_path, map_location=torch.device("cpu"))["state_dict"]


s3.Bucket('automl-benchmark-bingzzhu').download_file(
    'ec2/2022_09_14/cross_table_pretrain/iter_500/pretrained.ckpt',
    './pretrain2.ckpt'
)
pretrain_path = os.path.join("./", 'pretrain2.ckpt')
state_dict3 = torch.load(pretrain_path, map_location=torch.device("cpu"))["state_dict"]

import matplotlib.pyplot as plt
all_1, all_2, all_3 = [], [], []
for i in state_dict1:
    if "bias" in i or "normalization" in i:
        continue
    all_1.append(torch.flatten(state_dict1[i]))
    all_2.append(torch.flatten(state_dict2[i]))
    all_3.append(torch.flatten(state_dict3[i]))
all_1 = torch.cat(all_1)
all_2 = torch.cat(all_2)
all_3 = torch.cat(all_3)
plt.hist(all_1, 1000, density=True)
plt.xlabel('Weights', fontsize=20)
plt.ylabel('Probability', fontsize=20)
plt.xticks(fontsize=20, rotation=45)
plt.xticks(fontsize=20)
ax = plt.gca()
plt.savefig('weight_dist1.png', bbox_inches='tight')

plt.figure()
plt.hist(all_2, 1000, density=True)
plt.xlabel('Weights', fontsize=20)
plt.ylabel('Probability', fontsize=20)
plt.xticks(fontsize=20, rotation=45)
plt.xticks(fontsize=20)
ax = plt.gca()
plt.savefig('weight_dist2.png', bbox_inches='tight')

plt.figure()
plt.hist(all_3, 1000, density=True)
plt.xlabel('Weights', fontsize=20)
plt.ylabel('Probability', fontsize=20)
plt.xticks(fontsize=20, rotation=45)
plt.xticks(fontsize=20)
ax = plt.gca()
plt.savefig('weight_dist3.png', bbox_inches='tight')
