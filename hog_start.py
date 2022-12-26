import boto3

## delete the pretrained
while True:
    try:
        s3_client = boto3.client('s3')
        response = s3_client.list_objects_v2(Bucket='automl-benchmark-bingzzhu',
                                             Prefix='ec2/2022_09_14/cross_table_pretrain/pretrain_reconstruction_saint'
                                             )
        for object in response['Contents']:
            print('Deleting', object['Key'])
            s3_client.delete_object(Bucket='automl-benchmark-bingzzhu', Key=object['Key'])
    except:
        break


## delete the init
from autogluon.multimodal.models.ft_transformer import FT_Transformer, CLSToken
import os, torch, json
from torch import nn

# s3_client = boto3.client('s3')

# BUCKET = 'automl-benchmark-bingzzhu'

# PREFIX = 'ec2/2022_09_14/cross_table_pretrain/pretrained_hogwild.ckpt'
# response = s3_client.list_objects_v2(Bucket=BUCKET, Prefix=PREFIX)
# try:
#     for object in response['Contents']:
#         print('Deleting', object['Key'])
#         s3_client.delete_object(Bucket=BUCKET, Key=object['Key'])
# except:
#     pass


## save new
# with open("../backbones/FTTx3") as f:
#     kwarg = json.loads(f.read())

# # FastFormer
# # kwarg["additive_attention"] = True

# # Saint-V
# # kwarg["row_attention"] = True
# # kwarg["row_attention_layer"] = "shared"

# # Saint-V
# # kwarg["row_attention"] = True
# # kwarg["row_attention_layer"] = "shared"

# # part of backbone
# kwarg["n_blocks"] = 0


# backbone = FT_Transformer(
#     **kwarg
# )

# # with CLS tokens
# # kwarg["share_qv_weights"] = True

# cls_token = CLSToken(
#                 d_token=192,
#                 initialization="uniform",
#             )
# backbone =  nn.ModuleDict(
#                 {
#                     "fusion_transformer": backbone,
#                     "cls_token": cls_token,
#                 }
#             )

# checkpoint = {
#     "state_dict": {name: param for name, param in
#                    backbone.state_dict().items()}
# }
# torch.save(checkpoint, os.path.join("./", "pretrained.ckpt"))
# s3 = boto3.resource('s3')
# s3.Bucket('automl-benchmark-bingzzhu').upload_file('./pretrained.ckpt',
#                                                    'ec2/2022_09_14/cross_table_pretrain/pretrained_hogwild_only_cls.ckpt')
