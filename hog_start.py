# use this file to initialize a shared backbone
# upload the shared backbone to s3 so that all instances can access
import boto3

bucket = "your-s3-bucket-name"
folder = "the-folder-in-s3-bucket"

## delete the pretrained
while True:
    try:
        s3_client = boto3.client('s3')
        response = s3_client.list_objects_v2(Bucket=bucket,
                                             Prefix=folder
                                             )
        for object in response['Contents']:
            print('Deleting', object['Key'])
            s3_client.delete_object(Bucket=bucket, Key=object['Key'])
    except:
        break


## delete the init
from autogluon.multimodal.models.ft_transformer import FT_Transformer, CLSToken
import os, torch, json
from torch import nn


# save new
with open("./FTTx3") as f:
    kwarg = json.loads(f.read())

# # FastFormer
# # kwarg["additive_attention"] = True

# # Saint-V
# kwarg["row_attention"] = True
# kwarg["row_attention_layer"] = "shared"

# # part of backbone
# # kwarg["n_blocks"] = 0


backbone = FT_Transformer(
    **kwarg
)

# # with CLS tokens
# # kwarg["share_qv_weights"] = True
# # 
# # cls_token = CLSToken(
# #                 d_token=192,
# #                 initialization="uniform",
# #             )
# # backbone =  nn.ModuleDict(
# #                 {
# #                     "fusion_transformer": backbone,
# #                     "cls_token": cls_token,
# #                 }
# #             )

checkpoint = {
    "state_dict": {name: param for name, param in
                   backbone.state_dict().items()}
}
torch.save(checkpoint, os.path.join("./", "pretrained.ckpt"))
s3 = boto3.resource('s3')
s3.Bucket(bucket).upload_file('./pretrained.ckpt', 'name-of-the-inilizated-checkpoint')
