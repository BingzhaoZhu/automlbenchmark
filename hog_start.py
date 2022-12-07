import boto3

s3_client = boto3.client('s3')

BUCKET = 'automl-benchmark-bingzzhu'

PREFIX = 'ec2/2022_09_14/cross_table_pretrain/pretrained_hogwild.ckpt'
response = s3_client.list_objects_v2(Bucket=BUCKET, Prefix=PREFIX)
try:
    for object in response['Contents']:
        print('Deleting', object['Key'])
        s3_client.delete_object(Bucket=BUCKET, Key=object['Key'])
except:
    pass

while True:
    try:
        s3_client = boto3.client('s3')
        response = s3_client.list_objects_v2(Bucket='automl-benchmark-bingzzhu',
                                             Prefix='ec2/2022_09_14/cross_table_pretrain'
                                             )
        for object in response['Contents']:
            print('Deleting', object['Key'])
            s3_client.delete_object(Bucket='automl-benchmark-bingzzhu', Key=object['Key'])
    except:
        break


from autogluon.multimodal.models.ft_transformer import FT_Transformer
import os, torch, json

with open("../backbones/FTTx3") as f:
    kwarg = json.loads(f.read())
# kwarg["additive_attention"] = True
backbone = FT_Transformer(
    **kwarg
)

checkpoint = {
    "state_dict": {name: param for name, param in
                   backbone.state_dict().items()}
}
torch.save(checkpoint, os.path.join("./", "pretrained.ckpt"))
s3 = boto3.resource('s3')
s3.Bucket('automl-benchmark-bingzzhu').upload_file('./pretrained.ckpt',
                                                   'ec2/2022_09_14/cross_table_pretrain/pretrained_hogwild.ckpt')
