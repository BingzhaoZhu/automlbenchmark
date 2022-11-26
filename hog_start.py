import boto3

s3_client = boto3.client('s3')

BUCKET = 'automl-benchmark-bingzzhu'

PREFIX = 'ec2/2022_09_14/cross_table_pretrain/pretrained_hogwild.ckpt'
response = s3_client.list_objects_v2(Bucket=BUCKET, Prefix=PREFIX)
for object in response['Contents']:
    print('Deleting', object['Key'])
    s3_client.delete_object(Bucket=BUCKET, Key=object['Key'])

s3_client = boto3.client('s3')
response = s3_client.list_objects_v2(Bucket='automl-benchmark-bingzzhu',
                                     Prefix='ec2/2022_09_14/cross_table_pretrain/job_status/'
                                     )
for object in response['Contents']:
    print('Deleting', object['Key'])
    s3_client.delete_object(Bucket='automl-benchmark-bingzzhu', Key=object['Key'])

# import json
# job_status = {}
# with open('./job_status.txt', 'w') as fp:
#     fp.write(json.dumps(job_status))
# s3 = boto3.resource('s3')
# s3.Bucket('automl-benchmark-bingzzhu').upload_file('./job_status.txt',
#                                                    'ec2/2022_09_14/cross_table_pretrain/job_status.txt')
