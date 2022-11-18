import boto3

s3_client = boto3.client('s3')

BUCKET = 'automl-benchmark-bingzzhu'

PREFIX = 'cross_table_pretrain/'
response = s3_client.list_objects_v2(Bucket=BUCKET, Prefix=PREFIX)
for object in response['Contents']:
    print('Deleting', object['Key'])
    s3_client.delete_object(Bucket=BUCKET, Key=object['Key'])

s3 = boto3.client('s3')
directory_name = 'cross_table_pretrain/raw'
s3.put_object(Bucket=BUCKET, Key=(directory_name+'/'))
