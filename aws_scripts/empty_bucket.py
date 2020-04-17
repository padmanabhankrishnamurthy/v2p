# Clear Bucket before each new run
# - Since we want only one mp4 file to exist in the bucket at a time

import boto3

s3 = boto3.client('s3')
bucket_name = 'helen-v1bf79919b2e794658b84abfe4dde08f44pkhelenenv-pkhelenenv'

files = s3.list_objects(Bucket=bucket_name)
if 'Contents' in files.keys():
    files = files['Contents']
    for dict in files:
        s3.delete_object(Bucket=bucket_name, Key=dict['Key'])
    print('Bucket Emptied')
