import boto3
from pprint import pprint

s3 = boto3.client('s3')
file_names = []

while(True):
    files = s3.list_objects(Bucket='helen-v1a3fe64bb46824d798ad59f77810cdaa6helenenv-helenenv')
    files = files['Contents']


    for dict in files:
        if dict['Key'] not in file_names:
            print('New File Added: ', dict['Key'])
            file_names.append(dict['Key'])
            print(file_names)