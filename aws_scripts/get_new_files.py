import os
import boto3
from pprint import pprint

s3 = boto3.client('s3')
bucket_name = 'helen-v1a3fe64bb46824d798ad59f77810cdaa6helenenv-helenenv'
file_names = []

listen = True

while(True):
    files = s3.list_objects(Bucket=bucket_name)
    
    if 'Contents' in files.keys():    
        files = files['Contents']

        for dict in files:
            if dict['Key'] not in file_names:
                print('New File Added: ', dict['Key'])
                file_names.append(dict['Key'])
                print(file_names)
                print('Downloading')
                s3.download_file(bucket_name, dict['Key'], 's3_vid.mpg')
                print('Download Complete')

                os.system('python3 predict.py -w /home/ubuntu/lipnet/data/res/lipnet_447.hdf5 -v s3_vid.mpg')
                if os.path.exists('output.csv'):
                    print('Uploading Results')
                    s3.upload_file('output.csv', bucket_name, 'public/output.csv')
                    print('Upload Completed')
                    listen = False

