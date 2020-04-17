# THIS PROGRAM IS FOR EXCLUSIVE USE ON THE AWS EC2 INSTANCE
# - File is always running
# - Listens to bucket for existence of mp4 file uploaded from iOS
# - Converts mp4 to mpg
# - Runs prediction on Lipnet stored on EC2
# - Uploads output.txt file to Bucket
# - Terminates Listening


import os
import boto3
from pprint import pprint

print('Listening')
s3 = boto3.client('s3')
bucket_name = 'helen-v1bf79919b2e794658b84abfe4dde08f44pkhelenenv-pkhelenenv'
file_names = []

listen = True

while(listen):
    files = s3.list_objects(Bucket=bucket_name)
    #pprint(files)    
    if 'Contents' in files.keys():    
        files = files['Contents']
        for dict in files:
            if dict['Key'] not in file_names:
                print('New File Added: ', dict['Key'])
                file_names.append(dict['Key'])
                print(file_names)
                
                if '.mp4' in dict['Key']:
                    print('Downloading')
                    s3.download_file(bucket_name, dict['Key'], 'input_vid.mp4')
                    print('Download Complete')
                
                print('Converting')
                os.system('ffmpeg -i {} {}'.format('input_vid.mp4', 'input_vid.mpg'))
                print('Conversion Complete')
                
                os.system('python3 predict.py -w /home/ubuntu/lipnet/data/res/lipnet_447.hdf5 -v input_vid.mpg')
                #os.system('python3 predict.py -v input_vid.mpg -w /home/ubuntu/lipnet/data/res/2018-09-26-02-30/lipnet_065_1.96.hdf5')
                if os.path.exists('output.txt'):
                    print('Uploading Results')
                    s3.upload_file('output.txt', bucket_name, 'public/output.txt')
                    print('Upload Completed')
                    listen = False

if not listen:
	'''files = s3.list_objects(Bucket=bucket_name)
	if 'Contents' in files.keys():
		files = files['Contents']
		for dict in files:
			if '.mp4' in dict['Key']:
				s3.delete_object(Bucket=bucket_name, Key=dict['Key'])
		print('Bucket Emptied')
	'''
	os.system('rm input_vid.mp4 && rm input_vid.mpg && rm output.txt')

