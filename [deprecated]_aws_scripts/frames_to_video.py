import cv2
import boto3
from pprint import pprint
import os
import imutils
import time

s3 = boto3.client('s3')
bucket_name = 'helen-v1bf79919b2e794658b84abfe4dde08f44pkhelenenv-pkhelenenv'

start = time.time()
files = s3.list_objects(Bucket=bucket_name)['Contents']
for dict in files:
    file_name = dict['Key']
    if '.mp4' in file_name:
        print('Downloading')
        s3.download_file(bucket_name, file_name, 'downloaded_files/{}'.format(file_name[file_name.find('/')+1:]))
        print('Downloaded: ', time.time() - start)
        start = time.time()


frames_path = '/Users/padmanabhankrishnamurthy/PycharmProjects/helen_v2p/aws_scripts/downloaded_files'
frame_array = []
fps = 13
size = None

for i in range(2,27):
    print(os.path.join(frames_path, 'fileName{}.png'.format(str(i))))
    image = cv2.imread(os.path.join(frames_path, 'fileName{}.png'.format(str(i))))
    image = imutils.rotate_bound(image, 90)
    height, width, channels = image.shape
    size = (width, height)
    frame_array.append(image)

out = cv2.VideoWriter(os.path.join(frames_path, 'output.avi'),cv2.VideoWriter_fourcc(*'MJPG'), fps, size)
for i in range(len(frame_array)):
    out.write(frame_array[i])
out.release()