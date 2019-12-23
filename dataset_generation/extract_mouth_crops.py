import dlib
from imutils import face_utils
import cv2
import skvideo.io
import numpy as np
import os
from pprint import pprint
from colorama import Back, Style

videos_path = '/Users/padmanabhankrishnamurthy/Desktop/lrs3/test-3'
mouth_crops_dir = '/Users/padmanabhankrishnamurthy/Desktop/lrs3/mouth_crops/'
predictor_path = '/Users/padmanabhankrishnamurthy/PycharmProjects/helen_v2p/data/shape_predictor_68_face_landmarks.dat' #dlib face-landmark predictor; acts upon faces detected by dlib face detector

def extract_mouth_crop(file:str, show_crop = False, visualize = False):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)

    video_data = skvideo.io.vread(file)
    print(video_data.shape)
    mouth_data = []

    broken = False

    for index, frame in enumerate(video_data):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detected = detector(gray, 1)
        frame_copy = frame #save original frame

        if(len(detected) <= 0): #if no face detected at a particular frame, the whole video is unusable, don't include it in dataset
            print(index, ' : NO FACE DETECTED')
            broken = True
            break

        else:
            shape = predictor(gray, detected[0]) #detected[0] ensures only first detected face is chosen for further processing
            shape = face_utils.shape_to_np(shape) #shape predictor outputs 68 coordinate pairs, each responsible for a particular landmark
            '''
            points 48-68 responsible for mouth; see https://www.pyimagesearch.com/2017/04/10/detect-eyes-nose-lips-jaw-dlib-opencv-python/
            print(list(face_utils.FACIAL_LANDMARKS_IDXS.items())[0])
            '''
            mouth_points = shape[48:68]
            center = np.mean(mouth_points, axis=0)
            center = (center[1], center[0]) #flip x and y coords of center since cv2 takes y coordinates first

            size = (50,100) #mouth crop size
            start = tuple(int(a - b // 2) for a, b in zip(center, size))
            end = tuple(int(a + b) for a, b in zip(start, size))
            slices = tuple(slice(a, b) for a, b in zip(start, end))
            crop = frame[slices]
            mouth_data.append(crop)

            if(show_crop):
                cv2.imshow('window', crop)
                cv2.waitKey(25)

            if(visualize): #only to visualize mouth crops
                hull = cv2.convexHull(mouth_points) #draw convex hull around mouth
                cv2.drawContours(frame_copy, [hull], -1, (19, 199, 109), -1)
                cv2.imshow('window', frame_copy)
                cv2.waitKey(50)

    if not broken:
        try:
            mouth_data = np.array(mouth_data)
            print(mouth_data.shape)
            return mouth_data
        except ValueError:
            return False
    else:
        return False

# sample_video = '/Users/padmanabhankrishnamurthy/Desktop/lrs3/test-3/eZj5n8ScTkI/00005.mp4'
# extract_mouth_crop(sample_video, False, False)

# total_videos, processed_videos = 0,0
# explored_dirs = []
#
# for dir in os.listdir(videos_path):
#     speaker_name = dir
#     dir = os.path.join(videos_path, speaker_name)
#
#     if not os.path.exists(mouth_crops_dir + speaker_name): #create speaker directory in mouth crops directory
#         os.mkdir(mouth_crops_dir + speaker_name)
#
#     if not os.path.isdir(dir): #dont process .DS_Store
#         continue
#
#     for file in os.listdir(dir):
#         if '.mp4' in file:
#             total_videos+=1
#             video_name = file
#             file = os.path.join(dir, video_name)
#             print('=====', file, '=======')
#
#             if dir not in explored_dirs:
#                 mouth_crop_array = extract_mouth_crop(file)
#             else:
#                 processed_videos+=1
#                 print(Back.GREEN + 'Directory populated and exists at ', mouth_crops_dir + speaker_name, Style.RESET_ALL)
#                 continue
#
#             if type(mouth_crop_array) == bool:
#                 print(Back.RED + 'discard', Style.RESET_ALL)
#             else:
#                 processed_videos+=1
#                 if not os.path.exists(mouth_crops_dir + speaker_name + '/' + video_name + '.npy'):
#                     np.save(mouth_crops_dir + speaker_name + '/' + video_name + '.npy', mouth_crop_array)
#                 print(Back.GREEN + 'processed', Style.RESET_ALL)
#
#             print(processed_videos, '/', total_videos, ' : ', int(processed_videos * 100 / total_videos), '%')
#
#     explored_dirs.append(dir)

# print('TOTAL VIDEOS : {} \n PROCESSED VIDEOS : {}'.format(total_videos, processed_videos))

def rename():
    for speaker in os.listdir(mouth_crops_dir):
        speaker_name = speaker
        speaker = os.path.join(mouth_crops_dir, speaker_name)

        for file in os.listdir(speaker):
            file_name = file
            if '.mp4' in file_name:
                correct_file_name = file_name[:file_name.find('.mp4')] + '.npy'
                print(correct_file_name)
                file = os.path.join(speaker, file_name)

                if 'DS_Store' not in file:
                    os.rename(file, os.path.join(speaker, correct_file_name))

rename()
