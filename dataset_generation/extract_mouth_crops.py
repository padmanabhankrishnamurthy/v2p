import dlib
from imutils import face_utils
import cv2
import skvideo.io
import numpy as np
from pprint import pprint


frame_annotations_dir = '/Users/padmanabhankrishnamurthy/Desktop/lrs3/lrs3_frame_annotations/test'
sample_video = '/Users/padmanabhankrishnamurthy/Desktop/lrs3/test-3/0ZfSOArXbGQ/00003.mp4'

predictor_path = '/Users/padmanabhankrishnamurthy/PycharmProjects/helen_v2p/data/shape_predictor_68_face_landmarks.dat' #dlib face-landmark predictor; acts upon faces detected by dlib face detector

def extract_mouth_crop(show_crop:bool, visualize:bool):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)

    video_data = skvideo.io.vread(sample_video)

    mouth_data = []

    for frame in video_data:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detected = detector(gray, 1)
        frame_copy = frame #save original frame
        if(len(detected) <= 0):
            print('NO FACE DETECTED')
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
                cv2.waitKey(100)

            if(visualize): #only to visualize mouth crops
                hull = cv2.convexHull(mouth_points) #draw convex hull around mouth
                cv2.drawContours(frame_copy, [hull], -1, (19, 199, 109), -1)
                cv2.imshow('window', frame_copy)
                cv2.waitKey(100)

extract_mouth_crop(True, False)