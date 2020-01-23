import numpy as np
from matplotlib import patheffects as path_effects, pyplot as plt
from skvideo.io import vread
import dlib
from imutils import face_utils
import cv2

predictor_path = '/Users/padmanabhankrishnamurthy/PycharmProjects/helen_v2p/data/shape_predictor_68_face_landmarks.dat' #dlib face-landmark predictor; acts upon faces detected by dlib face detector

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

def visualize(video:str, subtitle:str):
    video = vread(video)

    ax = plt.subplot(111)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    plt.axis('off')

    text = plt.text(0.5, 0.1, "", ha='center', va='center', transform=ax.transAxes, fontdict={'fontsize': 14, 'color': 'yellow', 'fontweight': 500})
    text.set_path_effects([path_effects.Stroke(linewidth=3, foreground='black'), path_effects.Normal()])

    subs = subtitle.split()
    inc = max(len(video) / (len(subs) + 1), 0.01)

    img = None

    for i, frame in enumerate(video):
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # detected = detector(gray, 1)
        # shape = predictor(gray,detected[0])  # detected[0] ensures only first detected face is chosen for further processing
        # shape = face_utils.shape_to_np(shape)  # shape predictor outputs 68 coordinate pairs, each responsible for a particular landmark
        # mouth_points = shape[48:68]
        # hull = cv2.convexHull(mouth_points)  # draw convex hull around mouth
        # frame = cv2.drawContours(frame, [hull], -1, (100,255,0), 3)
        # cv2.imshow('test', frame)
        # cv2.waitKey(10)


        sub = " ".join(subs[:int(i / inc)])
        text.set_text(sub)

        if img is None:
            img = plt.imshow(frame)
        else:
            img.set_data(frame)

        plt.pause(1/250)

    plt.show()

# visualize('/Users/padmanabhankrishnamurthy/Downloads/opening_vid.mp4', 'almost five hundred million people')

