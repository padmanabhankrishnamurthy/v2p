from model.v2p import v2p
import os
from keras.backend import ctc_decode
from keras.utils import print_summary
import numpy as np
from pprint import pprint
from keras import backend as k
from helpers.labels import sequence_from_labels

weights_path = '/Users/padmanabhankrishnamurthy/Desktop/lrs3/weights'
weights = 'u27_frames_weights_04_Jan_13_22.hdf5'
frame_count = 22

v2p = v2p(frame_count, 3, 128, 128, 116, 68+1)
v2p = v2p.compile_model()
v2p.model.load_weights(os.path.join(weights_path, weights))
# print_summary(v2p.model, line_length=200)

def predict(filepath):

    x_data = filepath
    x_data = np.load(x_data)
    x_data = x_data.astype(np.float32) / 255
    x_data = np.array([x_data])
    # print(x_data.shape)

    y_pred = v2p.predict(x_data)
    # for batch in y_pred:
    #     for vector in batch:
    #         print(vector)
    #         print('=====')

    decoded = ctc_decode(y_pred, [frame_count], top_paths=3, beam_width=200)
    for path in decoded[0]:
        evaluated = path.eval(session = k.get_session())
        print(evaluated)
        sequence = sequence_from_labels(evaluated[0])
        print(sequence)
    # return sequence

filepath = '/Users/padmanabhankrishnamurthy/Desktop/lrs3/mouth_crops/iPE2SiCCo0w/00023_128.npy'
sequence = predict(filepath=filepath)