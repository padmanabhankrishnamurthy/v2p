from model.v2p import v2p
import os
from keras.backend import ctc_decode
from keras.utils import print_summary
import numpy as np
from pprint import pprint
from keras import backend as k
from helpers.labels import sequence_from_labels

weights_path = '/Users/padmanabhankrishnamurthy/Desktop/lrs3/weights'
weights = '0Fi_006_weights.hdf5'
frame_count = 23

v2p = v2p(frame_count, 3, 128, 128, 116, 68+1)
v2p = v2p.compile_model()
v2p.model.load_weights(os.path.join(weights_path, weights))
# print_summary(v2p.model, line_length=250)

def predict(filepath):

    x_data = filepath
    x_data = np.load(x_data)
    x_data = x_data.astype(np.float32) / 255
    x_data = np.array([x_data])
    # print(x_data.shape)

    y_pred = v2p.predict(x_data)
    decoded = ctc_decode(y_pred, [23])[0]
    decoded = decoded[0].eval(session = k.get_session())
    print(decoded[0])
    sequence = sequence_from_labels(decoded[0])
    print(sequence)
    return sequence

filepath = '/Users/padmanabhankrishnamurthy/Desktop/lrs3/mouth_crops/0Fi83BHQsMA/00006_128.npy'
sequence = predict(filepath=filepath)