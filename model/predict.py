from model.v2p import v2p
import os
from keras.backend import ctc_decode
from keras.utils import print_summary
import numpy as np
from pprint import pprint
from keras import backend as k
from helpers.labels import sequence_from_labels
from keras.preprocessing.sequence import pad_sequences
from colorama import Back, Style

weights_path = '/Users/padmanabhankrishnamurthy/Desktop/lrs3/weights'
weights = '11_Jan_21_26.hdf5'
dataset_dir = '/Users/padmanabhankrishnamurthy/Desktop/lrs3/test-3'
mouth_crops_dir = '/Users/padmanabhankrishnamurthy/Desktop/lrs3/mouth_crops'

max_frame_count = 23
max_seq_len = 19

v2p = v2p(max_frame_count, 3, 128, 128, max_seq_len, 68 + 1)
v2p = v2p.compile_model()
v2p.model.load_weights(os.path.join(weights_path, weights))
# print_summary(v2p.model, line_length=200)

def predict(filepath):

    x_data = filepath
    x_data = np.load(x_data)
    x_data = x_data.astype(np.float32) / 255
    x_data = np.array([x_data])
    print(x_data.shape, end = ' ==> ')
    x_data = pad_sequences(x_data, max_frame_count, value=-1)
    print(x_data.shape)

    y_pred = v2p.predict(x_data)

    encoded = []
    for batch in y_pred:
        for vector in batch:
            encoded.append(np.argmax(vector))
    # print('ENCODED: ', encoded)
    encoded = [element for element in encoded if element != 68]
    # print(sequence_from_labels(encoded))

    decoded = ctc_decode(y_pred, [max_frame_count], greedy=True, beam_width=200, top_paths=1)
    decoded = [path.eval(session=k.get_session()) for path in decoded[0]]
    decoded = decoded[0][0]
    print(decoded)
    print(sequence_from_labels(decoded), '\n')
    return decoded

# speaker = 'Zd71719SG8Y'
# vid = '00001'
# truth = open(os.path.join(dataset_dir, speaker, vid+'_phoneme.txt'), 'r').readline()
# print('TRUTH: ', truth, '\n', 'PREDICTED:\n')
#
# filepath = '/Users/padmanabhankrishnamurthy/Desktop/lrs3/mouth_crops/{}/{}_128.npy'.format(speaker, vid)
# sequence = predict(filepath=filepath)

for speaker_name in os.listdir(mouth_crops_dir):
    speaker = os.path.join(mouth_crops_dir, speaker_name)

    if os.path.isdir(speaker):

        for video_name in os.listdir(speaker):
            video = os.path.join(speaker, video_name)
            video_name = video_name[:video_name.find('_128')]

            if '_128.npy' in video:
                video_data = np.load(video)
                if len(video_data) < 20 or len(video_data) > max_frame_count:
                    continue
                truth = open(os.path.join(dataset_dir, speaker_name, video_name+'_phoneme.txt'), 'r').readline()
                print(Back.BLUE + speaker_name, ' ', video_name,  ' ', len(video_data), Style.RESET_ALL)
                print(Back.GREEN + 'Truth: ', truth, Style.RESET_ALL)
                sequence = predict(filepath=video)



