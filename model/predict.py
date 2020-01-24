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
from visualization.visualization import visualize
from dataset_generation.extract_cmu_phonemes import phoneme_to_word
from dataset_generation.extract_mouth_crops import extract_mouth_crop

weights_path = '/Users/padmanabhankrishnamurthy/Desktop/lrs3/weights'
weights = '[BEST]_11_Jan_21_26.hdf5'
# dataset_dir = '/Users/padmanabhankrishnamurthy/Desktop/lrs3/test-3'
labels_dir = '/Users/padmanabhankrishnamurthy/Desktop/lrs3/labels_only'
mouth_crops_dir = '/Users/padmanabhankrishnamurthy/Desktop/lrs3/mouth_crops'

vids_dir = '/Users/padmanabhankrishnamurthy/Desktop/s_demo_data/vids'
test_mouth_crops_dir = '/Users/padmanabhankrishnamurthy/Desktop/s_demo_data/test_mouth_crops/pk'
weights_path = '/Users/padmanabhankrishnamurthy/Desktop/s_demo_data/weights'
weights = '23_Jan_19_23.hdf5'

max_frame_count = 91
max_seq_len = 24

print('LOADING MODEL =====>', end = ' ')
v2p = v2p(max_frame_count, 3, 128, 128, max_seq_len, 68 + 1)
v2p = v2p.compile_model()
v2p.model.load_weights(os.path.join(weights_path, weights))
print("MODEL LOADED")
# v2p.model.load_weights('/Users/padmanabhankrishnamurthy/PycharmProjects/helen_v2p/scratch/21_Jan_15_27.hdf5')
# print_summary(v2p.model, line_length=200)

def predict(filepath:str, slice:tuple = None) -> str:
    print('PREDICTING FOR {}'.format(filepath))
    x_data = filepath
    x_data = np.load(x_data)

    if slice:
        x_data = x_data[slice[0]:slice[1]]

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
    decoded = sequence_from_labels(decoded)
    print(decoded)
    return decoded

def predict_batch():
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

                    truth = open(os.path.join(labels_dir, speaker_name, video_name+'_phoneme.txt'), 'r').readline().strip()
                    print(Back.BLUE + speaker_name, ' ', video_name,  ' ', len(video_data), Style.RESET_ALL)
                    print(Back.GREEN + 'Truth:', truth, Style.RESET_ALL)

                    sequence = predict(filepath=video)
                    sequence = sequence_from_labels(sequence)
                    sequence = sequence.replace('sp', ' ').strip()
                    if sequence == truth:
                        print(Back.RED, end='')
                    print(sequence, Style.RESET_ALL, '\n')


def decode_to_regular(sequence):
    sentence = []
    word = []

    for phoneme in sequence.split():
        if phoneme != 'sp':
            word.append(phoneme)
        else:
            sentence.append(word)
            word = []
    sentence.append(word) #last word not appended inside loop coz no sp at end of sequence
    # print("SENTENCE: ", sentence)

    decoded_to_words = []
    for phoneme_word in sentence:
        word = phoneme_to_word(phoneme_word)
        if word:
            decoded_to_words.append(word)
        else:
            decoded_to_words.append(''.join(phoneme_word))
    print(decoded_to_words)
    return decoded_to_words

vid = '/Users/padmanabhankrishnamurthy/Downloads/pk_gmsf.mp4'
# extract_mouth_crop(vid, save_path=os.path.join(test_mouth_crops_dir, 'pk_gmsf_128.npy'))
# sequence = predict(os.path.join(test_mouth_crops_dir, 'pk_gmsf_128.npy'))
# decoded = decode_to_regular(sequence)
visualize(vid, ' '.join(['good', 'morning', 'san', 'FRAE0SKOW0'])) #alternatively, visualize sequence
