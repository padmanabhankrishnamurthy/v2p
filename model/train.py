import numpy as np
import os
from helpers.labels import get_labels
from model.v2p import v2p
from keras.preprocessing.sequence import pad_sequences

#test training
np.random.seed(7)

#all data values need to be converted to lists to add an extra dimension at the start == batch

x_data = []
y_data = []
input_length = []
label_length = []

frame_lengths = []

samples = 10
frame_count = 75

mouth_crops_dir = '/Users/padmanabhankrishnamurthy/Desktop/lrs3/mouth_crops'
labels_dir = '/Users/padmanabhankrishnamurthy/Desktop/lrs3/test-3'
weights_path = '/Users/padmanabhankrishnamurthy/Desktop/lrs3/weights'

ctr = 0
for speaker in os.listdir(mouth_crops_dir):
    speaker_name = speaker
    speaker = os.path.join(mouth_crops_dir, speaker_name)

    if os.path.isdir(speaker):
        for file in os.listdir(speaker):
            if ctr == samples:
                break
            file_name = file
            file = os.path.join(speaker, file_name)

            if '_128.npy' in file:
                print(speaker_name, file_name, end = ' ')
                ctr+=1
                #get video
                video = np.load(file)

                frame_lengths.append(len(video))
                print(len(video))

                video = video.astype(np.float32) / 255
                x_data.append(video)

                #get label
                phoneme_sentence = open(os.path.join(labels_dir, speaker_name, file_name[:file_name.find('_128')] + '_phoneme.txt'), 'r').readline()
                label, unpadded_length = get_labels(phoneme_sentence)
                y_data.append(label)

                #input length and label length
                input_length.append(min(len(video), frame_count))
                label_length.append(unpadded_length)


x_data = pad_sequences(x_data, maxlen=frame_count, value=-1)
x_data = np.array(x_data)[:samples]

y_data = np.array(y_data)[:samples]

input_length = np.array(input_length)[:samples]

label_length = np.array(label_length)[:samples]

def print_shapes(index = 0):
    for array in [x_data, y_data, input_length, label_length]:
        if index != 0:
            if len(array.shape) > 1:
                print(array[index].shape)
            else:
                print(array[index])
        else:
            print(array.shape)

print_shapes(8)

V2P = v2p(frame_count, 3, 128, 128, 116, 68+1)
V2P.compile_model()
# print_summary(V2P.model, line_length=125)
V2P.model.fit(x={'input_layer':x_data, 'labels_layer':y_data, 'input_length_layer':input_length, 'label_length_layer':label_length}, shuffle=False, y={'ctc_layer':np.zeros([len(x_data)])}, epochs=10, batch_size=10)
# V2P.model.save(filepath=os.path.join(weights_path, 'test_model_2.hdf5'))