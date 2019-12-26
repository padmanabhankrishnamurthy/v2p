import numpy as np
import os
from helpers.labels import get_labels
from model.v2p import v2p
from keras.preprocessing.sequence import pad_sequences

#test training

#all data values need to be converted to lists to add an extra dimension at the start == batch

x_data = []
y_data = []
input_length = []
label_length = []

frame_lengths = []

mouth_crops_dir = '/Users/padmanabhankrishnamurthy/Desktop/lrs3/mouth_crops'
labels_dir = '/Users/padmanabhankrishnamurthy/Desktop/lrs3/test-3'
weights_path = '/Users/padmanabhankrishnamurthy/Desktop/lrs3/weights'

for speaker in os.listdir(mouth_crops_dir):

    speaker_name = speaker
    speaker = os.path.join(mouth_crops_dir, speaker_name)

    if os.path.isdir(speaker):
        for file in os.listdir(speaker):
            file_name = file
            file = os.path.join(speaker, file_name)

            if '_128.npy' in file:
                #get video
                video = np.load(file)

                frame_lengths.append(video.shape[0])

                video = video.astype(np.float32) / 255
                x_data.append(video)

                #get label
                phoneme_sentence = open(os.path.join(labels_dir, speaker_name, file_name[:file_name.find('_128')] + '_phoneme.txt'), 'r').readline()
                label, unpadded_length = get_labels(phoneme_sentence)
                y_data.append(label)

                #input length and label length
                input_length.append(video.shape[0])
                label_length.append(unpadded_length)

# frame_lengths = sorted(frame_lengths, reverse=True)
# for frame_length in frame_lengths:
#     print(frame_length, ' : ', frame_lengths.count(frame_length))

x_data = pad_sequences(x_data, maxlen=75, value=-1)
x_data = np.array(x_data)[:1]
y_data = np.array(y_data)[:1]
input_length = np.array(input_length)[:1]
label_length = np.array(label_length)[:1]

V2P = v2p(75, 3, 128, 128, 116, 68+1)
V2P.compile_model()
# # print_summary(V2P.model, line_length=125)
V2P.model.fit(x={'input_layer':x_data, 'labels_layer':y_data, 'input_length_layer':input_length, 'label_length_layer':label_length}, y={'ctc_layer':np.zeros([len(x_data)])}, epochs=50, batch_size=2)
V2P.model.save(filepath=os.path.join(weights_path, 'test_model.hdf5'))