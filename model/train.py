import numpy as np
import os
from helpers.labels import get_labels
from model.v2p import v2p
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import ModelCheckpoint, TensorBoard
from scipy.stats import mode
from pprint import pprint
from datetime import datetime
from keras.models import load_model


#test training
np.random.seed(7)

#all data values need to be converted to lists to add an extra dimension at the start == batch

x_data = []
y_data = []
input_length = []
label_length = []

frame_lengths = []

samples = 64
max_frame_count = 23
max_seq_length = 19

mouth_crops_dir = '/Users/padmanabhankrishnamurthy/Desktop/lrs3/mouth_crops'
labels_dir = '/Users/padmanabhankrishnamurthy/Desktop/lrs3/test-3'
weights_path = '/Users/padmanabhankrishnamurthy/Desktop/lrs3/weights'
tensorboard_log_dir = '/Users/padmanabhankrishnamurthy/PycharmProjects/helen_v2p/data/tensorboard_logs/'

ctr = 0
for speaker in os.listdir(mouth_crops_dir):
    speaker_name = speaker
    speaker = os.path.join(mouth_crops_dir, speaker_name)

    if os.path.isdir(speaker):
        for file in os.listdir(speaker):
            if ctr == samples: #comment out do nothing flag to take only 'samples' samples; comment out 'break' to take all samples satisfying particular conditions set out below
                do_nothing_flag = 1
                # break

            file_name = file
            file = os.path.join(speaker, file_name)

            if '_128.npy' in file:

                #get video
                video = np.load(file)

                if len(video) > max_frame_count or len(video) < 20: # ***** best results yet, comment out other constraints and use only this
                    # do_nothing_flag = 1
                    continue

                if len(video) != max_frame_count: #choose videos strictly equal to max_frame_count
                    do_nothing_flag = 1
                    # continue

                if len(video) > 100: #choosing only videos lesser than 100 frames
                    do_nothing_flag = 1
                    # continue

                frame_lengths.append(len(video))
                video = video.astype(np.float32) / 255

                #get label
                phoneme_sentence = open(os.path.join(labels_dir, speaker_name, file_name[:file_name.find('_128')] + '_phoneme.txt'), 'r').readline()
                label, unpadded_length = get_labels(phoneme_sentence)

                # if unpadded_length < 13:
                #     continue

                ctr+=1
                print(ctr, speaker_name, file_name, len(video), unpadded_length)

                #add each video thrice - desparate times etc.

                for i in range(3):
                    # ctr+=1
                    #x and y data
                    x_data.append(video)
                    y_data.append(label)

                    #input length and label length
                    # input_length.append(min(len(video), max_frame_count))
                    input_length.append(len(video))
                    label_length.append(unpadded_length)

print('\n', np.max(frame_lengths), np.mean(frame_lengths), np.std(frame_lengths), mode(frame_lengths))
print(np.max(label_length), np.mean(label_length), np.std(label_length), mode(label_length), '\n')


samples = ctr
max_frame_count = np.max(frame_lengths)

x_data = pad_sequences(x_data, maxlen=max_frame_count, value=-1)
x_data = np.array(x_data)[:samples]

y_data = np.array(y_data)[:samples].astype(np.int32)

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

print_shapes()
# print('\n', y_data)

V2P = v2p(max_frame_count, 3, 128, 128, max_seq_length, 68 + 1)
V2P = V2P.compile_model()
V2P.model.load_weights('/Users/padmanabhankrishnamurthy/Desktop/lrs3/weights/11_Jan_21_26.hdf5')
# print_summary(V2P.model, line_length=125)
tensorboard = TensorBoard(log_dir=tensorboard_log_dir)
model_checkpoint = ModelCheckpoint(filepath=os.path.join(weights_path, '{}.hdf5'.format(datetime.now().strftime('%d_%b_%H_%M'))), monitor='loss', save_best_only=True)
V2P.model.fit(x={'input_layer':x_data, 'labels_layer':y_data, 'input_length_layer':input_length, 'label_length_layer':label_length}, shuffle=False, y={'ctc_layer':np.zeros([len(x_data)])}, epochs=300, batch_size=8, callbacks=[model_checkpoint, tensorboard])