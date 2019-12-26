import numpy as np
from helpers.labels import get_labels
from model.v2p import v2p

#test training

#all data values need to be converted to lists to add an extra dimension at the start == batch

x_data = '/Users/padmanabhankrishnamurthy/Desktop/lrs3/mouth_crops/0Fi83BHQsMA/00006_128.npy'
x_data = np.load(x_data)
x_data = x_data.astype(np.float32) / 255
x_data = [x_data]
x_data = np.array(x_data)


y_data = open('/Users/padmanabhankrishnamurthy/Desktop/lrs3/test-3/0Fi83BHQsMA/00006_phoneme.txt', 'r').readline()
y_data, og_label_length = get_labels(y_data)
y_data = np.array([y_data])

input_length = np.array([x_data.shape[1]])
label_length = np.array([og_label_length])

V2P = v2p(x_data.shape[1], 3, 128, 128, 116, 68+1)
V2P.compile_model()
# print_summary(V2P.model, line_length=125)

# print('X_DATA: ', x_data.shape)
# print('Y_DATA: ', y_data.shape)
# print('OG_LABEL_LENGTH: ', label_length.shape)

V2P.model.fit(x={'input_layer':x_data, 'labels_layer':y_data, 'input_length_layer':input_length, 'label_length_layer':label_length}, y={'ctc_layer':np.zeros([len(x_data)])}, epochs=10)
