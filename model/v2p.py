from keras import backend as k
from keras.models import Model
from keras.layers import Input
from keras.backend import temporal_padding
from keras.layers import ZeroPadding3D
from keras.layers import Conv3D
from keras.layers import BatchNormalization
from keras.layers import Activation
from keras.layers import MaxPooling3D
from keras.layers import TimeDistributed, Flatten
from keras.layers import Bidirectional
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Lambda
from keras.backend import ctc_batch_cost
from keras.optimizers import Adam
from keras.utils import print_summary

class v2p():
    def __init__(self, frames:int, channels:int, height:int, width:int, max_seq_length:int, output_size:int):

        self.input_layer = Input(shape=(frames, width, height, channels))
        #print(self.input_layer.shape)

        # self.tmp_pad1 = temporal_padding(x=self.input_layer, padding=(1,1))
        self.pad1 = ZeroPadding3D((1,0,0))(self.input_layer)
        self.conv1 = Conv3D(filters=64, kernel_size=(3,3,3), strides=(1,2,2))(self.pad1)
        self.norm1 = BatchNormalization()(self.conv1)
        self.act1 = Activation(activation='relu')(self.norm1)
        self.pool1 = MaxPooling3D(pool_size=(1,2,2), strides=(1,2,2))(self.act1)
        #print(self.pool1.shape)

        # self.tmp_pad2 = temporal_padding(x=self.pool1, padding=(1, 1))
        self.pad2 = ZeroPadding3D((1,0,0))(self.pool1)
        self.conv2 = Conv3D(filters=128, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu')(self.pad2)
        self.norm2 = BatchNormalization()(self.conv2)
        self.act2 = Activation(activation='relu')(self.norm2)
        self.pool2 = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2))(self.act2)
        #print(self.pool2.shape)

        # self.tmp_pad3 = temporal_padding(x=self.pool2, padding=(1, 1))
        self.pad3 = ZeroPadding3D((1, 0, 0))(self.pool2)
        self.conv3 = Conv3D(filters=256, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu')(self.pad3)
        self.norm3 = BatchNormalization()(self.conv3)
        self.act3 = Activation(activation='relu')(self.norm3)
        self.pool3 = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2))(self.act3)
        #print(self.pool3.shape)

        # self.tmp_pad4 = temporal_padding(x=self.pool3, padding=(1, 1))
        self.pad4 = ZeroPadding3D((1,0,0))(self.pool3)
        self.conv4 = Conv3D(filters=512, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu')(self.pad4)
        self.norm4 = BatchNormalization()(self.conv4)
        self.act4 = Activation(activation='relu')(self.norm4)
        #print(self.act4.shape)

        # self.tmp_pad5 = temporal_padding(x=self.act4, padding=(1, 1))
        self.pad5 = ZeroPadding3D((1,0,0))(self.act4)
        self.conv5 = Conv3D(filters=512, kernel_size=(3, 3, 3), strides=(1, 1, 1))(self.pad5)
        self.norm5 = BatchNormalization()(self.conv5)
        self.act5 = Activation(activation='relu')(self.norm5)
        self.pool5 = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 1, 1))(self.act5)
        #print(self.pool5.shape)

        self.time_dist1 = TimeDistributed(Flatten())(self.pool5)
        #print(self.time_dist1.shape)

        self.bilstm6 = Bidirectional(LSTM(768, activation='relu', return_sequences=True))(self.time_dist1)
        self.act6 = Activation(activation='relu')(self.bilstm6)
        #print(self.act6.shape)

        self.bilstm7 = Bidirectional(LSTM(768, activation='relu', return_sequences=True))(self.act6)
        self.act7 = Activation(activation='relu')(self.bilstm7)
        #print(self.act7.shape)

        self.bilstm8 = Bidirectional(LSTM(768, activation='relu', return_sequences=True))(self.act7)
        self.act8 = Activation(activation='relu')(self.bilstm8)
        #print(self.act8.shape)

        self.fc1 = Dense(768, activation='relu')(self.bilstm8)
        #print(self.fc1.shape)

        self.y_pred = Dense(output_size, activation='softmax')(self.fc1)

        self.input_labels = Input(shape=[max_seq_length])
        self.input_length = Input(shape=[1], dtype='int64')
        self.label_length = Input(shape=[1], dtype='int64')

        # self.loss_out = ctc_batch_cost(self.input_labels, self.y_pred, self.input_length, self.label_length)
        self.loss_out = Lambda(self.ctc_lambda_function, output_shape=(1,))([self.input_labels, self.y_pred, self.input_length, self.label_length])

        self.model = Model(inputs=[self.input_layer, self.input_labels, self.input_length, self.label_length], outputs=self.loss_out)
        print_summary(self.model, line_length=250)

    def ctc_lambda_function(self, args):
        return ctc_batch_cost(args[0], args[1], args[2], args[3])


# print(k.image_data_format())
model = v2p(75, 3, 128, 128, 116, 68+1)

