from keras import backend as k
from keras.models import Model
from keras.layers import Input
from keras.layers import Masking
from keras.layers import ZeroPadding3D
from keras.layers import Conv3D
from keras.layers import BatchNormalization
from keras.layers import Activation
from keras.layers import MaxPooling3D
from keras.layers import GlobalAveragePooling3D
from keras.layers import TimeDistributed, Flatten
from keras.layers import Bidirectional
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Lambda
from keras.backend import ctc_batch_cost
from keras.optimizers import Adam
from keras_contrib.layers.normalization.groupnormalization import GroupNormalization


class v2p():
    def __init__(self, frames:int, channels:int, height:int, width:int, max_seq_length:int, output_size:int):

        self.masking_layer = Masking(mask_value=-1)
        self.input_layer = Input(shape=(frames, width, height, channels), name='input_layer')
        # self.input_layer = Input(shape=(None, width, height, channels), name='input_layer')
        #print(self.input_layer.shape)

        # self.tmp_pad1 = temporal_padding(x=self.input_layer, padding=(1,1))
        self.pad1 = ZeroPadding3D((1,0,0))(self.input_layer)
        self.conv1 = Conv3D(filters=64, kernel_size=(3,3,3), strides=(1,2,2), kernel_initializer='he_normal')(self.pad1)#(self.input_layer)
        self.norm1 = BatchNormalization()(self.conv1)
        # self.norm1 = GroupNormalization(groups=2)(self.conv1)
        self.act1 = Activation(activation='relu')(self.norm1)
        self.pool1 = MaxPooling3D(pool_size=(1,2,2), strides=(1,2,2))(self.act1)
        #print(self.pool1.shape)

        # self.tmp_pad2 = temporal_padding(x=self.pool1, padding=(1, 1))
        self.pad2 = ZeroPadding3D((1,0,0))(self.pool1)
        self.conv2 = Conv3D(filters=128, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu', kernel_initializer='he_normal')(self.pad2)
        self.norm2 = BatchNormalization()(self.conv2)
        # self.norm2 = GroupNormalization(groups=2)(self.conv2)
        self.act2 = Activation(activation='relu')(self.norm2)
        self.pool2 = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2))(self.act2)
        #print(self.pool2.shape)

        # self.tmp_pad3 = temporal_padding(x=self.pool2, padding=(1, 1))
        self.pad3 = ZeroPadding3D((1, 0, 0))(self.pool2)
        self.conv3 = Conv3D(filters=256, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu', kernel_initializer='he_normal')(self.pad3)
        self.norm3 = BatchNormalization()(self.conv3)
        # self.norm3 = GroupNormalization(groups=2)(self.conv3)
        self.act3 = Activation(activation='relu')(self.norm3)
        self.pool3 = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2))(self.act3)
        #print(self.pool3.shape)

        # self.tmp_pad4 = temporal_padding(x=self.pool3, padding=(1, 1))
        self.pad4 = ZeroPadding3D((1,0,0))(self.pool3)
        self.conv4 = Conv3D(filters=512, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu', kernel_initializer='he_normal')(self.pad4)
        self.norm4 = BatchNormalization()(self.conv4)
        # self.norm4 = GroupNormalization(groups=2)(self.conv4)
        self.act4 = Activation(activation='relu')(self.norm4)
        #print(self.act4.shape)

        # self.tmp_pad5 = temporal_padding(x=self.act4, padding=(1, 1))
        self.pad5 = ZeroPadding3D((1,0,0))(self.act4)
        self.conv5 = Conv3D(filters=512, kernel_size=(3, 3, 3), strides=(1, 1, 1), kernel_initializer='he_normal')(self.pad5)
        self.norm5 = BatchNormalization()(self.conv5)
        # self.norm5 = GroupNormalization(groups=2)(self.conv5)
        self.act5 = Activation(activation='relu')(self.norm5)
        self.pool5 = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 1, 1))(self.act5)
        #print(self.pool5.shape)

        self.gap1 = GlobalAveragePooling3D()

        self.time_dist1 = TimeDistributed(Flatten())(self.pool5)
        #print(self.time_dist1.shape)

        self.bilstm6 = Bidirectional(LSTM(768, activation='relu', return_sequences=True, kernel_initializer='orthogonal'))(self.time_dist1)
        self.norm6 = BatchNormalization()(self.bilstm6)
        # self.norm6 = GroupNormalization(groups=2)(self.bilstm6)
        self.act6 = Activation(activation='relu')(self.norm6)
        #print(self.act6.shape)

        self.bilstm7 = Bidirectional(LSTM(768, activation='relu', return_sequences=True, kernel_initializer='orthogonal'))(self.act6)
        self.norm7 = BatchNormalization()(self.bilstm7)
        # self.norm7 = GroupNormalization(groups=2)(self.bilstm7)
        self.act7 = Activation(activation='relu')(self.norm7)
        #print(self.act7.shape)

        self.bilstm8 = Bidirectional(LSTM(768, activation='relu', return_sequences=True, kernel_initializer='orthogonal'))(self.act7)
        self.norm8 = BatchNormalization()(self.bilstm8)
        # self.norm8 = GroupNormalization(groups=2)(self.bilstm8)
        self.act8 = Activation(activation='relu')(self.norm8)
        #print(self.act8.shape)

        self.fc1 = Dense(768)(self.act8)
        self.norm9 = BatchNormalization()(self.fc1)
        self.act9 = Activation(activation='relu')(self.norm9)
        # self.norm9 = GroupNormalization(groups=2)(self.fc1)
        #print(self.fc1.shape)

        self.y_pred = Dense(output_size, activation='softmax')(self.act9)

        self.input_labels = Input(shape=[max_seq_length], name='labels_layer')
        self.input_length = Input(shape=[1], dtype='int64', name='input_length_layer')
        self.label_length = Input(shape=[1], dtype='int64', name='label_length_layer')

        #keras doesn't support custom loss functions with more than 2 params (y_pred and y_true) => implement CTC loss as a custom keras Lambda layer
        self.loss_out = Lambda(self.ctc_lambda_function, output_shape=(1,), name='ctc_layer')([self.input_labels, self.y_pred, self.input_length, self.label_length])

        self.model = Model(inputs=[self.input_layer, self.input_labels, self.input_length, self.label_length], outputs=self.loss_out)
        # print_summary(self.model, line_length=250)

    def ctc_lambda_function(self, args):
        return ctc_batch_cost(args[0], args[1], args[2], args[3])

    def compile_model(self):
        adam = Adam(lr=1e-4, epsilon=1e-8) #need to play around with learning rate to avoid nan loss
        self.model.compile(loss={'ctc_layer': lambda y_true, y_pred : y_pred}, optimizer=adam)
        return self

    def predict(self, input_batch):
        return self.capture_softmax_output([input_batch, 0])[0]

    @property
    def capture_softmax_output(self):
        return k.function([self.input_layer, k.learning_phase()], [self.y_pred])

# print(k.image_data_format())



