"""
@author sourabhxiii
"""

import keras
from keras.models import Model, Sequential, load_model
import keras.applications as applications
from keras.applications.mobilenet import MobileNet
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization, Input
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Concatenate, Reshape, LSTM
from keras.layers import Bidirectional
from keras import backend as K

class TLModel:
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape

    def get_model(self):
        with K.name_scope('input'):
            input_c = Input(shape=self.input_shape)
        with K.name_scope('base_model'):
            base_model = applications.VGG16(input_tensor=input_c, input_shape=self.input_shape
                , weights='imagenet', include_top=False)
            q = base_model.output
        with K.name_scope('common_path'):
            xy = Flatten()(q)
            xy = Dense(256, activation='sigmoid', name='Dense1')(xy)
            xy = BatchNormalization()(xy)

            xy = Dense(256, activation='sigmoid', name='Dense2')(xy)
            xy = BatchNormalization()(xy)

            xy = Dense(256, activation='sigmoid', name='Dense3')(xy)
            xy = BatchNormalization()(xy)

            xy = Dense(256, activation='sigmoid', name='Dense4')(xy)
            xy = Dropout(0.5)(xy)
            xy = BatchNormalization()(xy)

            xy = Dense(128, activation='sigmoid', name='Dense5')(xy)
            xy = Dropout(0.5)(xy)
            xy = BatchNormalization()(xy)

            xy = Dense(64, activation='sigmoid', name='Dense6')(xy)
            xy = Dropout(0.35)(xy)
            xy = BatchNormalization()(xy)

            xy = Dense(32, activation='sigmoid', name='Dense7')(xy)
            xy = Dropout(0.35)(xy)
            xy = BatchNormalization()(xy)

        with K.name_scope('prediction'):
            preds = Dense(self.output_shape, activation='softmax', name='DenseP')(xy)
        
        # freeze the base layers:
        for _, layer in enumerate(base_model.layers):
            layer.trainable = False

        model = Model(input_c, preds)
        print(model.summary())
        return model
        
class LSTMModel:
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape

    def get_model(self):
        with K.name_scope('input'):
            input_c = Input(shape=self.input_shape)
        with K.name_scope('l_path'):
            x = Reshape(target_shape=(self.input_shape[0], -1))(input_c)
            x = Bidirectional(LSTM(128, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))(x)
            x = Bidirectional(LSTM(128, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))(x)
            x = Bidirectional(LSTM(128, dropout=0.2, recurrent_dropout=0.2, return_sequences=False))(x)
        with K.name_scope('d_path'):
            xy = Dense(128, activation='sigmoid', name='Dense1')(x)
            xy = Dense(64, activation='sigmoid', name='Dense2')(xy)
            xy = Dense(32, activation='sigmoid', name='Dense3')(xy)

        with K.name_scope('prediction'):
            preds = Dense(self.output_shape, activation='softmax', name='DenseP')(xy)

        model = Model(input_c, preds)
        print(model.summary())
        return model
        

class BestModel:
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape

    def get_model(self):
        model = Sequential()
        model.add(Conv2D(64, (3, 3), padding='same',
                        input_shape=self.input_shape))
        model.add(Activation('relu'))
        model.add(Conv2D(64, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Conv2D(32, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(Conv2D(32, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dropout(0.7))        
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dropout(0.7))
        model.add(Dense(256))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(256))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(128))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(128))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(64))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.output_shape, activation='softmax'))

        print(model.summary())
        return model
