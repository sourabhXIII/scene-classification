import numpy as np
import tensorflow as tf

'''
https://github.com/taki0112/SENet-Tensorflow
https://towardsdatascience.com/squeeze-and-excitation-networks-9ef5e71eacd7

CNNs weights each of its channels equally when creating the output feature maps.
SENets are all about changing this by adding a content aware mechanism to weight each channel adaptively.
'''
class SeNetBlock(tf.keras.Model):
    def __init__(self, out_dim, ratio, layer_name):
        super(SeNetBlock, self).__init__()

        self.output_dim = out_dim
        self.reduction_ratio = ratio
        self.layer_name = layer_name

        self.squeeze = tf.keras.layers.GlobalAveragePooling2D()
        self.dense1 = tf.keras.layers.Dense(int(np.ceil(self.output_dim / self.reduction_ratio))
                        , activation='relu', name=layer_name+'_fully_connected1')
        self.dense2 = tf.keras.layers.Dense(self.output_dim
                        , activation='sigmoid', name=layer_name+'_fully_connected2')

    def call(self, inputs):
        squeeze = self.squeeze(inputs)

        excitation = self.dense1(squeeze)
        excitation = self.dense2(excitation)

        excitation = tf.keras.backend.reshape(excitation, [-1,1,1,self.output_dim])
        scale = inputs * excitation

        return scale



class SeNetModel:
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.channel = 32
        self.reduction_ratio = 4
    
    def squeeze_excitation_layer(self, x, out_dim, layer_name):
        '''
        SE module performs inter-channel weighting.
        '''
        squeeze = tf.keras.layers.GlobalAveragePooling2D()(x)
        
        excitation = tf.keras.layers.Dense(int(np.ceil(out_dim / self.reduction_ratio))
                            , activation='relu', name=layer_name+'_fully_connected1')(squeeze)
        excitation = tf.keras.layers.Dense(out_dim
                        , activation='sigmoid', name=layer_name+'_fully_connected2')(excitation)
        excitation = tf.keras.layers.Reshape((1,1,out_dim))(excitation)
        
        scale = tf.keras.layers.Multiply()([x,excitation])
        
        return scale

    def get_model(self):
        with tf.keras.backend.name_scope('input'):
            input_c = tf.keras.layers.Input(shape=self.input_shape)
        
        x = tf.keras.layers.Conv2D(self.channel, (3, 3), padding='valid', dilation_rate=(2, 2))(input_c)
        x = tf.keras.layers.PReLU()(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Conv2D(self.channel, (3, 3), padding='valid', dilation_rate=(2, 2))(x)
        x = tf.keras.layers.PReLU()(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = self.squeeze_excitation_layer(x, self.channel, 'xconv1')
        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = tf.keras.layers.Dropout(0.25)(x)


        x = tf.keras.layers.Conv2D(self.channel, (3, 3), padding='same', dilation_rate=(2, 2))(x)
        x = tf.keras.layers.PReLU()(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Conv2D(self.channel, (3, 3), padding='same', dilation_rate=(2, 2))(x)
        x = tf.keras.layers.PReLU()(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = self.squeeze_excitation_layer(x, self.channel, 'xconv2')
        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = tf.keras.layers.Dropout(0.25)(x)
        

        x = tf.keras.layers.Conv2D(self.channel, (3, 3), padding='valid', dilation_rate=(2, 2))(x)
        x = tf.keras.layers.PReLU()(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Conv2D(self.channel, (3, 3), padding='valid', dilation_rate=(2, 2))(x)
        x = tf.keras.layers.PReLU()(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = self.squeeze_excitation_layer(x, self.channel, 'xconv3')
        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = tf.keras.layers.Dropout(0.25)(x)

        x = tf.keras.layers.Conv2D(self.channel, (3, 3), padding='same', dilation_rate=(2, 2))(x)
        x = tf.keras.layers.PReLU()(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Conv2D(self.channel, (3, 3), padding='same', dilation_rate=(2, 2))(x)
        x = tf.keras.layers.PReLU()(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = self.squeeze_excitation_layer(x, self.channel, 'xconv4')        
        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = tf.keras.layers.Dropout(0.25)(x)
        
        x = tf.keras.layers.Flatten()(x)

        # ------------------------------------
        '''
        z = tf.keras.layers.Conv2D(self.channel, (5, 5), padding='same')(input_c)
        z = tf.keras.layers.Activation('relu')(z)
        z = tf.keras.layers.BatchNormalization()(z)
        z = tf.keras.layers.Conv2D(self.channel, (5, 5))(z)
        z = tf.keras.layers.Activation('relu')(z)
        z = tf.keras.layers.BatchNormalization()(z)
        z = self.squeeze_excitation_layer(z, self.channel, 'zconv1')
        z = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(z)

        z = tf.keras.layers.Conv2D(self.channel, (5, 5), padding='same')(z)
        z = tf.keras.layers.Activation('relu')(z)
        z = tf.keras.layers.BatchNormalization()(z)
        z = tf.keras.layers.Conv2D(self.channel, (5, 5), padding='same')(z)
        z = tf.keras.layers.Activation('relu')(z)
        z = tf.keras.layers.BatchNormalization()(z)
        z = self.squeeze_excitation_layer(z, self.channel, 'zconv2')
        z = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(z)
        

        z = tf.keras.layers.Conv2D(self.channel, (5, 5), padding='same')(z)
        z = tf.keras.layers.Activation('relu')(z)
        z = tf.keras.layers.BatchNormalization()(z)
        z = tf.keras.layers.Conv2D(self.channel, (5, 5))(z)
        z = tf.keras.layers.Activation('relu')(z)
        z = tf.keras.layers.BatchNormalization()(z)
        z = self.squeeze_excitation_layer(z, self.channel, 'zconv3')
        z = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(z)

        z = tf.keras.layers.Conv2D(self.channel, (5, 5), padding='same')(z)
        z = tf.keras.layers.Activation('relu')(z)
        z = tf.keras.layers.BatchNormalization()(z)
        z = tf.keras.layers.Conv2D(self.channel, (5, 5))(z)
        z = tf.keras.layers.Activation('relu')(z)
        z = tf.keras.layers.BatchNormalization()(z)
        z = self.squeeze_excitation_layer(z, self.channel, 'zconv4')        
        z = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(z)

        z = tf.keras.layers.Flatten()(z)

        y = tf.keras.layers.Concatenate()([x, z])
        '''
        # -------------------------------------------

        y = tf.keras.layers.Dense(512)(x)
        y = tf.keras.layers.Activation('relu')(y)
        y = tf.keras.layers.Dropout(0.7)(y)
        y = tf.keras.layers.BatchNormalization()(y)
        y = tf.keras.layers.Dense(512)(y)
        y = tf.keras.layers.Activation('relu')(y)
        y = tf.keras.layers.Dropout(0.7)(y)
        y = tf.keras.layers.BatchNormalization()(y)
        y = tf.keras.layers.Dense(256)(y)
        y = tf.keras.layers.Activation('relu')(y)
        y = tf.keras.layers.Dropout(0.5)(y)
        y = tf.keras.layers.BatchNormalization()(y)
        y = tf.keras.layers.Dense(256)(y)
        y = tf.keras.layers.Activation('relu')(y)
        y = tf.keras.layers.Dropout(0.5)(y)
        y = tf.keras.layers.BatchNormalization()(y)
        y = tf.keras.layers.Dense(128)(y)
        y = tf.keras.layers.Activation('relu')(y)
        y = tf.keras.layers.Dropout(0.5)(y)
        y = tf.keras.layers.BatchNormalization()(y)
        y = tf.keras.layers.Dense(128)(y)
        y = tf.keras.layers.Activation('relu')(y)
        y = tf.keras.layers.Dropout(0.5)(y)
        y = tf.keras.layers.BatchNormalization()(y)
        y = tf.keras.layers.Dense(64)(y)
        y = tf.keras.layers.Activation('relu')(y)
        y = tf.keras.layers.Dropout(0.5)(y)
        y = tf.keras.layers.BatchNormalization()(y)
        preds = tf.keras.layers.Dense(self.output_shape, activation='softmax')(y)

        model = tf.keras.Model(input_c, preds)
        print(model.summary())
        tf.keras.utils.plot_model(model, to_file='senet_model.png', show_shapes=True)
        return model
