"""
@author sourabhxiii
"""

import os
import pandas as pd
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

from utility import Utility as mutil
import datagen as mdgen
from snetmodel import SeNetModel
import keras
from keras.applications.vgg16 import preprocess_input
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from sklearn.model_selection import train_test_split
import keras.optimizers as optimizers

import tensorflow as tf
# tf.enable_eager_execution()

TRAIN_FOLDER = 'train-scene_classification'+os.sep+'train'
TEST_FOLDER = 'train-scene_classification'+os.sep+'test'

BATCH_SIZE = 64
IMG_HEIGHT = 70
IMG_WIDTH = 70
CHANNELS = 3
N_CLASSES = 6
EPOCHS = 500

df = pd.read_csv('train-scene_classification'+os.sep+'train.csv')
test_df = pd.read_csv('train-scene_classification'+os.sep+'test.csv')

# set seed
mutil.set_seeds()

# get train and val df
train_df, val_df = mutil.get_train_val_df(df, 0.1)

# training data generator 
train_generator = mdgen.data_generator(train_df, N_CLASSES, BATCH_SIZE
                   , IMG_HEIGHT, IMG_WIDTH, CHANNELS)

# validation data generator 
valid_generator = mdgen.data_generator(val_df, N_CLASSES, BATCH_SIZE
                   , IMG_HEIGHT, IMG_WIDTH, CHANNELS, is_validation_data=True)

datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    zoom_range=0.3
    , brightness_range=(1, 1.3)
    # , preprocessing_function=mutil.get_random_eraser(v_l=0, v_h=1, pixel_level=True)
    )

# train_generator = mdgen.MixupImageDataGenerator(datagen, train_df, BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, subset="training")
# valid_generator = mdgen.MixupImageDataGenerator(datagen, val_df, BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, subset="training")
# valid_generator = datagen.flow_from_dataframe(
#     dataframe=val_df,
#     directory=TRAIN_FOLDER,
#     x_col="image_name",
#     y_col="label",
#     has_ext=True,
#     # subset="validation",
#     batch_size=BATCH_SIZE,
#     seed=32,
#     shuffle=True,
#     class_mode="categorical",
#     target_size=(IMG_HEIGHT, IMG_WIDTH))

# have a look at the generated images
# x_batch, y_batch = next(valid_generator)
# for i in range (0, BATCH_SIZE):
#     plt.imshow(x_batch[i])
#     print(y_batch[i])
#     plt.show()


STEP_SIZE_TRAIN = int(np.ceil(len(train_df)/BATCH_SIZE))
STEP_SIZE_VALID = int(np.ceil(len(val_df)/BATCH_SIZE))

def get_model():
    import model_factory as mf

    # bm = mf.TLModel((IMG_HEIGHT, IMG_WIDTH, CHANNELS), N_CLASSES)
    bm = SeNetModel((IMG_HEIGHT, IMG_WIDTH, CHANNELS), N_CLASSES)
    model = bm.get_model()
    print('Loaded model.')

    return model

def train_model(model):
    print('Training Model:')

    # set up callbacks
    model_filepath = 'model.{epoch:02d}-{acc:.4f}-{loss:.4f}-{val_acc:.4f}-{val_loss:.4f}.hdf5'
    chkpoint = keras.callbacks.ModelCheckpoint(model_filepath, monitor='val_loss', verbose=1
        , save_best_only=True, save_weights_only=False, mode='auto', period=1)
    es = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.002, patience=5 # pylint: disable=unused-variable
        , verbose=1, mode='auto', baseline=None, restore_best_weights=True)
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, # pylint: disable=unused-variable
        patience=10, min_lr=0.0001, verbose=1) 

    import warmup_cosine_lr_decay_scheduler as wcos_lr_sch
    # number of warmup epochs
    warmup_epoch = 2
    # base learning rate after warmup.
    learning_rate_base = 0.001
    # total number of steps (NumEpoch * StepsPerEpoch)
    total_steps = int(EPOCHS * STEP_SIZE_TRAIN)
    # compute the number of warmup batches.
    warmup_steps = int(warmup_epoch * STEP_SIZE_TRAIN)
    # how many steps to hold the base lr
    hold_base_rate_epoch = 100
    hold_base_rate_steps = int(hold_base_rate_epoch * STEP_SIZE_TRAIN)
    # create the Learning rate scheduler.
    warm_up_lr = wcos_lr_sch.WarmUpCosineDecayScheduler(learning_rate_base=learning_rate_base,
                                            total_steps=total_steps,
                                            warmup_learning_rate=0.0001,
                                            warmup_steps=warmup_steps,
                                            hold_base_rate_steps=hold_base_rate_steps,
                                            verbose=1)

    # callback_list=[keras.callbacks.History(), chkpoint, warm_up_lr]
    callback_list=[keras.callbacks.History(), chkpoint]

    # compile model
    model.compile(optimizer=tf.train.AdamOptimizer(0.005) # optimizers.adam(lr=0.001, decay=1e-6)
            ,loss="categorical_crossentropy"
            ,metrics=["accuracy"]
            )

    model.fit_generator(generator=train_generator,
                        steps_per_epoch=STEP_SIZE_TRAIN,
                        validation_data=valid_generator,
                        validation_steps=STEP_SIZE_VALID,
                        callbacks=callback_list,
                        epochs=EPOCHS
                    )

def evaluate_model(model_file):
    print('Validating Model:')
    # model = load_model(model_file)
    # with tf.keras.utils.CustomObjectScope({'GlorotUniform': tf.keras.initializers.glorot_uniform()}):
    model = tf.keras.models.load_model(model_file)
    # compile model
    model.compile(optimizer=tf.train.AdamOptimizer(0.005) # optimizers.adam(lr=0.001, decay=1e-6)
            ,loss="categorical_crossentropy"
            ,metrics=["accuracy"]
            )
    y_pred = []
    y_true = []
    i = 0
    for X, y in valid_generator:
        if i % 10 == 0:
            print("Batch {}/{}".format(i, STEP_SIZE_VALID))
        preds = model.predict(X)
        y_pred += np.argmax(preds, axis=1).tolist()
        y_true += np.argmax(y, axis=1).tolist()
        i += 1
        if i == STEP_SIZE_VALID:
            print("Batch {}/{}".format(i, STEP_SIZE_VALID))
            break
    # y_pred = np.argmax(preds, axis=1)
    # y_true = valid_generator.classes
    target_names = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
    from sklearn.metrics import confusion_matrix
    print("Confusion matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("Labels: {}".format(target_names))



def test_model(model_filepath):
    print('Testing Model:')

    test_datagen=ImageDataGenerator(preprocessing_function=preprocess_input)
    test_generator=test_datagen.flow_from_dataframe(
        dataframe=test_df,
        directory=TEST_FOLDER,
        x_col="image_name",
        y_col=None,
        has_ext=True,
        batch_size=128,
        seed=42,
        shuffle=False,
        class_mode=None,
        target_size=(IMG_HEIGHT, IMG_WIDTH))

    model = tf.keras.models.load_model(model_filepath)
    # compile model
    model.compile(optimizer=tf.train.AdamOptimizer(0.005) # optimizers.adam(lr=0.001, decay=1e-6)
            ,loss="categorical_crossentropy"
            ,metrics=["accuracy"]
            )

    test_generator.reset()
    STEP_SIZE_TEST = 1 + test_generator.n//test_generator.batch_size
    pred=model.predict_generator(test_generator, steps=STEP_SIZE_TEST, verbose=1)
    predicted_class_indices=np.argmax(pred,axis=1)

    # labels = (train_generator.class_indices)
    # labels = dict((v,k) for k,v in labels.items())
    # predictions = [labels[k] for k in predicted_class_indices]

    filenames=test_generator.filenames
    results=pd.DataFrame({"image_name":filenames,
                        "label":predicted_class_indices})
    results.to_csv("results.csv",index=False)



if __name__ == '__main__':
    model = get_model()
    # train_model(model)
    evaluate_model('model.476-0.8184-0.5896-0.8582-0.4726.hdf5')
    test_model('model.476-0.8184-0.5896-0.8582-0.4726.hdf5')
    # test_model('model.197-0.8690-0.4002-0.8579-0.4002.hdf5')
