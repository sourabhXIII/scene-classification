"""
@author sourabhxiii
"""

import os
import pandas as pd
import numpy as np
from PIL import Image
from matplotlib import pyplot

from utility import Utility as mutil
import datagen as mdgen

import keras
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from sklearn.model_selection import train_test_split
import keras.optimizers as optimizers

TEST_FOLDER = 'train-scene_classification'+os.sep+'test'

BATCH_SIZE = 32
IMG_HEIGHT = 60
IMG_WIDTH = 60
CHANNELS = 3
N_CLASSES = 6
EPOCHS = 100

df = pd.read_csv('train-scene_classification'+os.sep+'train.csv')
test_df = pd.read_csv('train-scene_classification'+os.sep+'test.csv')

# set seed
mutil.set_seeds()

# get train and val df
train_df, val_df = mutil.get_train_val_df(df, 0.1)

#training data generator 
train_generator = mdgen.data_generator(train_df, N_CLASSES, BATCH_SIZE
                    , IMG_HEIGHT, IMG_WIDTH, CHANNELS)

# validation data generator 
valid_generator = mdgen.data_generator(val_df, N_CLASSES, BATCH_SIZE
                    , IMG_HEIGHT, IMG_WIDTH, CHANNELS, is_validation_data=True)

STEP_SIZE_TRAIN = int(np.ceil(len(train_df)/BATCH_SIZE))
STEP_SIZE_VALID = int(np.ceil(len(val_df)/BATCH_SIZE))

def get_model():
    import model_factory as mf

    bm = mf.TLModel((IMG_HEIGHT, IMG_WIDTH, CHANNELS), N_CLASSES)
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
    hold_base_rate_epoch = 30
    hold_base_rate_steps = int(hold_base_rate_epoch * STEP_SIZE_TRAIN)
    # create the Learning rate scheduler.
    warm_up_lr = wcos_lr_sch.WarmUpCosineDecayScheduler(learning_rate_base=learning_rate_base,
                                            total_steps=total_steps,
                                            warmup_learning_rate=0.000001,
                                            warmup_steps=warmup_steps,
                                            hold_base_rate_steps=hold_base_rate_steps,
                                            verbose=1)

    callback_list=[keras.callbacks.History(), chkpoint, warm_up_lr]

    # compile model
    model.compile(optimizers.adam(lr=0.001, decay=1e-6)
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

    # print('Validating Model:')
    # valid_generator.reset()
    # model.evaluate_generator(generator=valid_generator, steps=STEP_SIZE_VALID)


def test_model(model_filepath):
    print('Testing Model:')
    model = load_model(model_filepath)
    test_generator.reset()
    STEP_SIZE_TEST = 1 + test_generator.n//test_generator.batch_size
    pred=model.predict_generator(test_generator, steps=STEP_SIZE_TEST, verbose=1)
    predicted_class_indices=np.argmax(pred,axis=1)

    labels = (train_generator.class_indices)
    labels = dict((v,k) for k,v in labels.items())
    predictions = [labels[k] for k in predicted_class_indices]

    filenames=test_generator.filenames
    results=pd.DataFrame({"image_name":filenames,
                        "label":predictions})
    results.to_csv("results.csv",index=False)



if __name__ == '__main__':
    model = get_model()
    train_model(model)
    # test_model('model.197-0.8690-0.4002-0.8579-0.4002.hdf5')
