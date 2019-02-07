"""
@author sourabhxiii
"""

import os
import pandas as pd
import numpy as np
from PIL import Image
from matplotlib import pyplot

import imgaug.augmenters as iaa
import keras
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from sklearn.model_selection import train_test_split
import keras.optimizers as optimizers

TRAIN_FOLDER = 'train-scene_classification'+os.sep+'train'
TEST_FOLDER = 'train-scene_classification'+os.sep+'test'

BATCH_SIZE = 64
IMG_HEIGHT = 64
IMG_WIDTH = 64
CHANNELS = 3
N_CLASSES = 6
EPOCHS = 100

df = pd.read_csv('train-scene_classification'+os.sep+'train.csv')
test_df = pd.read_csv('train-scene_classification'+os.sep+'test.csv')

train_df, val_df = train_test_split(df, stratify=df['label'], test_size=0.1, random_state=42)
train_df.reset_index(drop=True, inplace=True)
val_df.reset_index(drop=True, inplace=True)

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    fill_mode='wrap',
    zoom_range=0.3
    , brightness_range=(1, 1.3)
    , validation_split=0.15
    )

sample_imgs = train_df['image_name'].sample(frac=0.1)
X_sample = np.ndarray(shape=(len(sample_imgs), IMG_HEIGHT, IMG_WIDTH, CHANNELS),
            dtype=np.float32)


i = 0
for _file in sample_imgs:
    img = Image.open(os.getcwd()+os.sep+'train-scene_classification'+os.sep+'train' +os.sep+ _file)  # this is a PIL image
    # img.thumbnail((IMG_HEIGHT, IMG_WIDTH), Image.ANTIALIAS) # creates thumbnails of images
    img = img.resize((IMG_HEIGHT, IMG_WIDTH))
    # Convert to Numpy Array
    x = img_to_array(img)
    x = x/255.
    x = x.reshape((IMG_HEIGHT, IMG_WIDTH, CHANNELS))
    X_sample[i] = x
    i += 1

'''
# ----------------->>
# peek into the resized image
# create a grid of 3x3 images
for i in range(0, 9):
	pyplot.subplot(330 + 1 + i)
	pyplot.imshow(X_sample[i])
# pyplot.show()
# <<----------------
'''

# ----------------->>
# peek inside a batch of image
img=load_img(os.getcwd()+os.sep+'train-scene_classification'+os.sep+'train' +os.sep+'0.jpg')
x=img_to_array(img)
x = x.reshape((1,) + x.shape)
save_dir=os.getcwd()+os.sep+'train-scene_classification'+os.sep+'preview'
for batch in train_datagen.flow(x, batch_size=1,
                            save_to_dir=save_dir, save_prefix='cat', save_format='jpeg'):
    i += 1
    if i > 20:
        break  # otherwise the generator would loop indefinitely
# <<-----------------

# let's say X_sample is a small-ish but statistically representative sample of my data
train_datagen.fit(X_sample)

train_generator = train_datagen.flow_from_dataframe(
    dataframe=df,
    directory=TRAIN_FOLDER,
    x_col="image_name",
    y_col="label",
    has_ext=True,
    color_mode='rgb',
    subset="training",
    batch_size=BATCH_SIZE,
    seed=42,
    shuffle=True,
    class_mode="categorical",
    target_size=(IMG_HEIGHT, IMG_WIDTH))

valid_generator=train_datagen.flow_from_dataframe(
    dataframe=df,
    directory=TRAIN_FOLDER,
    x_col="image_name",
    y_col="label",
    has_ext=True,
    subset="validation",
    batch_size=BATCH_SIZE,
    seed=42,
    shuffle=True,
    class_mode="categorical",
    target_size=(IMG_HEIGHT, IMG_WIDTH))

test_datagen=ImageDataGenerator(rescale=1./255.)

test_generator=test_datagen.flow_from_dataframe(
    dataframe=test_df,
    directory=TEST_FOLDER,
    x_col="image_name",
    y_col=None,
    has_ext=True,
    batch_size=32,
    seed=42,
    shuffle=False,
    class_mode=None,
    target_size=(IMG_HEIGHT, IMG_WIDTH))


STEP_SIZE_TRAIN = 1 + train_generator.n//train_generator.batch_size
STEP_SIZE_VALID = 1 + valid_generator.n//valid_generator.batch_size

def get_model():
    import model_factory as mf

    bm = mf.BestModel((IMG_HEIGHT, IMG_WIDTH, CHANNELS), N_CLASSES)
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
        , verbose=1, mode='auto', baseline=None)
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
                                            verbose=0)

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
