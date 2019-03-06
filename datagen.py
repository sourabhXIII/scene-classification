"""
@author sourabhxiii
"""
import cv2
import os
import numpy as np
import imgaug.augmenters as iaa

from keras.utils import to_categorical
from keras.applications.vgg16 import preprocess_input


TRAIN_FOLDER = 'train-scene_classification'+os.sep+'train'

# Augmentation sequence 
# seq = iaa.OneOf([
# iaa.Fliplr(), # horizontal flips
# iaa.Affine(rotate=20), # rotation
# iaa.Multiply((1.2, 1.5))]) # random brightness

seq = iaa.Sequential([
    iaa.Fliplr(0.5), # horizontal flips
    iaa.Crop(percent=(0, 0.1)), # random crops
    # Small gaussian blur with random sigma between 0 and 0.5.
    # But we only blur about 50% of all images.
    iaa.Sometimes(0.5,
        iaa.GaussianBlur(sigma=(0, 0.5))
    ),
    # Strengthen or weaken the contrast in each image.
    iaa.ContrastNormalization((0.75, 1.5)),
    # Add gaussian noise.
    # For 50% of all images, we sample the noise once per pixel.
    # For the other 50% of all images, we sample the noise per pixel AND
    # channel. This can change the color (not only brightness) of the
    # pixels.
    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
    # Make some images brighter and some darker.
    # In 20% of all cases, we sample the multiplier once per channel,
    # which can end up changing the color of the images.
    iaa.Multiply((0.8, 1.2), per_channel=0.2),
    # Apply affine transformations to each image.
    # Scale/zoom them, translate/move them, rotate them and shear them.
    iaa.Affine(
        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
        rotate=(-25, 25),
        shear=(-8, 8)
    )
], random_order=True) # apply augmenters in random order


def data_generator(data, nb_classes, batch_size, img_rows, img_cols, img_channels, is_validation_data=False):
    # Get total number of samples in the data
    n = len(data)
    nb_batches = int(np.ceil(n/batch_size))

    # Get a numpy array of all the indices of the input data
    indices = np.arange(n)
    
    # Define two numpy arrays for containing batch data and labels
    batch_data = np.zeros((batch_size, img_rows, img_cols, img_channels), dtype=np.float32)
    batch_labels = np.zeros((batch_size, nb_classes), dtype=np.float32)
    
    while True:
        if not is_validation_data:
            # shuffle indices for the training data
            np.random.shuffle(indices)
            
        for i in range(nb_batches):
            # get the next batch 
            next_batch_indices = indices[i*batch_size:(i+1)*batch_size]
            
            # process the next batch
            for j, idx in enumerate(next_batch_indices):
                img = cv2.imread(TRAIN_FOLDER+os.sep+data.iloc[idx]["image_name"])
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # img = img/255. # do not need this when there's preprocess_input
                label = data.iloc[idx]["label"]
                
                if not is_validation_data:
                    img = seq.augment_image(img)
                
                img = cv2.resize(img, (img_rows, img_cols)).astype(np.float32)
                batch_data[j] = img
                batch_labels[j] = to_categorical(label,num_classes=nb_classes)

            # in case of transfer learning from VGG
            batch_data = preprocess_input(batch_data)
            yield batch_data, batch_labels



class MixupImageDataGenerator():
    def __init__(self, generator, directory, batch_size, img_height, img_width, alpha=0.5, subset=None):
        """Constructor for mixup image data generator.
        Arguments:
            generator {object} -- An instance of Keras ImageDataGenerator.
            directory {str} -- Image directory.
            batch_size {int} -- Batch size.
            img_height {int} -- Image height in pixels.
            img_width {int} -- Image width in pixels.
        Keyword Arguments:
            alpha {float} -- Mixup beta distribution alpha parameter. (default: {0.2})
            subset {str} -- 'training' or 'validation' if validation_split is specified in
            `generator` (ImageDataGenerator).(default: {None})
        """

        self.batch_index = 0
        self.batch_size = batch_size
        self.alpha = alpha

        # First iterator yielding tuples of (x, y)
        # self.generator1 = generator.flow_from_directory(directory,
        #                                                 target_size=(
        #                                                     img_height, img_width),
        #                                                 class_mode="categorical",
        #                                                 batch_size=batch_size,
        #                                                 shuffle=True,
        #                                                 subset=subset)
        self.generator1 = generator.flow_from_dataframe(dataframe=directory,
                                                        directory=TRAIN_FOLDER,
                                                        x_col="image_name",
                                                        y_col="label",
                                                        has_ext=True,
                                                        color_mode='rgb',
                                                        subset=subset,
                                                        batch_size=batch_size,
                                                        seed=42,
                                                        shuffle=True,
                                                        class_mode="categorical",
                                                        target_size=(img_height, img_width))
        # Second iterator yielding tuples of (x, y)
        # self.generator2 = generator.flow_from_directory(directory,
        #                                                 target_size=(
        #                                                     img_height, img_width),
        #                                                 class_mode="categorical",
        #                                                 batch_size=batch_size,
        #                                                 shuffle=True,
        #                                                 subset=subset)
        self.generator2 = generator.flow_from_dataframe(dataframe=directory,
                                                        directory=TRAIN_FOLDER,
                                                        x_col="image_name",
                                                        y_col="label",
                                                        has_ext=True,
                                                        color_mode='rgb',
                                                        subset=subset,
                                                        batch_size=batch_size,
                                                        seed=12,
                                                        shuffle=True,
                                                        class_mode="categorical",
                                                        target_size=(img_height, img_width))
        # Number of images across all classes in image directory.
        self.n = self.generator1.samples

    def reset_index(self):
        """Reset the generator indexes array.
        """

        self.generator1._set_index_array()
        self.generator2._set_index_array()

    def on_epoch_end(self):
        self.reset_index()

    def reset(self):
        self.batch_index = 0

    def __len__(self):
        # round up
        return (self.n + self.batch_size - 1) // self.batch_size

    def get_steps_per_epoch(self):
        """Get number of steps per epoch based on batch size and
        number of images.
        Returns:
            int -- steps per epoch.
        """

        return self.n // self.batch_size

    def __next__(self):
        """Get next batch input/output pair.
        Returns:
            tuple -- batch of input/output pair, (inputs, outputs).
        """

        if self.batch_index == 0:
            self.reset_index()

        current_index = (self.batch_index * self.batch_size) % self.n
        if self.n > current_index + self.batch_size:
            self.batch_index += 1
        else:
            self.batch_index = 0

        # Get a pair of inputs and outputs from two iterators.
        X1, y1 = self.generator1.next()
        X2, y2 = self.generator2.next()

        n_elem_in_batch = X1.shape[0] # batch_size is not good, think of the last batch.
        # random sample the lambda value from beta distribution.
        l = np.random.beta(self.alpha, self.alpha, n_elem_in_batch)

        X_l = l.reshape(n_elem_in_batch, 1, 1, 1)
        y_l = l.reshape(n_elem_in_batch, 1)

        # Perform the mixup.
        X = X1 * X_l + X2 * (1 - X_l)
        y = y1 * y_l + y2 * (1 - y_l)
        return X, y

    def __iter__(self):
        while True:
            yield next(self)