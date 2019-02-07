"""
@author sourabhxiii
"""
import os
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
import tensorflow as tf
import imgaug as aug

class Utility:

    @staticmethod
    def set_seeds(self):

        # Set the seed for hash based operations in python
        os.environ['PYTHONHASHSEED'] = '0'

        seed=1234

        # Set the numpy seed
        np.random.seed(seed)

        # Set the random seed in tensorflow at graph level
        tf.set_random_seed(seed)

        # Make the augmentation sequence deterministic
        aug.seed(seed)

    @staticmethod
    def get_train_val_df(self, train_df, split=0.1):
        X=train_df['image_name']
        y=train_df['label']
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=0)

        X_train, X_val, y_train, y_val = None, None, None, None
        for train_index, test_index in sss.split(X, y):
            X_train, X_val = X[train_index], X[test_index]
            y_train, y_val = y[train_index], y[test_index]

        X_train = X_train.to_frame(name='image_name')
        X_train.reset_index(drop=True, inplace=True)
        X_val = X_val.to_frame(name='image_name')
        X_val.reset_index(drop=True, inplace=True)
        y_train = y_train.to_frame(name='label')
        y_train.reset_index(drop=True, inplace=True)
        y_val = y_val.to_frame(name='label')
        y_val.reset_index(drop=True, inplace=True)


        print("Total data length {}".format(len(train_df)))
        print("Training data length {}".format(len(X_train)))
        print("Validation data length {}".format(len(X_val)))
        return X_train, X_val, y_train, y_val