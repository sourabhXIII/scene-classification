"""
@author sourabhxiii
"""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
import tensorflow as tf
import imgaug as aug

class Utility:

    @staticmethod
    def set_seeds():

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
    def get_train_val_df(df, split=0.1):
        X=df['image_name']
        y=df['label']
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=0)

        X_train, X_val, y_train, y_val = None, None, None, None
        for train_index, test_index in sss.split(X, y):
            X_train, X_val = X[train_index], X[test_index]
            y_train, y_val = y[train_index], y[test_index]

        train_df = pd.DataFrame({'image_name': X_train, 'label': y_train})
        train_df.reset_index(drop=True, inplace=True)
        val_df = pd.DataFrame({'image_name': X_val, 'label': y_val})
        val_df.reset_index(drop=True, inplace=True)

        print("Total data length {}".format(len(df)))
        print("Training data length {}".format(len(train_df)))
        print("Validation data length {}".format(len(val_df)))
        return train_df, val_df