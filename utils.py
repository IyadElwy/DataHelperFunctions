# imports
import pathlib

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.feature_selection import mutual_info_regression
import os
import datetime
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler


###############################################################################################


def make_mi_scores(X, y):
    X = X.copy()
    for colname in X.select_dtypes(["object", "category"]):
        X[colname], _ = X[colname].factorize()
    # All discrete features should now have integer dtypes
    discrete_features = [pd.api.types.is_integer_dtype(t) for t in X.dtypes]
    mi_scores = mutual_info_regression(X, y, discrete_features=discrete_features, random_state=0)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores


###############################################################################################


def walk_through_data(folder_path):
    for dirpath, dirnames, filenames in os.walk(folder_path):
        print(f'There are {len(dirnames)} directories and {len(filenames)} images in {dirpath}')


###############################################################################################


def get_classes_in_data(folder_path):
    data_dir = pathlib.Path(folder_path)
    class_names = np.array(sorted([item.name for item in data_dir.glob('*')]))
    return class_names


###############################################################################################

def load_and_normalize_image(file_path, img_shape):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_image(img)
    img = tf.image.resize(img, size=[img_shape, img_shape])
    img = img / 255.
    return img


###############################################################################################


def create_save_tensorboard_callback(dir_name, experiment_name):
    log_dir = dir_name + '/' + experiment_name + '/' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
    print(f'Saving tensorboard log files to {log_dir}')
    return tensorboard_callback


###############################################################################################

def get_nan_columns(data):
    nan_columns = data.isna().sum().loc[data.isna().sum() != 0].sort_values(ascending=False)
    return nan_columns


###############################################################################################

def under_sample_data(X_train, y_train):
    rus = RandomUnderSampler(sampling_strategy=1)  # Numerical value
    X_train_t, y_train_t = rus.fit_resample(X_train, y_train)
    return X_train_t, y_train_t


###############################################################################################

def over_sample_data(X_train, y_train):
    rus = RandomOverSampler(sampling_strategy=1)  # Numerical value
    X_train_t, y_train_t = rus.fit_resample(X_train, y_train)
    return X_train_t, y_train_t

###############################################################################################
