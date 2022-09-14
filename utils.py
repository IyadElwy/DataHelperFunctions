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
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


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

def load_and_prep_image(filename, img_shape=224, scale=True):
    """
    Reads in an image from filename, turns it into a tensor and reshapes into
    (224, 224, 3).
    Parameters
    ----------
    filename (str): string filename of target image
    img_shape (int): size to resize target image to, default 224
    scale (bool): whether to scale pixel values to range(0, 1), default True
    """
    # Read in the image
    img = tf.io.read_file(filename)
    # Decode it into a tensor
    img = tf.image.decode_jpeg(img)
    # Resize the image
    img = tf.image.resize(img, [img_shape, img_shape])
    if scale:
        # Rescale the image (get all values between 0 and 1)
        return img / 255.
    else:
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


def calculate_results(y_true, y_pred):
    """
    Calculates model accuracy, precision, recall and f1 score of a binary classification model.
    Args:
        y_true: true labels in the form of a 1D array
        y_pred: predicted labels in the form of a 1D array
    Returns a dictionary of accuracy, precision, recall, f1-score.
    """
    # Calculate model accuracy
    model_accuracy = accuracy_score(y_true, y_pred) * 100
    # Calculate model precision, recall and f1 score using "weighted average
    model_precision, model_recall, model_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")
    model_results = {"accuracy": model_accuracy,
                     "precision": model_precision,
                     "recall": model_recall,
                     "f1": model_f1}
    return model_results

###############################################################################################
