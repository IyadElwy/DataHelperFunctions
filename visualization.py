# imports
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf
import seaborn as sns
import random
import os

from .utils import load_and_prep_image

###############################################################################################
from matplotlib import image as mpimg


def plot_decision_boundary(model, X, y):
    """
    Plots the decision boundary created by a model predicting on X.
    This function has been adapted from two phenomenal resources:
     1. CS231n - https://cs231n.github.io/neural-networks-case-study/
     2. Made with ML basics - https://github.com/GokuMohandas/MadeWithML/blob/main/notebooks/08_Neural_Networks.ipynb
    """
    # Define the axis boundaries of the plot and create a meshgrid
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))

    # Create X values (we're going to predict on all of these)
    x_in = np.c_[
        xx.ravel(), yy.ravel()]  # stack 2D arrays together: https://numpy.org/devdocs/reference/generated/numpy.c_.html

    # Make predictions using the trained model
    y_pred = model.predict(x_in)

    # Check for multi-class
    if model.output_shape[
        -1] > 1:  # checks the final dimension of the model's output shape, if this is > (greater than) 1,
        # it's multi-class
        print("doing multiclass classification...")
        # We have to reshape our predictions to get them ready for plotting
        y_pred = np.argmax(y_pred, axis=1).reshape(xx.shape)
    else:
        print("doing binary classifcation...")
        y_pred = np.round(np.max(y_pred, axis=1)).reshape(xx.shape)

    # Plot decision boundary
    plt.contourf(xx, yy, y_pred, cmap=plt.cm.RdYlBu, alpha=0.7)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())


###############################################################################################


def plot_mi_scores(scores):
    scores = scores.sort_values(ascending=True)
    width = np.arange(len(scores))
    ticks = list(scores.index)
    plt.barh(width, scores)
    plt.yticks(width, ticks)
    plt.title("Mutual Information Scores")


###############################################################################################

def plot_correlation_heatmap(data: pd.DataFrame):
    data_corr = data.copy()

    for col in data.columns:
        if data_corr[col].dtype == 'object':
            data_corr[col] = data['smoker'].factorize()[0]

    corr = data_corr.corr()
    sns.heatmap(corr,
                mask=np.zeros_like(corr, dtype=bool),
                cmap=sns.diverging_palette(240, 10, as_cmap=True))


###############################################################################################

def plot_confusion_matrix(cm, class_names):
    df_cm = pd.DataFrame(cm, index=[i for i in class_names], columns=[i for i in class_names])

    plt.figure(figsize=(20, 12))
    sns.heatmap(df_cm, annot=True, fmt='g');


###############################################################################################


def plot_predict_random_image(model, images, true_labels, classes):
    i = random.randint(0, len(images))

    target_image = images[i]
    pred_probs = model.predict(target_image.reshape(1, 28, 28))
    pred_label = classes[pred_probs.argmax()]
    true_label = classes[true_labels[i]]

    plt.imshow(target_image, cmap='binary')

    if pred_label == true_label:
        color = 'green'
    else:
        color = 'red'

    plt.xlabel(f'Pred: {pred_label} | {(100 * tf.reduce_max(pred_probs)):.2f} probability | True: {true_label}',
               color=color,
               size=20)


###############################################################################################

def view_random_image(target_dir, target_class, return_image=False):
    target_folder = f'{target_dir}/{target_class}'
    # get random image
    random_image = random.sample(os.listdir(target_folder), 1)
    # read in the image
    img = mpimg.imread(f'{target_folder}/{random_image[0]}')
    plt.imshow(img);
    plt.title(target_class, size=25)
    plt.axis('off')
    print(f'Image shape: {img.shape}')

    if return_image:
        return img


###############################################################################################

def plot_loss_curves(history):
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']
    epochs = range(len(history.history['loss']))

    # loss
    plt.figure()
    plt.plot(epochs, loss, label='training_loss')
    plt.plot(epochs, val_loss, label='val_loss')
    plt.title('loss')
    plt.xlabel('epochs')
    plt.legend()

    # accuracy
    plt.figure()
    plt.plot(epochs, accuracy, label='training_accuracy')
    plt.plot(epochs, val_accuracy, label='val_accuracy')
    plt.title('accuracy')
    plt.xlabel('epochs')
    plt.legend()


###############################################################################################


def pred_and_plot(model, file_path, image_shape, scale, class_names):
    img = load_and_prep_image(file_path, image_shape, scale)
    pred = model.predict(tf.expand_dims(img, axis=0))
    pred_class = class_names[int(tf.round(pred))]

    plt.figure()
    plt.imshow(img)
    plt.title(f'Prediction: {pred_class}')
    plt.axis(False)


###############################################################################################


def compare_historys(original_history, new_history, initial_epochs=5):
    """
    Compares two TensorFlow model History objects.

    Args:
      original_history: History object from original model (before new_history)
      new_history: History object from continued model training (after original_history)
      initial_epochs: Number of epochs in original_history (new_history plot starts from here) 
    """

    # Get original history measurements
    acc = original_history.history["accuracy"]
    loss = original_history.history["loss"]

    val_acc = original_history.history["val_accuracy"]
    val_loss = original_history.history["val_loss"]

    # Combine original history with new history
    total_acc = acc + new_history.history["accuracy"]
    total_loss = loss + new_history.history["loss"]

    total_val_acc = val_acc + new_history.history["val_accuracy"]
    total_val_loss = val_loss + new_history.history["val_loss"]

    # Make plots
    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(total_acc, label='Training Accuracy')
    plt.plot(total_val_acc, label='Validation Accuracy')
    plt.plot([initial_epochs - 1, initial_epochs - 1],
             plt.ylim(), label='Start Fine Tuning')  # reshift plot around epochs
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(total_loss, label='Training Loss')
    plt.plot(total_val_loss, label='Validation Loss')
    plt.plot([initial_epochs - 1, initial_epochs - 1],
             plt.ylim(), label='Start Fine Tuning')  # reshift plot around epochs
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.show()
