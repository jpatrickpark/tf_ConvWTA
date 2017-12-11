"""
Shared training utilities.
"""

import datetime
from sklearn.datasets import fetch_olivetti_faces

import matplotlib.pyplot as plt
import sklearn.decomposition
import sklearn.manifold
from sklearn.metrics import confusion_matrix, accuracy_score
import sklearn.svm
import tensorflow as tf
import numpy as np

import pickle
import os
from tensorflow.contrib.keras.python.keras.datasets.cifar10 import load_data
from tensorflow.examples.tutorials.mnist import input_data

def get_given_each_dim(which_data):
    if which_data == 0:
        return 28
    elif which_data == 3:
        return 64
    else:#1,2
        return 32

def cifar10_whitened_test(data_dir, each_dim):
    if os.path.exists(data_dir+"train_{}.p".format(each_dim)) and \
        os.path.exists(data_dir+"test_{}.p".format(each_dim)) and \
        os.path.exists(data_dir+"mean_{}.txt".format(each_dim)):
        return pickle.load( open( data_dir+"test_{}.p".format(each_dim), "rb" ) )

def cifar10_grayscale_test(data_dir, each_dim):
    if os.path.exists(data_dir+"train_{}.p".format(each_dim)) and \
        os.path.exists(data_dir+"test_{}.p".format(each_dim)):# and \
        return pickle.load( open( data_dir+"test_{}.p".format(each_dim), "rb" ) )

def cifar10_grayscale(data_dir, each_dim):
    if os.path.exists(data_dir+"train_{}.p".format(each_dim)) and \
        os.path.exists(data_dir+"test_{}.p".format(each_dim)):# and \
        return pickle.load( open( data_dir+"train_{}.p".format(each_dim), "rb" ) )

    if not os.path.isdir(data_dir):
        os.makedirs(data_dir)

    (X_train, y_train), (X_test, y_test) = load_data()

    # Reduce dimension by doing grayscale.
    X_train = rgb2gray(X_train)
    X_test = rgb2gray(X_test)

    # Crop center in order to be able to train quickly
    X_train = crop_center(X_train, each_dim)
    X_test = crop_center(X_test, each_dim)

    pickle.dump( X_train, open(data_dir+"train_{}.p".format(each_dim),"wb") )
    pickle.dump( X_test, open(data_dir+"test_{}.p".format(each_dim),"wb") )

    return X_train#, X_test # X_test not used in train.py

def read_data_and_labels(which_data, each_dim):
    if which_data == 0:
        # MNIST
        data_dir = "MNIST_data/"
        return input_data.read_data_sets(data_dir, one_hot=False)
    elif which_data == 2: #2
        data_dir = "CIFAR10_grayscale_data/"
        (_, y_train), (_,_) = load_data()
        return cifar10_grayscale(data_dir, each_dim), y_train
    elif which_data == 3:
        temp = fetch_olivetti_faces()
        return temp["data"], temp["target"]
    else:
        return None
    if which_data == 1:
        # CIFAR10 grayscale whitened
        data_dir = "CIFAR10_whitened_data/"
        return cifar10_whitened(data_dir, each_dim)

def read_test_data(which_data, each_dim, one_hot):
    if which_data == 0:
        # MNIST
        data_dir = "MNIST_data/"
        return input_data.read_data_sets(data_dir, one_hot=one_hot).test.images
    elif which_data == 1:
        # CIFAR10 grayscale whitened
        data_dir = "CIFAR10_whitened_data/"
        return cifar10_whitened_test(data_dir, each_dim)
    elif which_data == 2:
        # CIFAR10 grayscale
        data_dir = "CIFAR10_grayscale_data/"
        return cifar10_grayscale_test(data_dir, each_dim)
    elif which_data == 3:
        return fetch_olivetti_faces()["data"]

def read_test_labels(which_data, each_dim, one_hot):
    if which_data == 0:
        # MNIST
        data_dir = "MNIST_data/"
        return input_data.read_data_sets(data_dir, one_hot=one_hot).test.labels
    elif which_data == 2:
        (_, y_train), (_,y_test) = load_data()
        return y_test

def timestamp(format='%Y_%m_%d_%H_%M_%S'):
    """Returns the current time as a string."""
    return datetime.datetime.now().strftime(format)

def plot_dictionary(dictionary, shape, num_shown=20, row_length=10, filename="save_dictionary.png"):
    """Plots the code dictionary."""
    plt.close()
    rows = num_shown / row_length
    for i, image in enumerate(dictionary[:num_shown]):
        plt.subplot(rows, row_length, i + 1)
        plt.axis('off')
        if (len(shape)>2):
            image = (image-np.min(image))/(np.max(image)-np.min(image))
            #image = np.clip(image,0,1)
        plt.imshow(image.reshape(shape), cmap=plt.cm.gray)
    #plt.savefig(filename)
    #plt.close()
    plt.show()


def plot_reconstruction(truth, reconstructed, shape, num_shown=10, filename="save_reconstruction.png"):
    """Plots reconstructed images below the ground truth images."""
    plt.close()
    for i, image in enumerate(truth[:num_shown]):
        plt.subplot(2, num_shown, i + 1)
        plt.axis('off')
        image = np.clip(image,0,1)
        plt.imshow(image.reshape(shape), cmap=plt.cm.gray)
    for i, image in enumerate(reconstructed[:num_shown]):
        plt.subplot(2, num_shown, i + num_shown + 1)
        plt.axis('off')
        if (len(shape)>2):
            image = (image-np.min(image))/(np.max(image)-np.min(image))
            #image = np.clip(image,0,1)
        plt.imshow(image.reshape(shape), cmap=plt.cm.gray)
    #plt.savefig(filename,dpi=300)
    #plt.close()
    plt.show()


def plot_tsne(X, labels, filename):
    """Plots a t-SNE visualization of the given data."""
    if X.shape[1] > 50:
        X = sklearn.decomposition.PCA(50).fit_transform(X)
    X = sklearn.manifold.TSNE(learning_rate=200).fit_transform(X)
    plt.scatter(X[:,0], X[:,1], c=labels, cmap=plt.cm.viridis)
    plt.savefig(filename)
    #plt.show()
    plt.close()


def svm_acc(X_train, y_train, X_test, y_test, C):
    """Trains and evaluates a linear SVM with the given data and C value."""
    clf = sklearn.svm.LinearSVC(C=C, random_state=1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return accuracy_score(y_test, y_pred), confusion_matrix(y_test, y_pred)


def value_to_summary(value, tag):
    """Converts a numerical value into a tf.Summary object."""
    return tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
