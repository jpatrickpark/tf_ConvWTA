import tensorflow as tf
tf.set_random_seed(2017)

from model import ConvWTA

import os
import sys
from tensorflow.contrib.keras.python.keras.datasets.cifar10 import load_data
from tensorflow.examples.tutorials.mnist import input_data
import time

from util import timestamp
import numpy as np

default_dir_suffix = timestamp()

# One file to keep track of them all
with open("whichdir.txt", "a") as myfile:
    myfile.write("which_data: {}, sparsity: {}, path: {}\n".format(sys.argv[1],sys.argv[2],default_dir_suffix))

tf.app.flags.DEFINE_string('train_dir', 'train_%s' % default_dir_suffix,
                           'where to store checkpoints to (or load checkpoints from)')
tf.app.flags.DEFINE_string('log_path', 'log_%s.txt' % default_dir_suffix,
                           'where to store loss logs to (use with --write_logs)')
tf.app.flags.DEFINE_float('learning_rate', 1e-3,
                          'learning rate to use during training')
tf.app.flags.DEFINE_float('sparsity', float(sys.argv[2]),
                          'lifetime sparsity constraint to enforce')
tf.app.flags.DEFINE_integer('which_data', int(sys.argv[1]),
                           'which data to load')
tf.app.flags.DEFINE_integer('batch_size', 100,
                            'batch size to use during training')
tf.app.flags.DEFINE_integer('epochs', 100,
                            'total epoches to train')
tf.app.flags.DEFINE_integer('train_size', 55000,
                            'number of examples to use to train classifier')
tf.app.flags.DEFINE_integer('test_size', 10000,
                            'number of examples to use to test classifier')
tf.app.flags.DEFINE_boolean('write_logs', True,
                            'write log files')

FLAGS = tf.app.flags.FLAGS

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])/255

def next_batch(num, data):
    '''
    Return a total of `num` random samples and labels.
    '''
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[ i] for i in idx]

    return np.asarray(data_shuffle)

def crop_center_oneimage(img,cropx,cropy):
    y,x = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)
    return img[starty:starty+cropy,startx:startx+cropx]

def crop_center(img_array,trim_dim):
    result = []
    for img in img_array:
        result.append(crop_center_oneimage(img,trim_dim,trim_dim))
    return np.array(result)

def whiten_images(train, test):
    length_train = len(train)
    whole = np.concatenate((train,test))
    mean = whole.mean(axis=0, keepdims=True)
    whole = whole - mean
    whole_transpose = whole.transpose()
    u,s,v = np.linalg.svd(whole_transpose)
    whiten_matrix = u @ np.linalg.inv(np.diag(s)) @ u.transpose()
    result = []
    for img in whole:
        result.append((whiten_matrix @ img.transpose()).transpose())
    result = np.array(result)

    return result[:length_train], result[length_train:], mean

def cifar10_whitened(data_dir, each_dim):
    if os.path.exists(data_dir+"train_{}.p".format(each_dim)) and \
        os.path.exists(data_dir+"test_{}.p".format(each_dim)) and \
        os.path.exists(data_dir+"mean_{}.txt".format(each_dim)):
        return pickle.load( open( data_dir+"train_{}.p".format(each_dim), "rb" ) )
    (X_train, y_train), (X_test, y_test) = load_data()

    # With fully connected network, it will be too ambitious to use 32*32 color image.
    # Reduce dimension by doing grayscale.
    X_train = rgb2gray(X_train)
    X_test = rgb2gray(X_test)

    # Crop center in order to be able to train quickly
    X_train = crop_center(X_train, each_dim)
    X_test = crop_center(X_test, each_dim)

    # Flatten arrays
    X_train = X_train.reshape(X_train.shape[0],each_dim**2)
    X_test = X_test.reshape(X_test.shape[0],each_dim**2)
    y_train = y_train.reshape(y_train.shape[0])
    y_test = y_test.reshape(y_test.shape[0])

    # Whitten images
    X_train, X_test, image_mean = whiten_images(X_train,X_test)

    pickle.dump( X_train, open(data_dir+"train_{}.p".format(each_dim),"wb") )
    pickle.dump( X_test, open(data_dir+"test_{}.p".format(each_dim),"wb") )
    with open(data_dir+"mean_{}.txt", "a") as f:
        f.write(str(image_mean))

    return X_train#, X_test # X_test not used in train.py

def read_given_data(which_data, each_dim):
    # MNIST
    if which_data == 0:
        data_dir = "MNIST_data/"
        return input_data.read_data_sets(data_dir, one_hot=True)
    else:
        data_dir = "CIFAR10_whitened_data/"
        return cifar10_whitened(data_dir, each_dim)
        #fix
        #return input_data.read_data_sets(data_dir, one_hot=True)

def next_given_batch(data, batch_size, which_data):
    if which_data == 0:
        batch_x, _ = data.train.next_batch(batch_size)
        return batch_x
    else:
        return next_batch(batch_size, data)

def get_given_each_dim(which_data):
    if which_data == 0:
        return 28
    else:
        return 32

def main():
    if not os.path.isdir(FLAGS.train_dir):
        os.makedirs(FLAGS.train_dir)
    with open(FLAGS.log_path, "a") as f:
        f.write("which_data: {}, lifetime_sparsity: {}, learning_rate: {}, batch_size: {}, train_size: {}\n".format(
            FLAGS.which_data, FLAGS.sparsity, FLAGS.learning_rate, FLAGS.batch_size, FLAGS.train_size))

    each_dim = get_given_each_dim(FLAGS.which_data)
    shape = [FLAGS.batch_size, each_dim, each_dim, 1]

    # Basic tensorflow setting
    sess = tf.Session()
    ae = ConvWTA(sess)
    x = tf.placeholder(tf.float32, shape)
    loss = ae.loss(x, lifetime_sparsity=FLAGS.sparsity)

    optim = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
    train = optim.minimize(loss, var_list=ae.t_vars)

    sess.run(tf.global_variables_initializer())

    # Data read & train
    data = read_given_data(FLAGS.which_data, each_dim)

    start_time = time.time()
    for epoch in range(FLAGS.epochs):
        total_batch = int(FLAGS.train_size / FLAGS.batch_size)
        avg_loss = 0
        for i in range(total_batch):
            batch_x = next_given_batch(data, FLAGS.batch_size, FLAGS.which_data)

            batch_x = batch_x.reshape(shape)
            l, _ =  sess.run([loss, train], {x:batch_x})
            avg_loss += l / total_batch
            if i % 50 == 0:
                with open(FLAGS.log_path, "a") as f:
                    f.write("Loss : {:.9f}\n".format(avg_loss))

        ae.save(FLAGS.train_dir+"/model{}.ckpt".format(epoch))
        with open(FLAGS.log_path, "a") as f:
            f.write("Epoch : {:04d}, Loss : {:.9f}\n".format(epoch+1, avg_loss))
    with open(FLAGS.log_path, "a") as f:
        f.write("Training time : {}\n".format(time.time() - start_time))

    ae.save(FLAGS.train_dir+"/modelFinished.ckpt")

if __name__ == "__main__":
    main()
