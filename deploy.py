import tensorflow as tf
import numpy as np
from PIL import Image
import os

from model import ConvWTA

from util import plot_dictionary, plot_reconstruction

import sys
import util

restoreDir = 'train_'+sys.argv[1][4:-4]+'/'

with open(sys.argv[1]) as f:
    parameters = f.readline()

parameters = parameters.strip()
if parameters.endswith('\n'):
    parameters = parameters[:-2]

parametersList = parameters.split(',')
for each in parametersList:
    variableName, value = each.split(':')
    variableName = variableName.strip()
    value = value.strip()
    if variableName == 'lifetime_sparsity' or variableName == 'learning_rate':
        exec(variableName+'=float('+value+')')
    else:
        exec(variableName+'=int('+value+')')

try:
    stride
except:
    stride=1
try:
    filter_size
except:
    filter_size=5
try:
    first_num_filters
except:
    first_num_filters=128
try:
    second_num_filters
except:
    second_num_filters=128
try:
    each_dim
except:
    each_dim = util.get_given_each_dim(which_data)
#print(which_data, lifetime_sparsity, learning_rate, batch_size, train_size, num_features)

data = util.read_test_data(which_data, each_dim, False)

#dict_dir = "dict"
#if not os.path.isdir(dict_dir):
#  os.makedirs(dict_dir)
#recon_dir = "recon"
#if not os.path.isdir(recon_dir):
#  os.makedirs(recon_dir)

sess = tf.Session()
ae = ConvWTA(sess, num_features=num_features,stride=stride,filter_size=filter_size,first_num_filters=first_num_filters,second_num_filters=second_num_filters)#learning_rate
try:
    ae.restore(restoreDir+"modelFinished.ckpt")
except:
    loaded = False
else:
    loaded = True

numEpoch = 99
while not loaded:
    try:
        ae.restore(restoreDir+"model{}.ckpt".format(numEpoch))
    except:
        numEpoch -= 1
    else:
        loaded = True
    if numEpoch < 0:
        print("model files do not exist")
        sys.exit()


# Data read & train
#from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets("mnist/", one_hot=True)

# Save deconv kernels as images.
f = ae.features()
dictionary = []
for idx in range(f.shape[-1]):
  dictionary.append(f[:,:,0,idx])
plot_dictionary(dictionary, dictionary[0].shape, num_shown=int(np.sqrt(num_features))**2, row_length=int(np.sqrt(num_features)))

# Save recon images
x = tf.placeholder(tf.float32, [1, each_dim, each_dim, 1])
y = ae.reconstruct(x)

decoded = []
for i in range(20):
  image = data[i, :]
  image = image.reshape([1, each_dim, each_dim, 1])
  result = sess.run(y, {x:image})
  decoded.append(result[0,:,:,0])
plot_reconstruction(data[:20,:], decoded, decoded[0].shape, 20)
