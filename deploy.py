import tensorflow as tf
import numpy as np
from PIL import Image
import os

from model import ConvWTA

from util import plot_dictionary, plot_reconstruction

dict_dir = "dict"
if not os.path.isdir(dict_dir):
  os.makedirs(dict_dir)
recon_dir = "recon"
if not os.path.isdir(recon_dir):
  os.makedirs(recon_dir)

sess = tf.Session()
ae = ConvWTA(sess, num_features=16)
ae.restore("ckpt/model.ckpt")

# Data read & train
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("mnist/", one_hot=True)

# Save deconv kernels as images.
f = ae.features()
dictionary = []
for idx in range(f.shape[-1]):
  dictionary.append(f[:,:,0,idx])
plot_dictionary(dictionary, dictionary[0].shape, num_shown=16, row_length=4)

# Save recon images
x = tf.placeholder(tf.float32, [1, 28, 28, 1])
y = ae.reconstruct(x)

decoded = []
for i in range(20):
  image = mnist.test.images[i, :]
  image = image.reshape([1, 28, 28, 1])
  result = sess.run(y, {x:image})
  decoded.append(result[0,:,:,0])
plot_reconstruction(mnist.test.images[:20,:], decoded, decoded[0].shape, 20)
